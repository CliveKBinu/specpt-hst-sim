import argparse
import os
import yaml
import torch
import wandb
import numpy as np
import zipfile
from pathlib import Path

from ..model import SpecPT, EnhancedSpecPTForRedshift, EnhancedSpecPTForRedshiftMDN
from ..losses import NMADLoss, HuberNMADLoss, MDNMADLoss, BinnedRedshiftLoss
from ..dataloader import load_grism_data, split_data, create_dataloaders
from .eval import compute_metrics, predict_with_tta, decode_redshift_output


def load_config(config_path):
    base_dir = Path(config_path).parent
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    parent = cfg.get("parent")
    if parent:
        parent_path = base_dir / f"{parent}.yaml"
        if parent_path.exists():
            with open(parent_path, "r") as f:
                base = yaml.safe_load(f)
            merged = deep_merge(base, cfg)
            merged.pop("parent", None)
            merged.pop("changes", None)
            return merged
    return cfg


def deep_merge(base, override):
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, best_val_loss, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler=None, device="cpu"):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except (RuntimeError, zipfile.BadZipFile, EOFError) as e:
        print(f"Warning: Corrupted checkpoint at {path}: {e}")
        print("  Skipping checkpoint load — model will use current weights")
        return None, None, None, None
    
    model_state = dict(checkpoint["model_state_dict"])
    # Bin edges are deterministic config state; do not overwrite them on resume.
    if getattr(model, "binned_output", False):
        model_state.pop("prediction.bin_left_log1p", None)
    
    # Try loading with strict=False to handle architecture mismatches
    # (e.g., loading point-head checkpoint into MDN model)
    try:
        missing, unexpected = model.load_state_dict(model_state, strict=False)
    except RuntimeError as e:
        print(f"Checkpoint tensor-shape mismatch — loading compatible keys only: {e}")
        compatible = {
            key: value for key, value in model_state.items()
            if key in model.state_dict() and model.state_dict()[key].shape == value.shape
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(f"  Loaded {len(compatible)} compatible keys (head reset where needed)")
        return checkpoint["epoch"], checkpoint["train_losses"], checkpoint["val_losses"], checkpoint["best_val_loss"]
    if missing or unexpected:
        print(f"Checkpoint key mismatch — missing {len(missing)}, unexpected {len(unexpected)}")
        if missing:
            print(f"  Missing: {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected: {unexpected[:5]}...")
        # Only load shared keys (backbone weights)
        shared_keys = {k: v for k, v in model_state.items() if k in model.state_dict()}
        model.load_state_dict(shared_keys, strict=False)
        print(f"  Loaded {len(shared_keys)} shared keys (backbone only)")
        # Skip optimizer/scheduler loading — parameter groups won't match
        print("  Skipping optimizer/scheduler loading (architecture mismatch)")
        return checkpoint["epoch"], checkpoint["train_losses"], checkpoint["val_losses"], checkpoint["best_val_loss"]
    
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["train_losses"], checkpoint["val_losses"], checkpoint["best_val_loss"]


def compute_redshift_weights(train_loader, model, device, bins=None, prediction_type="point"):
    """Compute per-sample weights based on inverse error frequency by redshift bin.

    Args:
        train_loader: DataLoader for training set.
        model: Trained model for error estimation.
        device: torch device.
        bins: Redshift bin edges.

    Returns:
        weights: np.array of per-sample weights (shape: [n_samples]).
        bin_edges: The bins used.
    """
    if bins is None:
        bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0]

    model.eval()
    all_errors = []
    all_z_true = []

    with torch.no_grad():
        for X, Y, idx, t_id in train_loader:
            X, Y = X.to(device), Y.to(device)
            preds = decode_redshift_output(model(X), prediction_type)
            delz = torch.abs((preds - Y.flatten()) / (1 + Y.flatten()))
            all_errors.append(delz.cpu().numpy())
            all_z_true.append(Y.cpu().numpy().flatten())

    all_errors = np.concatenate(all_errors)
    all_z_true = np.concatenate(all_z_true)

    # Compute mean error per bin
    bin_errors = []
    for i in range(len(bins) - 1):
        mask = (all_z_true >= bins[i]) & (all_z_true < bins[i + 1])
        if np.sum(mask) > 0:
            bin_errors.append(np.mean(all_errors[mask]))
        else:
            bin_errors.append(0.0)

    # Inverse error weighting (higher error = higher weight)
    bin_errors = np.array(bin_errors)
    bin_errors = np.maximum(bin_errors, 1e-6)  # Avoid division by zero
    bin_weights = 1.0 / bin_errors
    bin_weights = bin_weights / bin_weights.sum()  # Normalize to sum to 1

    # Assign weights to samples
    sample_weights = np.ones(len(all_z_true))
    for i in range(len(bins) - 1):
        mask = (all_z_true >= bins[i]) & (all_z_true < bins[i + 1])
        sample_weights[mask] = bin_weights[i] * len(bins)  # Scale by number of bins

    print(f"Redshift bin weights: {dict(zip([f'[{bins[i]:.1f},{bins[i+1]:.1f})' for i in range(len(bins)-1)], bin_weights))}")

    return sample_weights, bins


def compute_curriculum_weights(
    train_loader, model, device, start_pct=0.5, ramp_epochs=100, epoch=0,
    prediction_type="point",
):
    """Compute curriculum weights that gradually increase dataset difficulty.

    Args:
        train_loader: DataLoader for training set.
        model: Model for difficulty estimation.
        device: torch device.
        start_pct: Starting percentage of easiest samples.
        ramp_epochs: Number of epochs to ramp from start_pct to 100%.
        epoch: Current epoch.

    Returns:
        sample_mask: Boolean mask of which samples to use.
    """
    # Linearly increase from start_pct to 1.0 over ramp_epochs
    progress = min(epoch / ramp_epochs, 1.0)
    current_pct = start_pct + (1.0 - start_pct) * progress

    model.eval()
    all_errors = []

    with torch.no_grad():
        for X, Y, idx, t_id in train_loader:
            X, Y = X.to(device), Y.to(device)
            preds = decode_redshift_output(model(X), prediction_type)
            delz = torch.abs((preds - Y.flatten()) / (1 + Y.flatten()))
            all_errors.append(delz.cpu().numpy())

    all_errors = np.concatenate(all_errors)

    # Select easiest samples based on current percentage
    threshold = np.percentile(all_errors, current_pct * 100)
    sample_mask = all_errors <= threshold

    print(f"Curriculum: epoch {epoch}, using {current_pct:.1%} of samples "
          f"({np.sum(sample_mask)}/{len(sample_mask)} samples, threshold={threshold:.4f})")

    return sample_mask


def main():
    parser = argparse.ArgumentParser(description="SpecPT Redshift Training")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--binned-output", action="store_true")
    parser.add_argument("--num-z-bins", type=int, default=None)
    parser.add_argument("--z-bin-max", type=float, default=None)
    parser.add_argument("--lambda-refine", type=float, default=None)
    parser.add_argument("--lambda-nmad", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    wandb_cfg = cfg.get("wandb", {})

    # Derive experiment name from config filename (e.g., configs/exp_013.yaml → exp_013)
    exp_name = os.path.splitext(os.path.basename(args.config))[0]
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_entity = args.wandb_entity or wandb_cfg.get("entity")
    wandb_project = args.wandb_project or wandb_cfg.get("project")
    cfg["experiment_name"] = exp_name
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config=cfg,
    )

    auto_model = SpecPT(
        input_size=model_cfg["input_size"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_encoder_layers=model_cfg["num_encoder_layers"],
        num_decoder_layers=model_cfg["num_decoder_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
    )
    auto_model = auto_model.to(device)

    ae_path = os.path.expanduser(data_cfg.get("pretrained_autoencoder", ""))
    if ae_path and os.path.exists(ae_path):
        ae_state = torch.load(ae_path, map_location=device, weights_only=False)
        if isinstance(ae_state, dict) and "model_state_dict" in ae_state:
            ae_state = ae_state["model_state_dict"]
        missing, unexpected = auto_model.load_state_dict(ae_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Autoencoder checkpoint {ae_path} does not match the configured "
                f"architecture (missing {len(missing)}, unexpected {len(unexpected)}). "
                f"Check d_model/nhead/layers/dim_feedforward against the checkpoint."
            )
        print(f"Loaded autoencoder from {ae_path}")
    else:
        print("No pretrained autoencoder — initializing AE from scratch.")

    # Check for alternate prediction types. Point regression remains the default.
    prediction_type = model_cfg.get("prediction_type", "point")
    num_mixtures = model_cfg.get("num_mixtures", 5)
    binned_output = args.binned_output or prediction_type == "binned"
    if binned_output:
        prediction_type = "binned"
    num_z_bins = args.num_z_bins if args.num_z_bins is not None else model_cfg.get("num_z_bins", 24)
    z_bin_max = args.z_bin_max if args.z_bin_max is not None else model_cfg.get("z_bin_max", 3.0)

    if prediction_type == "mdn":
        redshift_model = EnhancedSpecPTForRedshiftMDN(
            auto_model,
            output_features=1,
            num_mlp_blocks=model_cfg["num_mlp_blocks"],
            mlp_dim=model_cfg["mlp_dim"],
            dropout_rate=model_cfg.get("dropout_rate", model_cfg.get("dropout", 0.1)),
            num_mixtures=num_mixtures,
        )
        print(f"Using MDN head with {num_mixtures} mixtures")
    else:
        redshift_model = EnhancedSpecPTForRedshift(
            auto_model,
            output_features=1,
            num_mlp_blocks=model_cfg["num_mlp_blocks"],
            mlp_dim=model_cfg["mlp_dim"],
            dropout_rate=model_cfg.get("dropout_rate", model_cfg.get("dropout", 0.1)),
            binned_output=binned_output,
            num_z_bins=num_z_bins,
            z_bin_max=z_bin_max,
        )
        if binned_output:
            print(f"Using binned redshift head with {num_z_bins} bins over z<= {z_bin_max}")

    rz_path = os.path.expanduser(data_cfg.get("pretrained_redshift", "") or "")
    if rz_path and not binned_output:
        state_dict = torch.load(rz_path, map_location=device)
        autoencoder_prefixes = ("encoder.", "proj_to_d_model.", "pretrained_model.")
        head_only = {k: v for k, v in state_dict.items() if not k.startswith(autoencoder_prefixes)}
        redshift_model.load_state_dict(head_only, strict=False)
    elif rz_path and binned_output:
        print("Binned head enabled — skipping incompatible pretrained redshift head weights.")
    else:
        print("No pretrained redshift head weights — initializing head from scratch.")

    freeze_backbone = train_cfg.get("freeze_backbone", True)
    if freeze_backbone:
        for param in redshift_model.pretrained_model.parameters():
            param.requires_grad = False
        print("Backbone FROZEN — training head only")
    else:
        for param in redshift_model.parameters():
            param.requires_grad = True
        print("Backbone UNFROZEN — end-to-end training")
        total_trainable = sum(p.numel() for p in redshift_model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total_trainable:,}")

    def pin_frozen_bn_eval():
        if freeze_backbone:
            for m in redshift_model.pretrained_model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()

    redshift_model.to(device)

    data = load_grism_data(data_cfg["path"])
    split_by_group = data_cfg.get("split_by_group", False)
    group_col = data_cfg.get("group_column", "TARGETID") if split_by_group else None
    train_df, val_df, test_df = split_data(
        data,
        val_split=data_cfg.get("val_split", 0.1),
        test_split=data_cfg.get("test_split", 0.1),
        group_col=group_col,
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
          + (f" (grouped by {group_col})" if group_col else ""))

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        batch_size=train_cfg["batch_size"],
        num_workers=0,
    )
    os.makedirs("outputs/plots", exist_ok=True)

    loss_type = train_cfg.get("loss", "nmad")
    loss_delta = float(train_cfg.get("loss_delta", 0.15))
    
    lambda_refine = args.lambda_refine if args.lambda_refine is not None else train_cfg.get("lambda_refine", 0.3)
    lambda_nmad = args.lambda_nmad if args.lambda_nmad is not None else train_cfg.get("lambda_nmad", 0.7)
    label_smoothing = args.label_smoothing if args.label_smoothing is not None else train_cfg.get("label_smoothing", 0.05)

    # Structured-head losses take precedence over scalar regression losses.
    if prediction_type == "binned":
        criterion = BinnedRedshiftLoss(
            num_bins=num_z_bins,
            z_bin_max=z_bin_max,
            lambda_refine=lambda_refine,
            lambda_nmad=lambda_nmad,
            label_smoothing=label_smoothing,
        )
        print("Using BinnedRedshiftLoss")
    elif prediction_type == "mdn":
        criterion = MDNMADLoss(normalization_factor="std")
        print("Using MDNMADLoss for MDN prediction")
    elif loss_type == "huber_nmad":
        criterion = HuberNMADLoss(delta=loss_delta, normalization_factor="std")
        print(f"Using HuberNMADLoss with delta={loss_delta}")
    else:
        criterion = NMADLoss(normalization_factor="std")
        print("Using NMADLoss")
    optimizer = torch.optim.AdamW(
        redshift_model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0)),
    )

    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=train_cfg["epochs"] * len(train_loader),
    )

    wandb.watch(redshift_model, criterion=criterion, log="gradients")

    # ========== SAMPLE WEIGHTING (redshift bin inverse error) ==========
    sample_weighting = train_cfg.get("sample_weighting", None)
    weight_bins = train_cfg.get("weight_bins", [0, 0.5, 1.0, 1.5, 2.0, 3.0])
    sample_weights = None

    if sample_weighting == "redshift_inverse_error":
        print("\nComputing redshift-based sample weights...")
        sample_weights, _ = compute_redshift_weights(
            train_loader, redshift_model, device, bins=weight_bins,
            prediction_type=prediction_type,
        )
        sample_weights = torch.from_numpy(sample_weights).float().to(device)
        print(f"Sample weights computed: min={sample_weights.min():.3f}, max={sample_weights.max():.3f}")

    start_epoch = 0
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    patience_counter = 0
    patience = train_cfg["patience"]

    if args.resume:
        start_epoch, train_losses, val_losses, best_val_loss = load_checkpoint(
            args.resume, redshift_model, optimizer, scheduler, device
        )
        start_epoch += 1

    # ========== CURRICULUM LEARNING ==========
    curriculum = train_cfg.get("curriculum", False)
    curriculum_start_pct = float(train_cfg.get("curriculum_start_pct", 0.5))
    curriculum_ramp_epochs = train_cfg.get("curriculum_ramp_epochs", 100)

    for epoch in range(start_epoch, train_cfg["epochs"]):
        redshift_model.train()
        pin_frozen_bn_eval()
        loss_epoch = 0.0
        batch_count = 0

        # Compute curriculum mask if enabled
        curriculum_mask = None
        if curriculum and epoch < curriculum_ramp_epochs:
            curriculum_mask = compute_curriculum_weights(
                train_loader, redshift_model, device,
                start_pct=curriculum_start_pct,
                ramp_epochs=curriculum_ramp_epochs,
                epoch=epoch,
                prediction_type=prediction_type,
            )

        for X, Y, idx, t_id in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            
            outputs = redshift_model(X)
            if prediction_type == "mdn":
                loss = criterion(*outputs, Y)
            else:
                loss = criterion(outputs, Y)
            if isinstance(loss, dict):
                loss = loss["total"]

            # Apply curriculum mask if available
            if curriculum_mask is not None:
                # Get mask for current batch indices and move to device
                batch_mask = torch.from_numpy(curriculum_mask[idx]).float().to(device)
                if batch_mask.sum() == 0:
                    continue  # Skip batch if no samples selected
                loss = (loss * batch_mask).sum() / batch_mask.sum()

            # Apply sample weights if available
            elif sample_weights is not None:
                # idx contains the original indices in the dataset
                weights = sample_weights[idx]
                loss = (loss * weights).mean()

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(redshift_model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_epoch += loss.item()
            batch_count += 1

        scheduler.step()
        train_loss = loss_epoch / max(batch_count, 1)
        train_losses.append(train_loss)

        redshift_model.eval()
        val_loss_epoch = 0.0
        val_count = 0
        all_preds = []
        all_true = []
        val_component_sums = {"loss_cls": 0.0, "loss_refine": 0.0, "loss_nmad": 0.0}

        with torch.no_grad():
            for X, Y, idx, t_id in val_loader:
                X, Y = X.to(device), Y.to(device)
                
                outputs = redshift_model(X)
                eval_mask = (Y.squeeze(-1) <= z_bin_max) if binned_output else None
                if eval_mask is not None and not eval_mask.any():
                    continue
                loss_outputs = outputs
                loss_target = Y
                if eval_mask is not None:
                    loss_target = Y[eval_mask]
                    if isinstance(outputs, dict):
                        loss_outputs = {key: value[eval_mask] for key, value in outputs.items()}
                    elif isinstance(outputs, tuple):
                        loss_outputs = tuple(value[eval_mask] for value in outputs)
                if prediction_type == "mdn":
                    val_loss_batch = criterion(*loss_outputs, loss_target)
                else:
                    val_loss_batch = criterion(loss_outputs, loss_target)
                loss_components = val_loss_batch if isinstance(val_loss_batch, dict) else None
                if loss_components is not None:
                    val_loss_batch = loss_components["total"]
                if torch.isnan(val_loss_batch) or torch.isinf(val_loss_batch):
                    continue
                if loss_components is not None:
                    for key in val_component_sums:
                        val_component_sums[key] += loss_components[key].item()
                preds = decode_redshift_output(outputs, prediction_type)
                if eval_mask is not None:
                    preds = preds[eval_mask]
                val_loss_epoch += val_loss_batch.item()
                val_count += 1
                all_preds.append(preds.cpu().numpy().flatten())
                all_true.append((Y[eval_mask] if eval_mask is not None else Y).cpu().numpy().flatten())

        val_loss = val_loss_epoch / max(val_count, 1)
        val_losses.append(val_loss)

        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)
        if binned_output:
            eval_mask = all_true <= z_bin_max
            all_preds = all_preds[eval_mask]
            all_true = all_true[eval_mask]
        metrics = compute_metrics(all_true, all_preds)

        log_data = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_nmad": metrics["nmad"],
                "val_z_bias": metrics["bias"],
                "catastrophic_outliers": metrics["eta"],
                "val_rmse": metrics["rmse"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }
        if prediction_type == "binned" and val_count:
            for key, value in val_component_sums.items():
                log_data[f"val_{key}"] = value / val_count
        wandb.log(log_data)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                redshift_model, optimizer, scheduler, epoch,
                train_losses, val_losses, best_val_loss,
                f"checkpoints/{exp_name}_best_model.pth",
            )
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                redshift_model, optimizer, scheduler, epoch,
                train_losses, val_losses, best_val_loss,
                "checkpoints/latest_checkpoint.pth",
            )

        print(
            f"Epoch {epoch+1}/{train_cfg['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"NMAD: {metrics['nmad']:.4f} | Bias: {metrics['bias']:.4f} | "
            f"η: {metrics['eta']:.2f}%"
        )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # ========== TWO-STAGE TRAINING (Stage 2) ==========
    two_stage = train_cfg.get("two_stage", False)
    if two_stage:
        stage2_epochs = train_cfg.get("stage2_epochs", 200)
        outlier_weight = float(train_cfg.get("outlier_weight", 4.0))
        outlier_threshold = float(train_cfg.get("outlier_threshold", 0.15))
        stage2_patience = train_cfg.get("stage2_patience", train_cfg["patience"])

        print("\n" + "="*60)
        print(f"STAGE 2: Outlier Re-weighting (weight={outlier_weight}, threshold={outlier_threshold})")
        print("="*60)

        # Load best Stage 1 checkpoint
        best_ckpt_path = f"checkpoints/{exp_name}_best_model.pth"
        if os.path.exists(best_ckpt_path):
            load_checkpoint(best_ckpt_path, redshift_model, optimizer, scheduler, device)
            print(f"Loaded best Stage 1 checkpoint from {best_ckpt_path}")

        # Identify outliers on validation set
        redshift_model.eval()
        val_errors = []
        val_indices = []
        with torch.no_grad():
            for X, Y, idx, t_id in val_loader:
                X, Y = X.to(device), Y.to(device)
                preds = decode_redshift_output(redshift_model(X), prediction_type)
                delz = torch.abs((preds - Y.flatten()) / (1 + Y.flatten()))
                val_errors.append(delz.cpu().numpy())
                val_indices.append(idx.numpy())

        val_errors = np.concatenate(val_errors)
        val_indices = np.concatenate(val_indices)
        n_outliers = np.sum(val_errors > outlier_threshold)
        print(f"Identified {n_outliers}/{len(val_errors)} outliers in validation set")

        # Create weighted sampler for training set
        # Compute per-sample weights based on error from Stage 1 model
        redshift_model.eval()
        train_errors = np.zeros(len(train_loader.dataset), dtype=np.float32)
        with torch.no_grad():
            for X, Y, idx, t_id in train_loader:
                X, Y = X.to(device), Y.to(device)
                preds = decode_redshift_output(redshift_model(X), prediction_type)
                delz = torch.abs((preds - Y.flatten()) / (1 + Y.flatten()))
                train_errors[idx.numpy()] = delz.cpu().numpy()

        # Weight: 1.0 for normal samples, outlier_weight for outliers
        sample_weights = np.where(
            train_errors > outlier_threshold, outlier_weight, 1.0
        )
        print(f"Outlier weights applied: {np.sum(sample_weights > 1)}/{len(sample_weights)} samples weighted {outlier_weight}x")

        # Create weighted sampler
        weighted_sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        stage2_train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            sampler=weighted_sampler,
            num_workers=train_loader.num_workers,
            pin_memory=True,
        )

        # Reset optimizer and scheduler for Stage 2
        optimizer = torch.optim.AdamW(
            redshift_model.parameters(),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg.get("weight_decay", 0)),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=20
        )

        stage2_patience_counter = 0
        stage2_best_val_loss = float("inf")

        for epoch in range(stage2_epochs):
            redshift_model.train()
            pin_frozen_bn_eval()
            loss_epoch = 0.0
            batch_count = 0

            for X, Y, idx, t_id in stage2_train_loader:
                X, Y = X.to(device), Y.to(device)
                optimizer.zero_grad()
                outputs = redshift_model(X)
                loss = criterion(*outputs, Y) if prediction_type == "mdn" else criterion(outputs, Y)
                if isinstance(loss, dict):
                    loss = loss["total"]

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(redshift_model.parameters(), max_norm=1.0)
                optimizer.step()
                loss_epoch += loss.item()
                batch_count += 1

            train_loss = loss_epoch / max(batch_count, 1)

            redshift_model.eval()
            val_loss_epoch = 0.0
            val_count = 0
            all_preds = []
            all_true = []

            with torch.no_grad():
                for X, Y, idx, t_id in val_loader:
                    X, Y = X.to(device), Y.to(device)
                    outputs = redshift_model(X)
                    eval_mask = (Y.squeeze(-1) <= z_bin_max) if binned_output else None
                    if eval_mask is not None and not eval_mask.any():
                        continue
                    loss_outputs = outputs
                    loss_target = Y
                    if eval_mask is not None:
                        loss_target = Y[eval_mask]
                        if isinstance(outputs, dict):
                            loss_outputs = {key: value[eval_mask] for key, value in outputs.items()}
                        elif isinstance(outputs, tuple):
                            loss_outputs = tuple(value[eval_mask] for value in outputs)
                    val_loss_batch = criterion(*loss_outputs, loss_target) if prediction_type == "mdn" else criterion(loss_outputs, loss_target)
                    if isinstance(val_loss_batch, dict):
                        val_loss_batch = val_loss_batch["total"]
                    preds = decode_redshift_output(outputs, prediction_type)
                    if eval_mask is not None:
                        preds = preds[eval_mask]
                    if torch.isnan(val_loss_batch) or torch.isinf(val_loss_batch):
                        continue
                    val_loss_epoch += val_loss_batch.item()
                    val_count += 1
                    all_preds.append(preds.cpu().numpy().flatten())
                    all_true.append((Y[eval_mask] if eval_mask is not None else Y).cpu().numpy().flatten())

            val_loss = val_loss_epoch / max(val_count, 1)
            scheduler.step(val_loss)

            all_preds = np.concatenate(all_preds)
            all_true = np.concatenate(all_true)
            if binned_output:
                eval_mask = all_true <= z_bin_max
                all_preds = all_preds[eval_mask]
                all_true = all_true[eval_mask]
            metrics = compute_metrics(all_true, all_preds)

            wandb.log({
                "stage2_train_loss": train_loss,
                "stage2_val_loss": val_loss,
                "stage2_val_nmad": metrics["nmad"],
                "stage2_val_z_bias": metrics["bias"],
                "stage2_catastrophic_outliers": metrics["eta"],
                "stage2_val_rmse": metrics["rmse"],
                "stage2_lr": optimizer.param_groups[0]["lr"],
                "stage2_epoch": epoch,
            })

            if val_loss < stage2_best_val_loss:
                stage2_best_val_loss = val_loss
                stage2_patience_counter = 0
                save_checkpoint(
                    redshift_model, optimizer, scheduler, epoch,
                    train_losses, val_losses, stage2_best_val_loss,
                    f"checkpoints/{exp_name}_stage2_best_model.pth",
                )
            else:
                stage2_patience_counter += 1

            print(
                f"Stage2 Epoch {epoch+1}/{stage2_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"NMAD: {metrics['nmad']:.4f} | Bias: {metrics['bias']:.4f} | "
                f"η: {metrics['eta']:.2f}%"
            )

            if stage2_patience_counter >= stage2_patience:
                print(f"Stage 2 early stopping at epoch {epoch+1}")
                break

        # Load Stage 2 best checkpoint for final evaluation
        stage2_ckpt_path = f"checkpoints/{exp_name}_stage2_best_model.pth"
        if os.path.exists(stage2_ckpt_path):
            load_checkpoint(stage2_ckpt_path, redshift_model, optimizer, scheduler, device)
            print(f"Loaded best Stage 2 checkpoint from {stage2_ckpt_path}")

    # ========== POST-TRAINING TEST EVALUATION ==========
    print("\n" + "="*60)
    print("POST-TRAINING EVALUATION ON TEST SET")
    print("="*60)

    best_ckpt_path = f"checkpoints/{exp_name}_best_model.pth"
    if os.path.exists(best_ckpt_path):
        result = load_checkpoint(best_ckpt_path, redshift_model, optimizer, scheduler, device)
        if result[0] is not None:
            print(f"Loaded best checkpoint from {best_ckpt_path}")
        else:
            print(f"Could not load checkpoint — using last training state")

    redshift_model.eval()
    test_true = []
    test_preds = []
    test_target_ids = []

    with torch.no_grad():
        for X, Y, idx, t_id in test_loader:
            X, Y = X.to(device), Y.to(device)
            
            preds = decode_redshift_output(redshift_model(X), prediction_type)
            
            test_true.extend(Y.cpu().numpy().flatten().tolist())
            test_preds.extend(preds.cpu().numpy().flatten().tolist())
            test_target_ids.extend(list(t_id))

    test_true = np.array(test_true)
    test_preds = np.array(test_preds)
    if binned_output:
        eval_mask = test_true <= z_bin_max
        test_true = test_true[eval_mask]
        test_preds = test_preds[eval_mask]
        print(f"Binned evaluation restricted to true z <= {z_bin_max}")

    test_metrics = compute_metrics(test_true, test_preds)
    print(f"\nTest Metrics:")
    print(f"  NMAD:  {test_metrics['nmad']:.5f}")
    print(f"  \u03b7:     {test_metrics['eta']:.2f}%")
    print(f"  RMSE:  {test_metrics['rmse']:.5f}")
    print(f"  Bias:  {test_metrics['bias']:.5f}")

    wandb.log({
        "test_nmad": test_metrics["nmad"],
        "test_eta": test_metrics["eta"],
        "test_rmse": test_metrics["rmse"],
        "test_bias": test_metrics["bias"],
    })

    # ========== TEST-TIME AUGMENTATION (TTA) ==========
    tta_cfg = cfg.get("tta", {})
    if tta_cfg.get("enabled", False):
        print("\n" + "="*60)
        print("TEST-TIME AUGMENTATION (TTA)")
        print("="*60)

        tta_true, tta_preds = predict_with_tta(
            redshift_model, test_loader, device, tta_cfg, prediction_type
        )
        tta_metrics = compute_metrics(tta_true, tta_preds)
        print(f"\nTTA Test Metrics (n_aug={tta_cfg.get('n_augmentations', 10)}):")
        print(f"  NMAD:  {tta_metrics['nmad']:.5f}")
        print(f"  \u03b7:     {tta_metrics['eta']:.2f}%")
        print(f"  RMSE:  {tta_metrics['rmse']:.5f}")
        print(f"  Bias:  {tta_metrics['bias']:.5f}")

        # Compare with standard inference
        nmad_delta = tta_metrics["nmad"] - test_metrics["nmad"]
        eta_delta = tta_metrics["eta"] - test_metrics["eta"]
        print(f"\n  vs Standard: NMAD {'+' if nmad_delta >= 0 else ''}{nmad_delta:.5f}, "
              f"\u03b7 {'+' if eta_delta >= 0 else ''}{eta_delta:.2f}%")

        wandb.log({
            "tta_nmad": tta_metrics["nmad"],
            "tta_eta": tta_metrics["eta"],
            "tta_rmse": tta_metrics["rmse"],
            "tta_bias": tta_metrics["bias"],
        })

        # Use TTA predictions for the true vs predicted z plot
        test_true = tta_true
        test_preds = tta_preds
        test_metrics = tta_metrics
        print("Using TTA predictions for plots.")

    # ========== TRUE vs PREDICTED Z PLOT ==========
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    delz = (test_true - test_preds) / (1 + test_true)
    frac = 0.15
    min_lim = -0.05
    max_lim = np.ceil(test_true.max()) + 0.1

    fig = plt.figure(figsize=(5.7, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 1], wspace=0.1, hspace=0.01)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    ax1.scatter(test_true, test_preds, marker=".", s=10, c="blue", alpha=0.6)
    ax1.plot([min_lim, max_lim], [min_lim, max_lim], c="k", zorder=9)
    x_line = np.linspace(0, max_lim)
    ax1.plot(x_line, (1 + x_line) * frac + x_line, c="k", linestyle="dotted", zorder=9)
    ax1.plot(x_line, (1 + x_line) * -frac + x_line, c="k", linestyle="dotted", zorder=9)
    ax1.text(min_lim + 0.07, max_lim - 0.1, f"NMAD: {test_metrics['nmad']:.4f}", fontsize=11)
    ax1.text(min_lim + 0.07, max_lim - 0.2, f"\u03b7: {test_metrics['eta']:.2f}%", fontsize=11)
    ax1.set_ylabel(r"Predicted $z$", fontsize=16)
    ax1.set_xlim(min_lim, max_lim)
    ax1.set_ylim(min_lim, max_lim)
    ax1.set_title("Test Set: True vs Predicted Redshift")
    ax1.minorticks_on()
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.scatter(test_true, delz, marker=".", s=10, c="blue", alpha=0.6)
    ax2.axhline(y=0, c="k", zorder=9)
    ax2.axhline(y=-frac, c="k", linestyle="dotted", zorder=9)
    ax2.axhline(y=frac, c="k", linestyle="dotted", zorder=9)
    ax2.set_ylabel(r"$\Delta z/(1+z)$", fontsize=14)
    ax2.set_xlabel(r"True $z$", fontsize=16)
    ax2.minorticks_on()

    plt.tight_layout()
    plt.savefig("outputs/plots/test_z_comparison.png", dpi=150, bbox_inches="tight")
    wandb.log({"test_redshift_plot": wandb.Image(fig)})
    plt.close(fig)
    print("Saved: outputs/plots/test_z_comparison.png")

    # ========== UMAP 3D PLOT ==========
    import umap

    features_list = []
    with torch.no_grad():
        for X, Y, idx, t_id in test_loader:
            X = X.to(device)
            x = X.unsqueeze(1)
            x = redshift_model.pretrained_model.forward_conv(x)
            x = x.flatten(start_dim=1)
            x = redshift_model.proj_to_d_model(x)
            x = x.unsqueeze(0)
            encoded = redshift_model.encoder(x)
            encoded = encoded.squeeze(0)
            attn_out, _ = redshift_model.attention(encoded, encoded, encoded)
            x = attn_out + encoded
            feat = redshift_model.mlp_blocks(x)
            features_list.append(feat.cpu().numpy())

    features = np.concatenate(features_list, axis=0)

    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42, metric="euclidean")
    embedding = reducer.fit_transform(features)

    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    scatter = ax_3d.scatter(
        embedding[:, 0], embedding[:, 1], embedding[:, 2],
        c=test_true, cmap="seismic", alpha=0.6, s=20, edgecolors="w", linewidth=0.5,
    )
    ax_3d.set_title("Test Set: 3D UMAP of Learned Features", fontsize=14, pad=20)
    ax_3d.set_xlabel("UMAP 1", fontsize=12)
    ax_3d.set_ylabel("UMAP 2", fontsize=12)
    ax_3d.set_zlabel("UMAP 3", fontsize=12)
    cbar = plt.colorbar(scatter, ax=ax_3d, pad=0.1, shrink=0.8)
    cbar.set_label("Redshift (z)", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/plots/test_umap_3d.png", dpi=150, bbox_inches="tight")
    wandb.log({"test_umap_3d": wandb.Image(fig_3d)})
    plt.close(fig_3d)
    print("Saved: outputs/plots/test_umap_3d.png")

    print("="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

    wandb.finish()


if __name__ == "__main__":
    main()
