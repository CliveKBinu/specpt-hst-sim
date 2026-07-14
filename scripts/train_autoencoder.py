#!/usr/bin/env python3
"""
SpecPT Autoencoder Training.

Trains the SpecPT autoencoder (conv + transformer encoder/decoder + linear
reconstruction) on HST sim data.  Target = input (reconstruction).

Usage:
    python scripts/train_autoencoder.py --config configs/autoencoder_regrid.yaml
"""
import argparse
import os
import yaml
import torch
import wandb
import numpy as np
import zipfile
from pathlib import Path
from time import time

from src.specpt.model import SpecPT, SpectrumNormalizer
from src.specpt.losses import NMADLoss
from src.specpt.dataloader import load_grism_data, split_data, create_autoenc_dataloaders


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
        checkpoint = torch.load(path, map_location=device)
    except (RuntimeError, zipfile.BadZipFile, EOFError) as e:
        print(f"Warning: Corrupted checkpoint at {path}: {e}")
        print("  Skipping checkpoint load — model will use current weights")
        return None, None, None, None

    model_state = checkpoint["model_state_dict"]

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        print(f"Checkpoint key mismatch — missing {len(missing)}, unexpected {len(unexpected)}")
        if missing:
            print(f"  Missing: {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected: {unexpected[:5]}...")
        shared_keys = {k: v for k, v in model_state.items() if k in model.state_dict()}
        model.load_state_dict(shared_keys, strict=False)
        print(f"  Loaded {len(shared_keys)} shared keys (backbone only)")
        print("  Skipping optimizer/scheduler loading (architecture mismatch)")
        return checkpoint["epoch"], checkpoint["train_losses"], checkpoint["val_losses"], checkpoint["best_val_loss"]

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["train_losses"], checkpoint["val_losses"], checkpoint["best_val_loss"]


def main():
    parser = argparse.ArgumentParser(description="SpecPT Autoencoder Training")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    wandb_cfg = cfg.get("wandb", {})

    exp_name = os.path.splitext(os.path.basename(args.config))[0]
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wandb_entity = args.wandb_entity or wandb_cfg.get("entity")
    wandb_project = args.wandb_project or wandb_cfg.get("project")
    cfg["experiment_name"] = exp_name
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config=cfg,
    )

    # ========== MODEL ==========
    model = SpecPT(
        input_size=model_cfg["input_size"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_encoder_layers=model_cfg["num_encoder_layers"],
        num_decoder_layers=model_cfg["num_decoder_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
    )

    ae_path = os.path.expanduser(data_cfg.get("pretrained_autoencoder", ""))
    if ae_path and os.path.exists(ae_path):
        state_dict = torch.load(ae_path, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained autoencoder from {ae_path}")
        print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print("No pretrained autoencoder — initializing from scratch.")

    for param in model.parameters():
        param.requires_grad = True

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ========== DATA ==========
    data = load_grism_data(data_cfg["path"])
    train_df, val_df, test_df = split_data(
        data,
        val_split=data_cfg.get("val_split", 0.15),
        test_split=data_cfg.get("test_split", 0.15),
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_loader, val_loader, test_loader = create_autoenc_dataloaders(
        train_df, val_df, test_df,
        batch_size=train_cfg["batch_size"],
        num_workers=0,
    )

    # ========== LOSS / OPTIMIZER / SCHEDULER ==========
    criterion = NMADLoss(normalization_factor="std")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0)),
    )

    scheduler_type = train_cfg.get("lr_scheduler", "ReduceLROnPlateau")
    if scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=train_cfg.get("lr_scheduler_warmup", 500),
            num_training_steps=train_cfg["epochs"] * len(train_loader),
        )
        use_plateau = False
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=float(train_cfg.get("lr_scheduler_factor", 0.1)),
            patience=train_cfg.get("lr_scheduler_patience", 10),
        )
        use_plateau = True

    wandb.watch(model, criterion=criterion, log="gradients")

    # ========== TRAINING LOOP ==========
    start_epoch = 0
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    patience_counter = 0
    patience = train_cfg["patience"]

    if args.resume:
        start_epoch, train_losses, val_losses, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        if start_epoch is not None:
            start_epoch += 1
            print(f"Resuming from epoch {start_epoch}")

    t_start = time()
    for epoch in range(start_epoch, train_cfg["epochs"]):
        # --- Train ---
        model.train()
        loss_epoch = 0.0
        batch_count = 0

        for X, Y, idx, t_id in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, Y)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_epoch += loss.item()
            batch_count += 1

        train_loss = loss_epoch / max(batch_count, 1)
        train_losses.append(train_loss)

        if use_plateau:
            scheduler.step(train_loss)
        else:
            scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss_epoch = 0.0
        val_count = 0

        with torch.no_grad():
            for X, Y, idx, t_id in val_loader:
                X, Y = X.to(device), Y.to(device)
                preds = model(X)
                val_loss_batch = criterion(preds, Y)
                if torch.isnan(val_loss_batch) or torch.isinf(val_loss_batch):
                    continue
                val_loss_epoch += val_loss_batch.item()
                val_count += 1

        val_loss = val_loss_epoch / max(val_count, 1)
        val_losses.append(val_loss)

        lr = optimizer.param_groups[0]["lr"]

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "epoch": epoch,
        })

        elapsed = time() - t_start
        print(
            f"Epoch {epoch+1}/{train_cfg['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"LR: {lr:.2e} | {elapsed:.0f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_losses, val_losses, best_val_loss,
                f"checkpoints/{exp_name}_autoencoder_best.pth",
            )
            torch.save(model.state_dict(), f"checkpoints/{exp_name}_autoencoder_weights.pth")
            print(f"  -> Saved best (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_losses, val_losses, best_val_loss,
                "checkpoints/latest_autoenc_checkpoint.pth",
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # ========== POST-TRAINING TEST EVALUATION ==========
    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION ON TEST SET")
    print("=" * 60)

    best_ckpt_path = f"checkpoints/{exp_name}_autoencoder_best.pth"
    if os.path.exists(best_ckpt_path):
        load_checkpoint(best_ckpt_path, model, optimizer, scheduler, device)
        print(f"Loaded best checkpoint from {best_ckpt_path}")

    model.eval()
    test_loss_epoch = 0.0
    test_count = 0

    with torch.no_grad():
        for X, Y, idx, t_id in test_loader:
            X, Y = X.to(device), Y.to(device)
            preds = model(X)
            loss_batch = criterion(preds, Y)
            if torch.isnan(loss_batch) or torch.isinf(loss_batch):
                continue
            test_loss_epoch += loss_batch.item()
            test_count += 1

    test_loss = test_loss_epoch / max(test_count, 1)
    print(f"Test NMAD Loss: {test_loss:.4f}")

    wandb.log({"test_nmad_loss": test_loss})

    # ========== RECONSTRUCTION QUALITY METRICS ==========
    all_preds = []
    all_true = []
    with torch.no_grad():
        for X, Y, idx, t_id in test_loader:
            X = X.to(device)
            preds = model(X)
            all_preds.append(preds.cpu().numpy())
            all_true.append(Y.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    mae = np.mean(np.abs(all_preds - all_true))
    cos_sim = np.mean([
        np.dot(all_true[i], all_preds[i]) / (np.linalg.norm(all_true[i]) * np.linalg.norm(all_preds[i]) + 1e-8)
        for i in range(len(all_true))
    ])
    print(f"Test MAE: {mae:.6f}")
    print(f"Test Cosine Similarity: {cos_sim:.6f}")

    wandb.log({
        "test_mae": mae,
        "test_cosine_sim": cos_sim,
    })

    # ========== SAMPLE RECONSTRUCTION PLOTS ==========
    os.makedirs("outputs/plots", exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_samples = min(5, len(all_true))
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i in range(n_samples):
        axes[i].plot(all_true[i], alpha=0.7, label="Original")
        axes[i].plot(all_preds[i], alpha=0.7, label="Reconstructed")
        axes[i].set_title(f"Sample {i}")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig("outputs/plots/autoencoder_reconstruction.png", dpi=150, bbox_inches="tight")
    wandb.log({"reconstruction_plot": wandb.Image(fig)})
    plt.close(fig)
    print("Saved: outputs/plots/autoencoder_reconstruction.png")

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    wandb.finish()


if __name__ == "__main__":
    main()
