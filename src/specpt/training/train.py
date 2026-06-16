import argparse
import os
import yaml
import torch
import wandb
import numpy as np
from pathlib import Path

from ..model import SpecPT, EnhancedSpecPTForRedshift
from ..losses import NMADLoss
from ..dataloader import load_grism_data, split_data, create_dataloaders
from .eval import compute_metrics


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
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["train_losses"], checkpoint["val_losses"], checkpoint["best_val_loss"]


def main():
    parser = argparse.ArgumentParser(description="SpecPT Redshift Training")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_entity = args.wandb_entity or wandb_cfg.get("entity")
    wandb_project = args.wandb_project or wandb_cfg.get("project")
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

    ae_path = os.path.expanduser(data_cfg["pretrained_autoencoder"])
    state_dict = torch.load(ae_path, map_location=device)
    auto_model.load_state_dict(state_dict, strict=True)

    redshift_model = EnhancedSpecPTForRedshift(
        auto_model,
        output_features=1,
        num_mlp_blocks=model_cfg["num_mlp_blocks"],
        mlp_dim=model_cfg["mlp_dim"],
        dropout_rate=model_cfg["dropout"],
    )

    rz_path = os.path.expanduser(data_cfg["pretrained_redshift"])
    state_dict = torch.load(rz_path, map_location=device)
    autoencoder_prefixes = ("encoder.", "proj_to_d_model.", "pretrained_model.")
    head_only = {k: v for k, v in state_dict.items() if not k.startswith(autoencoder_prefixes)}
    redshift_model.load_state_dict(head_only, strict=False)

    for param in redshift_model.pretrained_model.parameters():
        param.requires_grad = False

    redshift_model.to(device)

    data = load_grism_data(data_cfg["path"])
    train_df, val_df, test_df = split_data(
        data,
        val_split=data_cfg.get("val_split", 0.1),
        test_split=data_cfg.get("test_split", 0.1),
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        batch_size=train_cfg["batch_size"],
        num_workers=0,
    )
    os.makedirs("outputs/plots", exist_ok=True)

    criterion = NMADLoss(normalization_factor="std")
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

    for epoch in range(start_epoch, train_cfg["epochs"]):
        redshift_model.train()
        loss_epoch = 0.0
        batch_count = 0

        for X, Y, idx, t_id in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            preds = redshift_model(X)
            loss = criterion(preds, Y)

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

        with torch.no_grad():
            for X, Y, idx, t_id in val_loader:
                X, Y = X.to(device), Y.to(device)
                preds = redshift_model(X)
                val_loss_batch = criterion(preds, Y)
                if torch.isnan(val_loss_batch) or torch.isinf(val_loss_batch):
                    continue
                val_loss_epoch += val_loss_batch.item()
                val_count += 1
                all_preds.append(preds.cpu().numpy().flatten())
                all_true.append(Y.cpu().numpy().flatten())

        val_loss = val_loss_epoch / max(val_count, 1)
        val_losses.append(val_loss)

        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)
        metrics = compute_metrics(all_true, all_preds)

        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_nmad": metrics["nmad"],
                "val_z_bias": metrics["bias"],
                "catastrophic_outliers": metrics["eta"],
                "val_rmse": metrics["rmse"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                redshift_model, optimizer, scheduler, epoch,
                train_losses, val_losses, best_val_loss,
                "checkpoints/best_model.pth",
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

    # ========== POST-TRAINING TEST EVALUATION ==========
    print("\n" + "="*60)
    print("POST-TRAINING EVALUATION ON TEST SET")
    print("="*60)

    best_ckpt_path = "checkpoints/best_model.pth"
    if os.path.exists(best_ckpt_path):
        load_checkpoint(best_ckpt_path, redshift_model, optimizer, scheduler, device)
        print(f"Loaded best checkpoint from {best_ckpt_path}")

    redshift_model.eval()
    test_true = []
    test_preds = []
    test_target_ids = []

    with torch.no_grad():
        for X, Y, idx, t_id in test_loader:
            X, Y = X.to(device), Y.to(device)
            preds = redshift_model(X)
            test_true.extend(Y.cpu().numpy().flatten().tolist())
            test_preds.extend(preds.cpu().numpy().flatten().tolist())
            test_target_ids.extend(list(t_id))

    test_true = np.array(test_true)
    test_preds = np.array(test_preds)

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
