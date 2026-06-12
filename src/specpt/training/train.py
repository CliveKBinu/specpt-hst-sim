import argparse
import os
import pickle
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

    train_loader, val_loader, _ = create_dataloaders(
        train_df, val_df, test_df,
        batch_size=train_cfg["batch_size"],
        num_workers=0,
    )

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

    wandb.finish()


if __name__ == "__main__":
    main()
