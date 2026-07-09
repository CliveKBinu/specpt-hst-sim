#!/usr/bin/env python3
"""Fine-tune exp_032 on real 3D-HST grism data (two stages).

Stage 1 — linear probe (freeze all except prediction head)
Stage 2 — partial freeze (unfreeze last transformer, attention, MLP, head)

Usage:
    python scripts/finetune_real_3dhst.py --stage 1
    python scripts/finetune_real_3dhst.py --stage 2 --init checkpoints/.../stage1_best_model.pth
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml
import umap
from src.specpt.model import SpecPT, EnhancedSpecPTForRedshift, SpectrumNormalizer
from src.specpt.dataloader import HSTGrismDataset
from src.specpt.training.eval import compute_metrics, compute_per_bin_metrics

TRAIN_WAVES = np.linspace(10311.4, 17464.6, 7781)
PADDING_MASK = (TRAIN_WAVES < 11000) | (TRAIN_WAVES > 16500)

STAGE_CONFIGS = {
    1: {
        "lr": 3e-4,
        "weight_decay": 1e-3,
        "epochs": 30,
        "patience": 5,
        "noise_std": 0.0,
        "warmup_epochs": 0,
    },
    2: {
        "lr": 1e-5,
        "weight_decay": 1e-4,
        "epochs": 40,
        "patience": 7,
        "noise_std": 0.01,
        "warmup_epochs": 2,
    },
}

BEST_VAL = {"nmad": 1e9, "epoch": 0}

def _load_and_fix_real_data(data_path, min_snr=2.5):
    print(f"[data] Loading real data: {data_path}")
    df = pd.read_pickle(data_path)
    print(f"   Total: {len(df)} spectra")
    df = df[df["SNR"] >= min_snr].reset_index(drop=True)
    print(f"   After SNR >= {min_snr}: {len(df)} spectra")

    safe_sens = df["sensitivity_resampled"].apply(
        lambda arr: np.where(arr == 0, 1e-8, arr)
    )
    df["flux_sensitivity"] = df["clean_flux_resampled"] / safe_sens

    specs_on_grid = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="   Interpolating"):
        ref_wl = np.asarray(row["wavelength_resampled"], dtype=np.float64)
        spec = np.asarray(row["flux_sensitivity"], dtype=np.float64)
        on_grid = np.interp(TRAIN_WAVES, ref_wl, spec, left=np.nan, right=np.nan)
        on_grid[PADDING_MASK] = np.nan
        specs_on_grid.append(on_grid.astype(np.float32))

    df["spec"] = specs_on_grid
    df = df.rename(columns={"grism_id": "TARGETID"})

    sample = df["spec"].iloc[0]
    print(f"   spec shape: {sample.shape}, dtype: {sample.dtype}")
    print(f"   nan fraction: {np.mean(np.isnan(sample)):.3f}")
    print(f"   z range: [{df['z'].min():.3f}, {df['z'].max():.3f}]")
    print(f"   SNR range: [{df['SNR'].min():.2f}, {df['SNR'].max():.2f}]")
    return df

def _split(df, train_frac=0.70, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_train = int(len(df) * train_frac)
    n_val = int(len(df) * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    print(f"   Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy(), df.iloc[test_idx].copy()

def _build_model(ckpt_path, ae_path, model_cfg_path, device):
    print(f"[model] Building from {model_cfg_path}")
    with open(model_cfg_path) as f:
        model_cfg = yaml.safe_load(f)["model"]

    ae_state = torch.load(ae_path, map_location="cpu", weights_only=False)
    specpt = SpecPT(
        input_size=model_cfg["input_size"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_encoder_layers=model_cfg["num_encoder_layers"],
        num_decoder_layers=model_cfg["num_decoder_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
    )
    specpt.load_state_dict(ae_state, strict=False)

    model = EnhancedSpecPTForRedshift(
        pretrained_model=specpt,
        num_mlp_blocks=model_cfg["num_mlp_blocks"],
        mlp_dim=model_cfg["mlp_dim"],
        dropout_rate=model_cfg.get("dropout", 0.1),
    )

    print(f"   Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"   Missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"   Unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    print(f"   Checkpoint epoch: {ckpt.get('epoch')}, best_val_loss: {ckpt.get('best_val_loss'):.4f}")

    return model.to(device)

def _apply_freeze_policy(model, stage):
    for p in model.parameters():
        p.requires_grad = False

    if stage == 1:
        for p in model.prediction.parameters():
            p.requires_grad = True
    elif stage == 2:
        for p in model.encoder.layers[-1].parameters():
            p.requires_grad = True
        for p in model.attention.parameters():
            p.requires_grad = True
        for p in model.mlp_blocks.parameters():
            p.requires_grad = True
        for p in model.prediction.parameters():
            p.requires_grad = True
        for p in model.pretrained_model.transformer_decoder.parameters():
            p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[freeze] Trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.1f}%)")
    return model

def _train_epoch(model, loader, optimizer, criterion, device, noise_std):
    model.train()
    # Keep frozen submodules in eval mode to disable dropout + BN updates
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
    model.encoder.eval()  # encoder dropout corrupts features for attention-over-batch
    model.attention.eval()  # attention dropout (if any) also safe to disable
    total_loss = 0
    for batch_idx, (X, Y, _, _) in enumerate(loader):
        X, Y = X.cuda(), Y.cuda()  # sync transfer — no non_blocking
        if noise_std > 0 and model.training:
            X = X + torch.randn_like(X) * noise_std
        optimizer.zero_grad()
        y_pred = model(X).flatten()
        nan_count = torch.isnan(y_pred).sum().item()
        if nan_count > 0:
            y_nonnan = y_pred[~torch.isnan(y_pred)]
            extra = f"  all_nan={nan_count==y_pred.numel()}"
            if y_nonnan.numel() > 0:
                extra = f"  nonnan_min={y_nonnan.min().item():.4f}, nonnan_max={y_nonnan.max().item():.4f}"
            print(f"   [dbg] NaN after pred: {nan_count}/{y_pred.numel()}{extra}")
            # Try step-by-step through prediction head
            h = model.prediction[0](x)   # Linear(1024,512)
            h = model.prediction[1](h)   # Swish
            h = model.prediction[2](h)   # Dropout
            h = model.prediction[3](h)   # Linear(512,1)
            print(f"   [dbg]  step pred: L0={torch.isnan(h).sum().item()}/{h.numel()}, "
                  f"range=[{h.min().item():.2f},{h.max().item():.2f}]")
            h = model.prediction[4](h)   # Softplus
            print(f"   [dbg]  step softplus: NaN={torch.isnan(h).sum().item()}/{h.numel()}")
            return float("nan")
        y_pred = y_pred.flatten()
        loss = criterion(y_pred, Y)
        loss = criterion(y_pred, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    all_pred, all_true, all_ids = [], [], []
    for X, Y, _, t_id in loader:
        X = X.to(device, non_blocking=True)
        y_pred = model(X).flatten()
        all_pred.append(y_pred.cpu().numpy())
        all_true.append(Y.numpy())
        all_ids.extend(t_id)
    preds = np.concatenate(all_pred)
    trues = np.concatenate(all_true)
    preds = np.clip(preds, 0.0, None)
    metrics = compute_metrics(trues, preds)
    metrics["per_zbin"] = compute_per_bin_metrics(trues, preds, bins=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    return metrics, (trues, preds, all_ids)

def _save_checkpoint(model, optimizer, epoch, best_val_nmad, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_nmad": best_val_nmad,
    }, path)
    print(f"   Checkpoint saved: {path}")

def _make_plots(trues, preds, snr, per_zbin, per_snrb, overall_metrics, stage):
    rng = np.random.default_rng(42)
    delz = (preds - trues) / (1 + trues)
    frac = 0.15
    max_lim = np.ceil(max(trues.max(), preds.max())) + 0.1
    min_lim = -0.05
    figures = {}

    fig = plt.figure(figsize=(5.7, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.01)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax1.scatter(trues, preds, marker=".", s=10, c="blue", alpha=0.6)
    ax1.plot([min_lim, max_lim], [min_lim, max_lim], c="k", zorder=9)
    xl = np.linspace(0, max_lim)
    ax1.plot(xl, (1+xl)*frac+xl, c="k", ls="dotted", zorder=9)
    ax1.plot(xl, (1+xl)*-frac+xl, c="k", ls="dotted", zorder=9)
    ax1.text(min_lim+0.07, max_lim-0.1, f"NMAD: {overall_metrics['nmad']:.4f}", fontsize=11)
    ax1.text(min_lim+0.07, max_lim-0.2, f"eta: {overall_metrics['eta']:.2f}%", fontsize=11)
    ax1.set_ylabel(r"Predicted $z$", fontsize=16)
    ax1.set_xlim(min_lim, max_lim)
    ax1.set_ylim(min_lim, max_lim)
    ax1.set_title(f"Stage {stage} Real 3D-HST: True vs Predicted Redshift", fontsize=12)
    ax1.minorticks_on()
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.scatter(trues, delz, marker=".", s=10, c="blue", alpha=0.6)
    ax2.axhline(y=0, c="k", zorder=9)
    ax2.axhline(y=-frac, c="k", ls="dotted", zorder=9)
    ax2.axhline(y=frac, c="k", ls="dotted", zorder=9)
    ax2.set_ylabel(r"$\Delta z/(1+z)$", fontsize=14)
    ax2.set_xlabel(r"True $z$", fontsize=16)
    ax2.minorticks_on()
    plt.tight_layout()
    figures["scatter_standard"] = fig
    plt.close(fig)

    fig = plt.figure(figsize=(6.0, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.01)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    sc = ax1.scatter(trues, preds, marker=".", s=10, c=snr, cmap="viridis", alpha=0.6)
    ax1.plot([min_lim, max_lim], [min_lim, max_lim], c="k", zorder=9)
    ax1.plot(xl, (1+xl)*frac+xl, c="k", ls="dotted", zorder=9)
    ax1.plot(xl, (1+xl)*-frac+xl, c="k", ls="dotted", zorder=9)
    cbar = plt.colorbar(sc, ax=ax1, pad=0.02)
    cbar.set_label("SNR", fontsize=10)
    ax1.set_ylabel(r"Predicted $z$", fontsize=16)
    ax1.set_xlim(min_lim, max_lim)
    ax1.set_ylim(min_lim, max_lim)
    ax1.set_title(f"Stage {stage} Real 3D-HST: Predicted z Colored by SNR", fontsize=12)
    ax1.minorticks_on()
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.scatter(trues, delz, marker=".", s=10, c="blue", alpha=0.6)
    ax2.axhline(y=0, c="k", zorder=9)
    ax2.axhline(y=-frac, c="k", ls="dotted", zorder=9)
    ax2.axhline(y=frac, c="k", ls="dotted", zorder=9)
    ax2.set_ylabel(r"$\Delta z/(1+z)$", fontsize=14)
    ax2.set_xlabel(r"True $z$", fontsize=16)
    ax2.minorticks_on()
    plt.tight_layout()
    figures["scatter_snr_colored"] = fig
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    per_z = [b for b in per_zbin if b.get("count", 0) > 0]
    bins_s = [b["bin"] for b in per_z]
    nmads = [b["nmad"] for b in per_z]
    counts = [b["count"] for b in per_z]
    bars = ax.bar(bins_s, nmads, color="steelblue", alpha=0.8)
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002, f"n={n}", ha="center", fontsize=8)
    ax.set_xlabel("Redshift bin (z_true)")
    ax.set_ylabel("NMAD")
    ax.set_title(f"Stage {stage} Real 3D-HST: NMAD per z bin")
    ax.axhline(y=0.020, c="r", ls="--", alpha=0.5, label="Target NMAD=0.02")
    ax.legend()
    plt.tight_layout()
    figures["per_z_bin"] = fig
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins_s = [b["bin"] for b in per_snrb]
    nmads = [b["nmad"] for b in per_snrb]
    counts = [b["count"] for b in per_snrb]
    bars = ax.bar(bins_s, nmads, color="darkorange", alpha=0.8)
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002, f"n={n}", ha="center", fontsize=8)
    ax.set_xlabel("SNR bin")
    ax.set_ylabel("NMAD")
    ax.set_title(f"Stage {stage} Real 3D-HST: NMAD per SNR bin")
    ax.axhline(y=0.020, c="r", ls="--", alpha=0.5, label="Target NMAD=0.02")
    ax.legend()
    plt.tight_layout()
    figures["per_snr_bin"] = fig
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(delz, bins=200, range=(-1, 1), color="steelblue", alpha=0.7, edgecolor="k")
    ax.axvspan(-frac, frac, alpha=0.15, color="green", label=f"Non-catastrophic (|delz|<0.15)")
    ax.set_xlabel(r"$\Delta z/(1+z)$")
    ax.set_ylabel("Count")
    ax.set_title(f"Stage {stage} Real 3D-HST: Residual Dist  (eta={overall_metrics['eta']:.2f}%)")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    figures["outlier_diagnostic"] = fig
    plt.close(fig)

    return figures

def _save_outputs(trues, preds, snr, all_ids, metrics, figures, cfg, stage):
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"stage{stage}_test_predictions.csv"
    pd.DataFrame({
        "TARGETID": all_ids,
        "z_true": trues,
        "z_pred": preds,
        "snr": snr,
    }).to_csv(csv_path, index=False)

    def to_python(obj):
        if isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_python(x) for x in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    metrics_path = out_dir / f"stage{stage}_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(to_python(metrics), f, indent=2)

    for name, fig in figures.items():
        fig.savefig(out_dir / f"stage{stage}_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"   Outputs saved to {out_dir}")

def _log_wandb(trues, preds, snr, all_ids, metrics, figures, cfg, stage):
    import wandb
    run = wandb.init(
        id=cfg["wandb_run_id"],
        resume="allow",
        project=cfg["wandb_project"],
        entity=cfg["wandb_entity"],
    )

    prefix = f"real_finetune_stage{stage}_"
    wandb.log({
        f"{prefix}nmad": metrics["nmad"],
        f"{prefix}rmse": metrics["rmse"],
        f"{prefix}bias": metrics["bias"],
        f"{prefix}eta": metrics["eta"],
        f"{prefix}n_test": len(trues),
    })

    for name, fig in figures.items():
        wandb.log({f"{prefix}{name}": wandb.Image(fig)})

    table = wandb.Table(dataframe=pd.DataFrame({
        "TARGETID": all_ids, "z_true": trues, "z_pred": preds, "snr": snr,
    }))
    wandb.log({f"{prefix}predictions": table})

    artifact = wandb.Artifact(
        name=f"real_finetune_exp_032_stage{stage}",
        type="real-data-finetune",
        description=f"Stage {stage} fine-tune on real 3D-HST AEGIS data. "
                    f"NMAD={metrics['nmad']:.5f}, eta={metrics['eta']:.2f}%, "
                    f"n_test={len(trues)}.",
    )
    out_dir = Path(cfg["output_dir"])
    artifact.add_file(str(out_dir / f"stage{stage}_test_predictions.csv"))
    artifact.add_file(str(out_dir / f"stage{stage}_test_metrics.json"))
    for name in figures:
        artifact.add_file(str(out_dir / f"stage{stage}_{name}.png"))
    run.log_artifact(artifact)
    wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
    parser.add_argument("--init", type=str,
                        default="/home/ckb2084/research/specpt-hst-sim/checkpoints/exp_032_best_model.pth")
    args = parser.parse_args()

    stage = args.stage
    stage_cfg = STAGE_CONFIGS[stage]

    CONFIG = {
        "data_path": "/home/ckb2084/research/SpecPT/data/grism_specPT_v5.pkl",
        "checkpoint": args.init,
        "autoencoder": "/home/ckb2084/research/galax_spec/pretrained_weights/SpecPT_DESI_combined_autoencoder_150_new.pth",
        "model_config": str(REPO_ROOT / "configs/exp_032.yaml"),
        "output_dir": "/home/ckb2084/research/specpt-hst-sim/outputs/real_3dhst",
        "ckpt_dir": "/home/ckb2084/research/specpt-hst-sim/checkpoints/finetune_real",
        "wandb_run_id": "ejfhtjlk",
        "wandb_entity": "ckb2084-rochester-institute-of-technology",
        "wandb_project": "specpt-hst-sim",
        "batch_size": 128,
        "num_workers": 0,
        "min_snr": 2.5,
    }

    print("="*60)
    print(f"STAGE {stage}: {'Linear Probe' if stage == 1 else 'Partial Freeze'}")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = _load_and_fix_real_data(CONFIG["data_path"], min_snr=CONFIG["min_snr"])
    train_df, val_df, test_df = _split(df)

    train_dataset = HSTGrismDataset(train_df, normalize_fn=SpectrumNormalizer.zscore_normalize)
    val_dataset = HSTGrismDataset(val_df, normalize_fn=SpectrumNormalizer.zscore_normalize)
    test_dataset = HSTGrismDataset(test_df, normalize_fn=SpectrumNormalizer.zscore_normalize)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
                             shuffle=False, num_workers=CONFIG["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=CONFIG["num_workers"])

    model = _build_model(CONFIG["checkpoint"], CONFIG["autoencoder"],
                         CONFIG["model_config"], device)
    # Sanity check: are weights NaN right after loading?
    for name, p in model.named_parameters():
        if torch.isnan(p).any():
            print(f"   [WARN] {name} has NaN right after loading! "
                  f"shape={p.shape}, nan_fraction={(torch.isnan(p).float().mean()*100):.1f}%")
    model = _apply_freeze_policy(model, stage)

    # Only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=stage_cfg["lr"],
                                   weight_decay=stage_cfg["weight_decay"])
    from src.specpt.losses import NMADLoss
    criterion = NMADLoss(normalization_factor="std")

    if stage_cfg["warmup_epochs"] > 0:
        def lr_lambda(epoch):
            if epoch < stage_cfg["warmup_epochs"]:
                return epoch / stage_cfg["warmup_epochs"]
            t = epoch - stage_cfg["warmup_epochs"]
            T = stage_cfg["epochs"] - stage_cfg["warmup_epochs"]
            return 0.5 * (1 + math.cos(math.pi * t / T)) if T > 0 else 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage_cfg["epochs"], eta_min=0)

    best_ckpt_path = f"{CONFIG['ckpt_dir']}/stage{stage}_best_model.pth"
    BEST_VAL["nmad"] = 1e9
    patience_counter = 0

    print(f"\n{'='*50}")
    print(f"TRAINING (Stage {stage})")
    print(f"{'='*50}")

    # Quick diagnostic forward pass on one val batch
    model.eval()
    for X_diag, Y_diag, _, _ in val_loader:
        X_diag = X_diag.to(device)
        with torch.no_grad():
            z_pred = model(X_diag).flatten()
        loss_diag = criterion(z_pred, Y_diag.to(device))
        print(f"[diag] loss={loss_diag.item():.4f}, z_range=[{z_pred.min().item():.4f},{z_pred.max().item():.4f}]")
        break
    model.train()

    for epoch in range(1, stage_cfg["epochs"] + 1):
        t0 = time.time()
        train_loss = _train_epoch(model, train_loader, optimizer, criterion,
                                   device, stage_cfg["noise_std"])
        val_metrics, _ = _evaluate(model, val_loader, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"Epoch {epoch:2d}/{stage_cfg['epochs']}  "
              f"loss={train_loss:.4f}  "
              f"val_nmad={val_metrics['nmad']:.5f}  "
              f"val_eta={val_metrics['eta']:.2f}%  "
              f"lr={current_lr:.2e}  "
              f"{elapsed:.0f}s")

        if val_metrics["nmad"] < BEST_VAL["nmad"]:
            BEST_VAL["nmad"] = val_metrics["nmad"]
            BEST_VAL["epoch"] = epoch
            _save_checkpoint(model, optimizer, epoch, BEST_VAL["nmad"], best_ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= stage_cfg["patience"]:
                print(f"   Early stopping at epoch {epoch}")
                break

    print(f"\n{'='*50}")
    print(f"BEST VAL: NMAD={BEST_VAL['nmad']:.5f} at epoch {BEST_VAL['epoch']}")
    print(f"{'='*50}")

    # Test with best model
    print(f"\n{'='*50}")
    print(f"TESTING")
    print(f"{'='*50}")
    if not Path(best_ckpt_path).exists():
        print(f"   No checkpoint saved (best_val_nmad was NaN). Skipping test.")
        return
    test_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(test_ckpt["model_state_dict"], strict=False)
    print(f"   Loaded best checkpoint from epoch {test_ckpt['epoch']}")

    test_metrics, (t_trues, t_preds, t_ids) = _evaluate(model, test_loader, device)
    snr_map = test_df.set_index("TARGETID")["SNR"].to_dict()
    t_snr = np.array([snr_map.get(tid, np.nan) for tid in t_ids])

    print(f"   Test NMAD={test_metrics['nmad']:.5f}  RMSE={test_metrics['rmse']:.4f}  "
          f"bias={test_metrics['bias']:+.5f}  eta={test_metrics['eta']:.2f}%  "
          f"n={len(t_trues)}")

    # Per-SNR bins
    snr_bins = [2.5, 5.0, 10.0, 20.0, np.inf]
    per_snrb = []
    for i in range(len(snr_bins) - 1):
        lo, hi = snr_bins[i], snr_bins[i+1]
        mask = (t_snr >= lo) & (t_snr < hi)
        if mask.sum() > 0:
            m = compute_metrics(t_trues[mask], t_preds[mask])
            m["bin"] = f"[{lo}, {hi if np.isfinite(hi) else chr(0x221E)})"
            m["count"] = int(mask.sum())
            per_snrb.append(m)
    test_metrics["per_snrb"] = per_snrb

    figures = _make_plots(t_trues, t_preds, t_snr, test_metrics["per_zbin"],
                          per_snrb, test_metrics, stage)

    _save_outputs(t_trues, t_preds, t_snr, t_ids, test_metrics, figures, CONFIG, stage)
    _log_wandb(t_trues, t_preds, t_snr, t_ids, test_metrics, figures, CONFIG, stage)

    print("\nDONE")


if __name__ == "__main__":
    main()
