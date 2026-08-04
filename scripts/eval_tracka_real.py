#!/usr/bin/env python3
"""Evaluate a Track A redshift checkpoint on real 3D-HST grism data.

Usage:
    python scripts/eval_tracka_real.py \
        --config configs/tracka_small_z.yaml \
        --checkpoint checkpoints/tracka_small_z_best_model.pth \
        --exp-name tracka_small_z
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path("/home/ckb2084/research/specpt-hst-sim")
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader

from src.specpt.model import SpecPT, EnhancedSpecPTForRedshift, SpectrumNormalizer
from src.specpt.dataloader import HSTGrismDataset
from src.specpt.training.eval import compute_metrics

TRAIN_WAVES = np.linspace(10800.0, 17100.0, 7781)
PAD = (TRAIN_WAVES < 11000) | (TRAIN_WAVES > 16500)
REAL_DATA_PATH = "/home/ckb2084/research/SpecPT/data/grism_specPT_v5.pkl"


def _spec_from_clean(row):
    safe = np.where(row["sensitivity_resampled"] == 0, 1e-8, row["sensitivity_resampled"])
    s = (row["clean_flux_resampled"] / safe).astype(np.float32)
    s[PAD] = np.nan
    return s


def load_real_data(min_snr=2.5):
    print(f"Loading real 3D-HST data: {REAL_DATA_PATH}")
    data = pd.read_pickle(REAL_DATA_PATH)
    print(f"  Loaded {len(data)} spectra")
    data = data[data["SNR"] >= min_snr].copy()
    print(f"  After SNR >= {min_snr}: {len(data)} spectra")
    data["spec"] = [_spec_from_clean(r) for _, r in data.iterrows()]
    data.rename(columns={"grism_id": "TARGETID"}, inplace=True)
    cols = ["TARGETID", "z", "spec", "SNR"]
    print(f"  z range: [{data['z'].min():.3f}, {data['z'].max():.3f}]")
    print(f"  SNR range: [{data['SNR'].min():.2f}, {data['SNR'].max():.2f}]"
          f" (median {data['SNR'].median():.2f})")
    return data[cols].copy()


def build_model(config_path, checkpoint_path, device):
    print(f"Building model from {config_path}")
    with open(REPO_ROOT / config_path) as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model"]

    auto_model = SpecPT(
        input_size=mc["input_size"],
        d_model=mc["d_model"],
        nhead=mc["nhead"],
        num_encoder_layers=mc["num_encoder_layers"],
        num_decoder_layers=mc["num_decoder_layers"],
        dim_feedforward=mc["dim_feedforward"],
        dropout=mc["dropout"],
    )

    model = EnhancedSpecPTForRedshift(
        auto_model,
        output_features=1,
        num_mlp_blocks=mc["num_mlp_blocks"],
        mlp_dim=mc["mlp_dim"],
        dropout_rate=mc["dropout"],
    )

    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys")
    if not missing and not unexpected:
        print(f"  All {len(list(sd.keys()))} keys matched")

    for param in model.pretrained_model.parameters():
        param.requires_grad = False

    model = model.to(device).eval()
    print(f"  Model on {device}, {sum(p.numel() for p in model.parameters()):,} params")
    return model


def run_inference(model, df, batch_size=128, device="cpu"):
    print(f"Running inference on {len(df)} spectra")
    dataset = HSTGrismDataset(df, normalize_fn=SpectrumNormalizer.zscore_normalize)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    preds, trues, ids, snrs = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (X, Y, idx, t_id) in enumerate(loader):
            X = X.to(device, non_blocking=True)
            y_pred = model(X)
            preds.append(y_pred.cpu().numpy().flatten())
            trues.append(Y.numpy().flatten())
            ids.extend(t_id)
            snrs.append(Y.numpy().flatten())
            if i % 20 == 0:
                print(f"  batch {i}/{len(loader)}  ({min((i+1)*batch_size, len(df))}/{len(df)})"
                      f"  {time.time()-t0:.1f}s")

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    preds = np.clip(preds, 0.0, None)

    result = pd.DataFrame({
        "grism_id": ids,
        "z_true": trues,
        "z_pred": preds,
        "snr": df["SNR"].values.copy(),
    })
    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.1f}s ({len(df)/max(elapsed,1):.0f} spectra/s)")
    return result


def compute_all_metrics(preds_df):
    y_true = preds_df["z_true"].values
    y_pred = preds_df["z_pred"].values
    snr = preds_df["snr"].values

    overall = compute_metrics(y_true, y_pred)
    print(f"Overall (n={len(y_true)}):")
    print(f"  NMAD={overall['nmad']:.5f}  RMSE={overall['rmse']:.4f}  "
          f"bias={overall['bias']:+.5f}  eta={overall['eta']:.2f}%")

    pred_std = np.std(y_pred)
    true_std = np.std(y_true)
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    print(f"  pred_std={pred_std:.4f}  true_std={true_std:.4f}  "
          f"std_ratio={pred_std/true_std:.4f}  R2={r2:.4f}")

    mask_in = y_true >= 0.2
    in_dist = compute_metrics(y_true[mask_in], y_pred[mask_in])
    print(f"In-distribution (z>=0.2, n={mask_in.sum()}):")
    print(f"  NMAD={in_dist['nmad']:.5f}  RMSE={in_dist['rmse']:.4f}  "
          f"bias={in_dist['bias']:+.5f}  eta={in_dist['eta']:.2f}%")

    snr_bins = [2.5, 5.0, 10.0, 20.0, np.inf]
    per_snrb = []
    for i in range(len(snr_bins) - 1):
        lo, hi = snr_bins[i], snr_bins[i+1]
        if np.isinf(hi):
            mask = snr >= lo
            label = f"[{lo:.0f}, inf)"
        else:
            mask = (snr >= lo) & (snr < hi)
            label = f"[{lo:.0f}, {hi:.0f})"
        if mask.sum() > 0:
            m = compute_metrics(y_true[mask], y_pred[mask])
            m["bin"] = label
            m["count"] = int(mask.sum())
            per_snrb.append(m)
            print(f"  SNR {label}: n={m['count']:5d}  NMAD={m['nmad']:.5f}  eta={m['eta']:.2f}%")

    return {
        "overall": overall,
        "in_dist_z02": in_dist,
        "per_snrb": per_snrb,
        "n_total": int(len(y_true)),
        "n_in_dist": int(mask_in.sum()),
        "pred_std": float(pred_std),
        "true_std": float(true_std),
        "std_ratio": float(pred_std / true_std),
        "r2": float(r2),
    }


def make_plots(preds_df, metrics, exp_name, output_dir):
    print("Generating plots")
    y_true = preds_df["z_true"].values
    y_pred = preds_df["z_pred"].values
    snr = preds_df["snr"].values
    delz = (y_pred - y_true) / (1 + y_true)
    frac = 0.15
    min_lim = -0.05
    max_lim = np.ceil(y_true.max()) + 0.1
    figures = {}

    fig = plt.figure(figsize=(5.7, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 1], wspace=0.1, hspace=0.01)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax1.scatter(y_true, y_pred, marker=".", s=10, c="blue", alpha=0.6)
    ax1.plot([min_lim, max_lim], [min_lim, max_lim], c="k", zorder=9)
    x_line = np.linspace(0, max_lim)
    ax1.plot(x_line, (1+x_line)*frac+x_line, c="k", linestyle="dotted", zorder=9)
    ax1.plot(x_line, (1+x_line)*-frac+x_line, c="k", linestyle="dotted", zorder=9)
    ax1.text(min_lim+0.07, max_lim-0.1, f"NMAD: {metrics['overall']['nmad']:.4f}", fontsize=11)
    ax1.text(min_lim+0.07, max_lim-0.2, f"\u03b7: {metrics['overall']['eta']:.2f}%", fontsize=11)
    ax1.text(min_lim+0.07, max_lim-0.3, f"R²: {metrics['r2']:.4f}", fontsize=11)
    ax1.set_ylabel(r"Predicted $z$", fontsize=16)
    ax1.set_xlim(min_lim, max_lim)
    ax1.set_ylim(min_lim, max_lim)
    ax1.set_title(f"Real 3D-HST: {exp_name}", fontsize=12)
    ax1.minorticks_on()
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.scatter(y_true, delz, marker=".", s=10, c="blue", alpha=0.6)
    ax2.axhline(y=0, c="k", zorder=9)
    ax2.axhline(y=-frac, c="k", linestyle="dotted", zorder=9)
    ax2.axhline(y=frac, c="k", linestyle="dotted", zorder=9)
    ax2.set_ylabel(r"$\Delta z/(1+z)$", fontsize=14)
    ax2.set_xlabel(r"True $z$", fontsize=16)
    ax2.minorticks_on()
    plt.tight_layout()
    figures["scatter_standard"] = fig
    plt.close(fig)

    fig = plt.figure(figsize=(6.0, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 1], wspace=0.1, hspace=0.01)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    sc = ax1.scatter(y_true, y_pred, marker=".", s=10, c=snr,
                     cmap="viridis", alpha=0.6, norm=matplotlib.colors.LogNorm())
    ax1.plot([min_lim, max_lim], [min_lim, max_lim], c="k", zorder=9)
    ax1.plot(x_line, (1+x_line)*frac+x_line, c="k", linestyle="dotted", zorder=9)
    ax1.plot(x_line, (1+x_line)*-frac+x_line, c="k", linestyle="dotted", zorder=9)
    cbar = plt.colorbar(sc, ax=ax1, pad=0.02)
    cbar.set_label("SNR", fontsize=10)
    ax1.set_ylabel(r"Predicted $z$", fontsize=16)
    ax1.set_xlim(min_lim, max_lim)
    ax1.set_ylim(min_lim, max_lim)
    ax1.set_title(f"Real 3D-HST: {exp_name} (SNR-colored)", fontsize=12)
    ax1.minorticks_on()
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.scatter(y_true, delz, marker=".", s=10, c="blue", alpha=0.6)
    ax2.axhline(y=0, c="k", zorder=9)
    ax2.axhline(y=-frac, c="k", linestyle="dotted", zorder=9)
    ax2.axhline(y=frac, c="k", linestyle="dotted", zorder=9)
    ax2.set_ylabel(r"$\Delta z/(1+z)$", fontsize=14)
    ax2.set_xlabel(r"True $z$", fontsize=16)
    ax2.minorticks_on()
    plt.tight_layout()
    figures["scatter_snr_colored"] = fig
    plt.close(fig)

    os.makedirs(output_dir, exist_ok=True)
    for name, fig in figures.items():
        path = os.path.join(output_dir, f"{exp_name}_{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    return figures


def main():
    parser = argparse.ArgumentParser(description="Evaluate Track A on real 3D-HST")
    parser.add_argument("--config", required=True, help="Track A redshift config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to redshift checkpoint")
    parser.add_argument("--exp-name", default=None, help="Experiment name for output files")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-snr", type=float, default=2.5)
    parser.add_argument("--output-dir", default="outputs/real_3dhst_tracka")
    parser.add_argument("--wandb-project", default="specpt-hst-sim-z")
    parser.add_argument("--wandb-entity", default="ckb2084-rochester-institute-of-technology")
    args = parser.parse_args()

    if args.exp_name is None:
        args.exp_name = Path(args.config).stem

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = load_real_data(min_snr=args.min_snr)
    model = build_model(args.config, args.checkpoint, device)
    preds_df = run_inference(model, data, batch_size=args.batch_size, device=device)
    metrics = compute_all_metrics(preds_df)
    output_dir = os.path.join(args.output_dir, args.exp_name)
    figures = make_plots(preds_df, metrics, args.exp_name, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    preds_path = os.path.join(output_dir, f"{args.exp_name}_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"Saved: {preds_path}")

    metrics_path = os.path.join(output_dir, f"{args.exp_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {metrics_path}")

    try:
        import wandb
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=f"real_{args.exp_name}",
            config={"config": args.config, "checkpoint": args.checkpoint},
        )
        wandb.log(metrics)
        for name, fig in figures.items():
            wandb.log({name: wandb.Image(fig)})
        wandb.finish()
    except Exception as e:
        print(f"W&B logging failed (non-fatal): {e}")

    print("EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
