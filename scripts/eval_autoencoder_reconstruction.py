#!/usr/bin/env python3
"""Evaluate trained autoencoder reconstruction on real 3D-HST grism data.

Usage:
    python scripts/eval_autoencoder_reconstruction.py \
        --ckpt checkpoints/autoencoder_regrid_autoencoder_best.pth \
        --data /home/ckb2084/research/SpecPT/data/grism_specPT_v5.pkl \
        --run-name rose-dragon-2 \
        --output-dir outputs/autoencoder_reconstruction
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.specpt.model import SpecPT, SpectrumNormalizer
from src.specpt.dataloader import AutoencoderDataset


def load_checkpoint(path, model, device):
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        print(f"   Full checkpoint: epoch={ckpt.get('epoch')}, "
              f"best_val_loss={ckpt.get('best_val_loss'):.4f}")
    else:
        state_dict = ckpt
        print(f"   Weights-only checkpoint")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"   WARNING: {len(missing)} missing keys (first 5: {missing[:5]})")
    if unexpected:
        print(f"   WARNING: {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")
    if not missing and not unexpected:
        print(f"   All keys matched")
    model.to(device)
    model.eval()
    return model


def load_real_data(data_path, min_snr=2.5):
    print(f"Loading real 3D-HST data: {data_path}")
    data = pd.read_pickle(data_path)
    print(f"   Loaded {len(data)} spectra")

    data = data[data["SNR"] >= min_snr].copy()
    print(f"   After SNR >= {min_snr} filter: {len(data)} spectra")

    safe_sensitivity = data["sensitivity_resampled"].apply(
        lambda arr: np.where(arr == 0, 1e-8, arr)
    )
    data["spec"] = data["clean_flux_resampled"] / safe_sensitivity
    data.rename(columns={"grism_id": "TARGETID"}, inplace=True)

    print(f"   z range: [{data['z'].min():.3f}, {data['z'].max():.3f}]")
    print(f"   SNR range: [{data['SNR'].min():.2f}, {data['SNR'].max():.2f}]")
    return data


def forward_with_latent(model, x):
    x = x.unsqueeze(1)
    x = model.forward_conv(x)
    x = x.flatten(start_dim=1)
    latent = model.proj_to_d_model(x)
    x = latent.unsqueeze(0)
    x = model.transformer_encoder(x)
    x = model.transformer_decoder(x, x)
    x = x.squeeze(0)
    x = F.relu(model.linear1(x))
    x = model.linear2(x)
    return x, latent


def run_inference(model, dataset, batch_size, device):
    print(f"Running inference on {len(dataset)} spectra")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds = []
    all_true = []
    all_latents = []
    all_ids = []
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, (X, Y, idx, t_id) in enumerate(loader):
            X = X.to(device, non_blocking=True)
            preds, latents = forward_with_latent(model, X)
            all_preds.append(preds.cpu().numpy())
            all_true.append(Y.cpu().numpy())
            all_latents.append(latents.cpu().numpy())
            all_ids.extend(t_id)

            if batch_idx % 20 == 0:
                done = batch_idx * batch_size
                print(f"   batch {batch_idx}  ({done}/{len(dataset)})  "
                      f"{time.time()-t0:.1f}s")

    preds = np.concatenate(all_preds, axis=0)
    true = np.concatenate(all_true, axis=0)
    latents = np.concatenate(all_latents, axis=0)

    print(f"   Done in {time.time()-t0:.1f}s")
    print(f"   Predictions: {preds.shape}, Latents: {latents.shape}")
    return preds, true, latents, all_ids


def compute_metrics(preds, true, latents, snr_values, redshifts):
    print("Computing metrics")
    n = len(true)
    mse_per_sample = np.mean((preds - true) ** 2, axis=1)
    cos_sim_per_sample = np.array([
        np.dot(true[i], preds[i]) / (
            np.linalg.norm(true[i]) * np.linalg.norm(preds[i]) + 1e-8
        )
        for i in range(n)
    ])
    latent_norm = np.linalg.norm(latents, axis=1)
    latent_var = np.var(latents, axis=1)

    print(f"   MSE: mean={mse_per_sample.mean():.6f}, "
          f"median={np.median(mse_per_sample):.6f}")
    print(f"   Cosine sim: mean={cos_sim_per_sample.mean():.6f}, "
          f"median={np.median(cos_sim_per_sample):.6f}")
    print(f"   Latent norm: mean={latent_norm.mean():.4f}, "
          f"std={latent_norm.std():.4f}")
    print(f"   Latent variance: mean={latent_var.mean():.6f}, "
          f"std={latent_var.std():.6f}")

    df = pd.DataFrame({
        "TARGETID": [str(i) for i in range(n)],
        "SNR": snr_values,
        "z": redshifts,
        "mse": mse_per_sample,
        "cosine_similarity": cos_sim_per_sample,
        "latent_norm": latent_norm,
        "latent_variance": latent_var,
    })
    return df


def _binned_medians(x, y, n_bins=20):
    bins = np.logspace(np.log10(x.min() + 1e-10), np.log10(x.max()), n_bins)
    centers = (bins[:-1] + bins[1:]) / 2
    medians = []
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        medians.append(np.median(y[mask]) if mask.sum() > 0 else np.nan)
    return centers, medians


def make_plots(metrics_df, true, preds, latents, run_name, output_dir):
    print("Generating plots")
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{run_name}"

    snr = metrics_df["SNR"].values
    mse = metrics_df["mse"].values
    cos_sim = metrics_df["cosine_similarity"].values
    latent_norm = metrics_df["latent_norm"].values
    latent_var = metrics_df["latent_variance"].values
    n = len(metrics_df)

    # --- 1. Metrics vs SNR (2x2 panel) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.scatter(snr, mse, s=5, alpha=0.4, c="steelblue")
    cx, cy = _binned_medians(snr, mse)
    ax.plot(cx, cy, "r-", lw=2, label="Binned median")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SNR")
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction MSE vs SNR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(snr, cos_sim, s=5, alpha=0.4, c="darkorange")
    cx, cy = _binned_medians(snr, cos_sim)
    ax.plot(cx, cy, "r-", lw=2, label="Binned median")
    ax.set_xscale("log")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Reconstruction Cosine Similarity vs SNR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(snr, latent_norm, s=5, alpha=0.4, c="forestgreen")
    cx, cy = _binned_medians(snr, latent_norm)
    ax.plot(cx, cy, "r-", lw=2, label="Binned median")
    ax.set_xscale("log")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Latent Norm (L2)")
    ax.set_title("Latent Representation Norm vs SNR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(snr, latent_var, s=5, alpha=0.4, c="purple")
    cx, cy = _binned_medians(snr, latent_var)
    ax.plot(cx, cy, "r-", lw=2, label="Binned median")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Latent Variance (across dims)")
    ax.set_title("Latent Variance vs SNR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Autoencoder Reconstruction on Real 3D-HST ({run_name})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_metrics_vs_snr.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   Saved: _metrics_vs_snr.png")

    # --- 2. Latent norm vs SNR colored by MSE ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(snr, latent_norm, s=10, c=mse, cmap="viridis",
                    alpha=0.6, norm=LogNorm())
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("MSE")
    ax.set_xscale("log")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Latent Norm (L2)")
    ax.set_title(f"Latent Norm vs SNR, colored by MSE ({run_name})")
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{prefix}_latent_norm_vs_snr.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   Saved: _latent_norm_vs_snr.png")

    # --- 3. Latent variance vs SNR colored by cosine similarity ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(snr, latent_var, s=10, c=cos_sim, cmap="RdYlGn",
                    alpha=0.6, norm=LogNorm(), vmin=0.5, vmax=1.0)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Cosine Similarity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Latent Variance (across dims)")
    ax.set_title(f"Latent Variance vs SNR, colored by Cosine Similarity ({run_name})")
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{prefix}_latent_variance_vs_snr.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   Saved: _latent_variance_vs_snr.png")

    # --- 4. Best / worst reconstructions ---
    n_examples = 5
    best_idx = np.argsort(mse)[:n_examples]
    worst_idx = np.argsort(mse)[-n_examples:][::-1]

    fig, axes = plt.subplots(2, n_examples, figsize=(5 * n_examples, 8))
    for col, idx in enumerate(best_idx):
        ax = axes[0, col]
        ax.plot(true[idx], alpha=0.7, label="Original", lw=1)
        ax.plot(preds[idx], alpha=0.7, label="Reconstructed", lw=1)
        ax.set_title(f"BEST {col+1}\nMSE={mse[idx]:.6f}  SNR={snr[idx]:.1f}", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Normalized flux")

    for col, idx in enumerate(worst_idx):
        ax = axes[1, col]
        ax.plot(true[idx], alpha=0.7, label="Original", lw=1)
        ax.plot(preds[idx], alpha=0.7, label="Reconstructed", lw=1)
        ax.set_title(f"WORST {col+1}\nMSE={mse[idx]:.6f}  SNR={snr[idx]:.1f}", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Normalized flux")

    plt.suptitle(f"Autoencoder Reconstruction: Best vs Worst ({run_name})", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_reconstruction_examples.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   Saved: _reconstruction_examples.png")

    # --- 5. Residual heatmap ---
    residuals = true - preds
    sort_by_snr = np.argsort(snr)
    fig, ax = plt.subplots(figsize=(14, 6))
    vlim = np.percentile(np.abs(residuals), 99)
    im = ax.imshow(residuals[sort_by_snr], aspect="auto", cmap="RdBu_r",
                   vmin=-vlim, vmax=vlim, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Residual (original - reconstruction)")
    ax.set_xlabel("Pixel")
    ax.set_ylabel(f"Spectrum (sorted by SNR, n={n})")
    ax.set_title(f"Reconstruction Residual Heatmap ({run_name})")
    twin = ax.twinx()
    twin.set_ylim(ax.get_ylim())
    twin.set_ylabel("SNR \u2192", fontsize=10)
    snr_sorted = np.sort(snr)
    y_pos = np.linspace(0, n - 1, 5).astype(int)
    twin.set_yticks(y_pos)
    twin.set_yticklabels([f"{snr_sorted[i]:.1f}" for i in y_pos])
    plt.savefig(os.path.join(output_dir, f"{prefix}_residual_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   Saved: _residual_heatmap.png")

    # --- 6. UMAP of latent space ---
    print("   Running UMAP on latent space...")
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                            random_state=42, metric="euclidean")
        embedding = reducer.fit_transform(latents)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        ax = axes[0]
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=snr,
                        cmap="viridis", alpha=0.6, s=10, norm=LogNorm())
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label("SNR")
        ax.set_title("Latent Space UMAP (colored by SNR)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        ax = axes[1]
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=mse,
                        cmap="inferno", alpha=0.6, s=10, norm=LogNorm())
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label("MSE")
        ax.set_title("Latent Space UMAP (colored by MSE)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        plt.suptitle(f"Autoencoder Latent Space UMAP ({run_name})", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_latent_umap.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("   Saved: _latent_umap.png")
    except ImportError:
        print("   umap not installed \u2014 skipping UMAP plot")

    print("   All plots generated")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate autoencoder reconstruction on real 3D-HST grism data"
    )
    parser.add_argument("--ckpt", required=True,
                        help="Path to autoencoder checkpoint (.pth)")
    parser.add_argument("--data", required=True,
                        help="Path to real 3D-HST pickle file")
    parser.add_argument("--run-name", default="autoencoder",
                        help="W&B run name for output filename prefix")
    parser.add_argument("--output-dir", default="outputs/autoencoder_reconstruction",
                        help="Output directory for plots and CSV")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-snr", type=float, default=2.5)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  AUTOENCODER RECONSTRUCTION EVAL ON REAL 3D-HST DATA")
    print(f"  Run: {args.run_name}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SpecPT(
        input_size=7781, d_model=512, nhead=8,
        num_encoder_layers=3, num_decoder_layers=3,
        dim_feedforward=2048, dropout=0.1,
    )
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    model = load_checkpoint(args.ckpt, model, device)

    df = load_real_data(args.data, min_snr=args.min_snr)

    dataset = AutoencoderDataset(df, normalize_fn=SpectrumNormalizer.zscore_normalize)
    print(f"Dataset: {len(dataset)} spectra after normalization filter")

    preds, true, latents, ids = run_inference(model, dataset, args.batch_size, device)

    df_snr = df.set_index("TARGETID").loc[
        [dataset.t_id[i] for i in range(len(dataset))]
    ]
    snr_values = df_snr["SNR"].values
    redshift_values = df_snr["z"].values

    metrics_df = compute_metrics(preds, true, latents, snr_values, redshift_values)
    metrics_df["TARGETID"] = ids

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{args.run_name}_eval_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    make_plots(metrics_df, true, preds, latents, args.run_name, str(output_dir))

    print("=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
