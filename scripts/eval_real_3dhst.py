"""Evaluate a trained SpecPT checkpoint on real 3D-HST G141 grism data.

Real data: /home/ckb2084/research/SpecPT/data/grism_specPT_v5.pkl
Reference checkpoint: exp_032_best_model.pth (val_nmad 0.00785 on synthetic Q1)

Domain shift (NOT compensated — the test's purpose):
  Training grid: 10,311-17,465 A, 0.92 A/pix, 7781 pix
  Real grid:    10,800-17,100 A, 0.81 A/pix, 7781 pix

Preprocessing: sensitivity-corrected clean_flux, SNR>=2.5, no interpolation.
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

CONFIG = {
    "data_path": "/home/ckb2084/research/SpecPT/data/grism_specPT_v5.pkl",
    "checkpoint": "/home/ckb2084/research/specpt-hst-sim/checkpoints/exp_032_best_model.pth",
    "autoencoder": "/home/ckb2084/research/galax_spec/pretrained_weights/SpecPT_DESI_combined_autoencoder_150_new.pth",
    "model_config": "configs/exp_032.yaml",
    "exp_name": "exp_032",
    "wandb_run_id": "ejfhtjlk",
    "wandb_entity": "ckb2084-rochester-institute-of-technology",
    "wandb_project": "specpt-hst-sim",
    "output_dir": "/home/ckb2084/research/specpt-hst-sim/outputs/real_3dhst",
    "batch_size": 128,
    "min_snr": 2.5,
    "z_bins": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    "snr_bins": [2.5, 5.0, 10.0, 20.0, np.inf],
}


def _load_real_data(cfg):
    print(f"[1/6] Loading real 3D-HST data: {cfg['data_path']}")
    data = pd.read_pickle(cfg["data_path"])
    print(f"   Loaded {len(data)} spectra")

    data = data[data["SNR"] >= cfg["min_snr"]].copy()
    print(f"   After SNR >= {cfg['min_snr']} filter: {len(data)} spectra")

    data_subset = data.copy()
    safe_sensitivity = data_subset["sensitivity_resampled"].apply(
        lambda arr: np.where(arr == 0, 1e-8, arr)
    )
    data_subset["flux_sensitivity"] = data_subset["clean_flux_resampled"] / safe_sensitivity
    data_subset.rename(
        columns={"flux_sensitivity": "spec", "grism_id": "TARGETID"}, inplace=True
    )

    sample_spec = data_subset["spec"].iloc[0]
    print(f"   spec shape: {sample_spec.shape}, dtype: {sample_spec.dtype}")
    nz = data_subset["spec"].apply(lambda arr: np.count_nonzero(arr))
    print(f"   non-zero pixels: median={nz.median():.0f}, min={nz.min()}, max={nz.max()}")
    vals = data_subset["spec"].apply(lambda a: a[a != 0])
    print(f"   spec range (nonzero): {vals.apply(np.min).min():.4e} to {vals.apply(np.max).max():.4e}")
    print(f"   z range: [{data_subset['z'].min():.3f}, {data_subset['z'].max():.3f}]")
    print(f"   SNR range: [{data_subset['SNR'].min():.2f}, {data_subset['SNR'].max():.2f}]"
          f" (median {data_subset['SNR'].median():.2f})")
    return data_subset


def _build_model(cfg, device):
    print(f"[2/6] Building model from {cfg['model_config']}")
    with open(REPO_ROOT / cfg["model_config"]) as f:
        model_cfg = yaml.safe_load(f)["model"]

    auto_model = SpecPT(
        input_size=model_cfg["input_size"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_encoder_layers=model_cfg["num_encoder_layers"],
        num_decoder_layers=model_cfg["num_decoder_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
    )

    model = EnhancedSpecPTForRedshift(
        auto_model,
        output_features=1,
        num_mlp_blocks=model_cfg["num_mlp_blocks"],
        mlp_dim=model_cfg["mlp_dim"],
        dropout_rate=model_cfg["dropout"],
    )

    # Full checkpoint contains pretrained_model + encoder + mlp + head weights
    ckpt = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    print(f"   Checkpoint: {len(ckpt['model_state_dict'])} keys, "
          f"epoch {ckpt.get('epoch')}, best_val_loss {ckpt.get('best_val_loss'):.4f}")

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"   WARNING: {len(missing)} missing keys (first 5: {missing[:5]})")
    if unexpected:
        print(f"   WARNING: {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")
    if not missing and not unexpected:
        print(f"   All 237 keys matched")

    for param in model.pretrained_model.parameters():
        param.requires_grad = False

    model = model.to(device)
    model.eval()
    print(f"   Model on {device}, {sum(p.numel() for p in model.parameters()):,} total params")
    return model


def _run_inference(model, df, cfg, device):
    print(f"[3/6] Running inference on {len(df)} spectra")
    dataset = HSTGrismDataset(df, normalize_fn=SpectrumNormalizer.zscore_normalize)
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=0, pin_memory=True,
    )

    preds_list = []
    trues_list = []
    ids_list = []
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (X, Y, idx, t_id) in enumerate(loader):
            X = X.to(device, non_blocking=True)
            y_pred = model(X)
            preds_list.append(y_pred.cpu().numpy().flatten())
            trues_list.append(Y.numpy().flatten())
            ids_list.extend(t_id)
            if batch_idx % 20 == 0:
                done = batch_idx * cfg["batch_size"]
                print(f"   batch {batch_idx}/{len(loader)}  ({done}/{len(df)})  {time.time()-t0:.1f}s")

    preds = np.concatenate(preds_list)
    trues = np.concatenate(trues_list)
    preds = np.clip(preds, 0.0, None)

    result = pd.DataFrame({
        "grism_id": ids_list,
        "z_true": trues,
        "z_pred": preds,
        "snr": df["SNR"].values.copy(),
    })
    print(f"   Inference done in {time.time()-t0:.1f}s ({len(df)/max(time.time()-t0, 1):.0f} spectra/s)")
    return result, dataset


def _compute_metrics(preds_df, cfg):
    print(f"[4/6] Computing metrics")
    y_true = preds_df["z_true"].values
    y_pred = preds_df["z_pred"].values
    snr = preds_df["snr"].values

    overall = compute_metrics(y_true, y_pred)
    print(f"   Overall (n={len(y_true)}):")
    print(f"     NMAD={overall['nmad']:.5f}  RMSE={overall['rmse']:.4f}  "
          f"bias={overall['bias']:+.5f}  eta={overall['eta']:.2f}%")

    mask_in = y_true >= 0.2
    in_dist = compute_metrics(y_true[mask_in], y_pred[mask_in])
    print(f"   In-distribution (z>=0.2, n={mask_in.sum()}):")
    print(f"     NMAD={in_dist['nmad']:.5f}  RMSE={in_dist['rmse']:.4f}  "
          f"bias={in_dist['bias']:+.5f}  eta={in_dist['eta']:.2f}%")

    per_zbin = []
    zb = cfg["z_bins"]
    for i in range(len(zb) - 1):
        lo, hi = zb[i], zb[i + 1]
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() > 0:
            m = compute_metrics(y_true[mask], y_pred[mask])
            m["bin"] = f"[{lo:.1f}, {hi:.1f})"
            m["count"] = int(mask.sum())
            per_zbin.append(m)
            print(f"   z-bin {m['bin']}: n={m['count']:5d}  "
                  f"NMAD={m['nmad']:.5f}  eta={m['eta']:.2f}%  rmse={m['rmse']:.3f}")

    per_snrb = []
    sb = cfg["snr_bins"]
    for i in range(len(sb) - 1):
        lo, hi = sb[i], sb[i + 1]
        if np.isinf(hi):
            mask = snr >= lo
            bin_label = f"[{lo:.0f}, {chr(0x221E)})"
        else:
            mask = (snr >= lo) & (snr < hi)
            bin_label = f"[{lo:.0f}, {hi:.0f})"
        if mask.sum() > 0:
            m = compute_metrics(y_true[mask], y_pred[mask])
            m["bin"] = bin_label
            m["count"] = int(mask.sum())
            per_snrb.append(m)
            print(f"   SNR-bin {m['bin']}: n={m['count']:5d}  "
                  f"NMAD={m['nmad']:.5f}  eta={m['eta']:.2f}%  rmse={m['rmse']:.3f}")

    return {
        "overall": overall,
        "in_dist_z02": in_dist,
        "per_zbin": per_zbin,
        "per_snrb": per_snrb,
        "n_total": int(len(y_true)),
        "n_in_dist": int(mask_in.sum()),
    }


def _make_plots(preds_df, metrics, df, dataset, model, device):
    print(f"[5/6] Generating plots")
    y_true = preds_df["z_true"].values
    y_pred = preds_df["z_pred"].values
    snr = preds_df["snr"].values
    delz = (y_pred - y_true) / (1 + y_true)
    frac = 0.15
    min_lim = -0.05
    max_lim = np.ceil(y_true.max()) + 0.1
    figures = {}

    # --- (1) Standard scatter (matches training style exactly) ---
    fig = plt.figure(figsize=(5.7, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 1], wspace=0.1, hspace=0.01)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax1.scatter(y_true, y_pred, marker=".", s=10, c="blue", alpha=0.6)
    ax1.plot([min_lim, max_lim], [min_lim, max_lim], c="k", zorder=9)
    x_line = np.linspace(0, max_lim)
    ax1.plot(x_line, (1 + x_line) * frac + x_line, c="k", linestyle="dotted", zorder=9)
    ax1.plot(x_line, (1 + x_line) * -frac + x_line, c="k", linestyle="dotted", zorder=9)
    ax1.text(min_lim + 0.07, max_lim - 0.1,
             f"NMAD: {metrics['overall']['nmad']:.4f}", fontsize=11)
    ax1.text(min_lim + 0.07, max_lim - 0.2,
             f"\u03b7: {metrics['overall']['eta']:.2f}%", fontsize=11)
    ax1.set_ylabel(r"Predicted $z$", fontsize=16)
    ax1.set_xlim(min_lim, max_lim)
    ax1.set_ylim(min_lim, max_lim)
    ax1.set_title("Real 3D-HST (AEGIS): True vs Predicted Redshift", fontsize=12)
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
    figures["real_3dhst_scatter_standard"] = fig
    plt.close(fig)
    print("   done: scatter_standard")

    # --- (2) SNR-colored scatter ---
    fig = plt.figure(figsize=(6.0, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 1], wspace=0.1, hspace=0.01)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    sc = ax1.scatter(y_true, y_pred, marker=".", s=10, c=snr,
                     cmap="viridis", alpha=0.6, norm=matplotlib.colors.LogNorm())
    ax1.plot([min_lim, max_lim], [min_lim, max_lim], c="k", zorder=9)
    x_line = np.linspace(0, max_lim)
    ax1.plot(x_line, (1 + x_line) * frac + x_line, c="k", linestyle="dotted", zorder=9)
    ax1.plot(x_line, (1 + x_line) * -frac + x_line, c="k", linestyle="dotted", zorder=9)
    cbar = plt.colorbar(sc, ax=ax1, pad=0.02)
    cbar.set_label("SNR", fontsize=10)
    ax1.set_ylabel(r"Predicted $z$", fontsize=16)
    ax1.set_xlim(min_lim, max_lim)
    ax1.set_ylim(min_lim, max_lim)
    ax1.set_title("Real 3D-HST (AEGIS): Predicted z Colored by SNR", fontsize=12)
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
    figures["real_3dhst_scatter_snr_colored"] = fig
    plt.close(fig)
    print("   done: scatter_snr_colored")

    # --- (3) Per-z-bin bar chart ---
    zb = metrics["per_zbin"]
    fig, ax = plt.subplots(figsize=(8, 4))
    labels_z = [b["bin"] for b in zb]
    nmads_z = [b["nmad"] for b in zb]
    counts_z = [b["count"] for b in zb]
    bars = ax.bar(labels_z, nmads_z, color="steelblue", alpha=0.8, edgecolor="k")
    for bar, n in zip(bars, counts_z):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"n={n}", ha="center", fontsize=8)
    ax.set_xlabel("Redshift bin (z_true)")
    ax.set_ylabel("NMAD")
    ax.set_title("Real 3D-HST: NMAD per Redshift Bin")
    ax.axhline(y=0.020, c="r", linestyle="--", alpha=0.5, label="Target NMAD=0.02")
    ax.legend()
    plt.tight_layout()
    figures["real_3dhst_per_z_bin"] = fig
    plt.close(fig)
    print("   done: per_z_bin")

    # --- (4) Per-SNR bin bar chart ---
    sb = metrics["per_snrb"]
    fig, ax = plt.subplots(figsize=(8, 4))
    labels_s = [b["bin"] for b in sb]
    nmads_s = [b["nmad"] for b in sb]
    counts_s = [b["count"] for b in sb]
    bars = ax.bar(labels_s, nmads_s, color="darkorange", alpha=0.8, edgecolor="k")
    for bar, n in zip(bars, counts_s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"n={n}", ha="center", fontsize=8)
    ax.set_xlabel("SNR bin")
    ax.set_ylabel("NMAD")
    ax.set_title("Real 3D-HST: NMAD per SNR Bin")
    ax.axhline(y=0.020, c="r", linestyle="--", alpha=0.5, label="Target NMAD=0.02")
    ax.legend()
    plt.tight_layout()
    figures["real_3dhst_per_snr_bin"] = fig
    plt.close(fig)
    print("   done: per_snr_bin")

    # --- (5) Outlier diagnostic ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(delz, bins=200, range=(-1, 1), color="steelblue", alpha=0.7, edgecolor="k")
    ax.axvspan(-frac, frac, alpha=0.15, color="green",
               label=f"Non-catastrophic ($|\\Delta z| < {frac}$)")
    ax.set_xlabel(r"$\Delta z / (1+z)$")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Real 3D-HST: Residual Distribution  "
        f"($\\eta={metrics['overall']['eta']:.2f}\\%$ catastrophic)")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    figures["real_3dhst_outlier_diagnostic"] = fig
    plt.close(fig)
    print("   done: outlier_diagnostic")

    # --- (6) Examples: 4 best + 8 random + 4 worst ---
    abs_delz = np.abs(delz)
    best_idx = np.argsort(abs_delz)[:4]
    worst_idx = np.argsort(abs_delz)[-4:][::-1]
    rng = np.random.default_rng(42)
    random_idx = rng.choice(len(y_true), size=8, replace=False)
    sel_idx = np.concatenate([best_idx, random_idx, worst_idx])
    sel_labels = (["BEST"] * 4) + (["random"] * 8) + (["WORST"] * 4)

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for ax, idx, lbl in zip(axes.flat, sel_idx, sel_labels):
        norm_spec = dataset.X[idx].numpy()
        wave = np.array(df.iloc[idx]["wavelength_resampled"])
        zt = y_true[idx]
        zp = y_pred[idx]
        ax.plot(wave, norm_spec, c="k", lw=0.5)
        ax.set_title(
            f"{lbl}  z_true={zt:.3f}  z_pred={zp:.3f}  "
            f"\u0394z={delz[idx]:+.3f}  SNR={snr[idx]:.1f}",
            fontsize=9,
        )
        ax.set_xlabel("Wavelength (A)")
        ax.set_ylabel("Norm. flux")
        ax.axvspan(11000, 16500, alpha=0.05, color="green")
    plt.suptitle("Real 3D-HST: Best / Random / Worst Predictions (sorted by |\u0394z|)", fontsize=13)
    plt.tight_layout()
    figures["real_3dhst_examples"] = fig
    plt.close(fig)
    print("   done: examples")

    # --- (7) UMAP 3D ---
    print("   Extracting encoder features for UMAP...")
    import umap

    features_list = []
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"],
                        shuffle=False, num_workers=0)
    with torch.no_grad():
        for X, _, _, _ in loader:
            X = X.to(device)
            x = X.unsqueeze(1)
            x = model.pretrained_model.forward_conv(x)
            x = x.flatten(start_dim=1)
            x = model.proj_to_d_model(x)
            x = x.unsqueeze(0)
            encoded = model.encoder(x)
            encoded = encoded.squeeze(0)
            attn_out, _ = model.attention(encoded, encoded, encoded)
            x = attn_out + encoded
            feat = model.mlp_blocks(x)
            features_list.append(feat.cpu().numpy())

    features = np.concatenate(features_list, axis=0)
    print(f"   Features shape: {features.shape}. Fitting UMAP (takes 1-2 min)...")
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                        random_state=42, metric="euclidean")
    embedding = reducer.fit_transform(features)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1], embedding[:, 2],
        c=y_true, cmap="seismic", alpha=0.6, s=20,
        edgecolors="w", linewidth=0.5,
    )
    ax.set_title("Real 3D-HST: 3D UMAP of Encoder Features (colored by $z_{true}$)",
                 fontsize=12, pad=20)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("$z_{true}$", fontsize=12)
    plt.tight_layout()
    figures["real_3dhst_umap_3d"] = fig
    plt.close(fig)
    print("   done: umap_3d")

    return figures


def _save_outputs(preds_df, metrics, figures, cfg):
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / f"{cfg['exp_name']}_real_eval.csv"
    preds_df.to_csv(csv_path, index=False)
    print(f"   CSV: {csv_path} ({len(preds_df)} rows)")

    serializable = {
        "overall": metrics["overall"],
        "in_dist_z02": metrics["in_dist_z02"],
        "per_zbin": metrics["per_zbin"],
        "per_snrb": metrics["per_snrb"],
        "n_total": metrics["n_total"],
        "n_in_dist": metrics["n_in_dist"],
    }

    def _to_python(obj):
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_python(x) for x in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    metrics_path = out / f"{cfg['exp_name']}_real_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(_to_python(serializable), f, indent=2)
    print(f"   Metrics: {metrics_path}")

    for name, fig in figures.items():
        png_path = out / f"{cfg['exp_name']}_{name}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"   Plot: {png_path}")


def _log_wandb(preds_df, metrics, figures, cfg):
    print(f"[6/6] Logging to W&B (run {cfg['wandb_run_id']})")
    import wandb

    run = wandb.init(
        id=cfg["wandb_run_id"],
        resume="allow",
        project=cfg["wandb_project"],
        entity=cfg["wandb_entity"],
    )

    overall = metrics["overall"]
    in_dist = metrics["in_dist_z02"]
    wandb.log({
        "real_3dhst_nmad": overall["nmad"],
        "real_3dhst_rmse": overall["rmse"],
        "real_3dhst_bias": overall["bias"],
        "real_3dhst_eta": overall["eta"],
        "real_3dhst_nmad_in_dist_z02": in_dist["nmad"],
        "real_3dhst_eta_in_dist_z02": in_dist["eta"],
        "real_3dhst_n_total": metrics["n_total"],
        "real_3dhst_n_in_dist": metrics["n_in_dist"],
    })

    for name, fig in figures.items():
        wandb.log({name: wandb.Image(fig)})

    table = wandb.Table(dataframe=preds_df)
    wandb.log({"real_3dhst_predictions": table})

    artifact = wandb.Artifact(
        name=f"real_3dhst_eval_{cfg['exp_name']}",
        type="real-data-evaluation",
        description=(
            f"exp_032 evaluated on real 3D-HST AEGIS grism data "
            f"(grism_specPT_v5.pkl, {metrics['n_total']} spectra "
            f"after SNR>={cfg['min_snr']} filter, "
            f"{metrics['n_in_dist']} with z>=0.2). "
            f"NO wavelength interpolation, NO NaN injection. "
            f"Overall NMAD={overall['nmad']:.5f}, "
            f"in-dist NMAD={in_dist['nmad']:.5f}. "
            f"Wavelength grid mismatch: training 10311-17465A "
            f"(0.92 A/pix) vs real 10800-17100A (0.81 A/pix)."
        ),
    )
    out = Path(cfg["output_dir"])
    artifact.add_file(str(out / f"{cfg['exp_name']}_real_eval.csv"))
    artifact.add_file(str(out / f"{cfg['exp_name']}_real_metrics.json"))
    for name in figures:
        artifact.add_file(str(out / f"{cfg['exp_name']}_{name}.png"))
    run.log_artifact(artifact)

    run.finish()
    print(f"   W&B done. Artifact 'real_3dhst_eval_{cfg['exp_name']}' uploaded.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run on first 100 spectra for quick validation")
    args = parser.parse_args()

    print("=" * 60)
    print("  exp_032 EVAL ON REAL 3D-HST AEGIS G141 GRISM DATA")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = _load_real_data(CONFIG)
    if args.smoke:
        df = df.head(100).copy()
        print(f"  [SMOKE MODE] Using first 100 spectra only")

    model = _build_model(CONFIG, device)
    preds_df, dataset = _run_inference(model, df, CONFIG, device)
    metrics = _compute_metrics(preds_df, CONFIG)
    figures = _make_plots(preds_df, metrics, df, dataset, model, device)
    _save_outputs(preds_df, metrics, figures, CONFIG)
    _log_wandb(preds_df, metrics, figures, CONFIG)

    print("=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
