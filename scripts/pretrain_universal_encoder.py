#!/usr/bin/env python3
"""Universal Spectral Encoder (USE): self-supervised pretraining.

Label-free pretraining on real 3D-HST spectra. Preserves broad spectral
information via a latent-distillation anchor to the frozen autoencoder and
learns robustness via masked / noise-corrupted views + two-view consistency.

Stages:
  A: masked reconstruction + distillation anchor
  B: + noise-corrupted reconstruction
  C: + two-view latent consistency

Reference: docs/universal_spectral_encoder.md
"""
import sys, os, argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, ".")
from src.specpt.model import SpectrumNormalizer
from src.specpt.self_supervised import (
    UniversalEncoderModel,
    SpectralViews,
    valid_pixel_mse,
    latent_consistency,
    latent_distill,
    cosine_similarity,
    build_use_model,
)

sys.path.insert(0, "/home/ckb2084/research/SpecPT")
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "augment", "/home/ckb2084/research/SpecPT/specpt/augment.py")
_aug = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_aug)
split_by_grism_id = _aug.split_by_grism_id

TRAIN_WAVES = np.linspace(10800.0, 17100.0, 7781)
PAD = (TRAIN_WAVES < 11000) | (TRAIN_WAVES > 16500)
VALID_MASK = ~PAD

REGRID_AE_CKPT = "/home/ckb2084/research/specpt-hst-sim/checkpoints/autoencoder_regrid_autoencoder_best.pth"
REAL_DATA_PATH = "/home/ckb2084/research/SpecPT/data/grism_specPT_v5.pkl"


def _spec_from_clean(row):
    safe = np.where(row["sensitivity_resampled"] == 0, 1e-8,
                    row["sensitivity_resampled"])
    s = (row["clean_flux_resampled"] / safe).astype(np.float32)
    s[PAD] = np.nan
    return s


def prepare_real(df, val_split, test_split, seed):
    specs = [_spec_from_clean(r) for _, r in df.iterrows()]
    df = df.copy()
    df["spec"] = specs
    df["TARGETID"] = df["grism_id"]
    train, val, test = split_by_grism_id(
        df, test_size=val_split + test_split,
        val_size=test_split / (val_split + test_split), random_state=seed,
    )
    cols = ["TARGETID", "z", "spec", "SNR"]
    return train[cols].copy(), val[cols].copy(), test[cols].copy()


class SelfSupervisedRealDataset(Dataset):
    """Returns z-score-normalized clean spectra (no labels)."""

    def __init__(self, df):
        specs = [SpectrumNormalizer.zscore_normalize(s) for s in df["spec"].values]
        self.specs = torch.from_numpy(np.stack(specs, axis=0).astype(np.float32))

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        return self.specs[idx]


def stage_defaults(stage):
    if stage == "A":
        return dict(lmd_mask=1.0, lmd_noise=0.0, lmd_consistency=0.0, lmd_distill=1.0)
    if stage == "B":
        return dict(lmd_mask=1.0, lmd_noise=0.5, lmd_consistency=0.0, lmd_distill=1.0)
    if stage == "C":
        return dict(lmd_mask=1.0, lmd_noise=0.5, lmd_consistency=1.0, lmd_distill=1.0)
    raise ValueError(f"Unknown stage {stage}")


def main():
    parser = argparse.ArgumentParser(description="USE self-supervised pretraining")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--stage", choices=["A", "B", "C"], default="A")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--ae_ckpt", default=REGRID_AE_CKPT)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--lambda_mask", type=float, default=None)
    parser.add_argument("--lambda_noise", type=float, default=None)
    parser.add_argument("--lambda_consistency", type=float, default=None)
    parser.add_argument("--lambda_distill", type=float, default=None)
    parser.add_argument("--mask_chunks", type=int, nargs=2, default=(2, 5))
    parser.add_argument("--mask_size", type=int, nargs=2, default=(30, 100))
    parser.add_argument("--noise_sigma", type=float, nargs=2, default=(0.01, 0.05))
    parser.add_argument("--stop_on_drift", action="store_true",
                        help="Early-stop if clean recon MSE exceeds drift threshold")
    parser.add_argument("--drift_threshold", type=float, default=2.0)
    args = parser.parse_args()

    defaults = stage_defaults(args.stage)
    lmd = {
        "mask": args.lambda_mask if args.lambda_mask is not None else defaults["lmd_mask"],
        "noise": args.lambda_noise if args.lambda_noise is not None else defaults["lmd_noise"],
        "consistency": args.lambda_consistency if args.lambda_consistency is not None else defaults["lmd_consistency"],
        "distill": args.lambda_distill if args.lambda_distill is not None else defaults["lmd_distill"],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Stage: {args.stage}  lambda={lmd}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading real data (SNR>=2.5)...")
    df = pd.read_pickle(REAL_DATA_PATH)
    df = df[df["SNR"] >= 2.5].reset_index(drop=True)
    train, val, test = prepare_real(df, args.val_split, args.test_split, args.seed)
    print(f"Train={len(train)}  Val={len(val)}  Test={len(test)} (test held out)")

    train_ds = SelfSupervisedRealDataset(train)
    val_ds = SelfSupervisedRealDataset(val)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    print("Building USE model (student init from regridded AE, frozen teacher)...")
    model = build_use_model(args.ae_ckpt, teacher=True, device=device)
    views = SpectralViews(mask_chunks=tuple(args.mask_chunks),
                          mask_size=tuple(args.mask_size),
                          noise_sigma=tuple(args.noise_sigma),
                          seed=args.seed)
    valid_mask = torch.tensor(VALID_MASK, device=device)

    student_params = [p for p in model.student.parameters() if p.requires_grad]
    nt = sum(p.numel() for p in student_params)
    print(f"Trainable (student): {nt:,}")
    opt = torch.optim.AdamW(student_params, lr=args.lr, weight_decay=args.weight_decay)

    # Fixed reference batch for drift / stability monitoring (val spectra).
    ref_batch = next(iter(val_ld))[:16].to(device)
    model.student.eval()
    with torch.no_grad():
        ref_recon = model.reconstruct(ref_batch)
        initial_recon_mse = valid_pixel_mse(ref_recon, ref_batch, valid_mask).item()
    print(f"Initial clean recon MSE (16 val samples): {initial_recon_mse:.6f}")

    run_name = f"{args.exp_name}_stage{args.stage}"
    wandb.init(project="specpt-hst-sim-z",
               entity="ckb2084-rochester-institute-of-technology",
               name=run_name,
               config=vars(args))

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.student.load_state_dict(ckpt["model_state_dict"], strict=True)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    def compute_loss(clean):
        """Compute the full USE loss on a batch of clean spectra.

        Returns (total, components dict) where components values are already
        weighted by their lambda.
        """
        batch = views(clean)
        h_clean = model.student_latent(batch["clean"])
        h_masked = model.student_latent(batch["masked"])
        h_noisy1 = model.student_latent(batch["noisy1"])
        h_noisy2 = model.student_latent(batch["noisy2"])
        with torch.no_grad():
            h_teacher = model.teacher_latent(batch["clean"])

        recon_masked = model.reconstruct(batch["masked"])
        recon_noisy = model.reconstruct(batch["noisy1"])

        comp = {}
        if lmd["mask"] > 0:
            comp["recon_mask"] = lmd["mask"] * valid_pixel_mse(
                recon_masked, batch["clean"], valid_mask)
        if lmd["noise"] > 0:
            comp["recon_noise"] = lmd["noise"] * valid_pixel_mse(
                recon_noisy, batch["clean"], valid_mask)
        if lmd["consistency"] > 0:
            comp["consistency"] = lmd["consistency"] * latent_consistency(
                h_noisy1, h_noisy2)
        if lmd["distill"] > 0:
            comp["distill"] = lmd["distill"] * (
                latent_distill(h_clean, h_teacher)
                + latent_distill(h_masked, h_teacher)
                + latent_distill(h_noisy1, h_teacher)
            )
        total = sum(comp.values())
        return total, comp

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.student.train()
        tl = 0.0
        n_batches = 0
        comp_acc = {k: 0.0 for k in ("recon_mask", "recon_noise", "consistency", "distill")}
        for X in train_ld:
            X = X.to(device)
            opt.zero_grad()
            loss, comp = compute_loss(X)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss at epoch {epoch+1}")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_params, max_norm=1.0)
            opt.step()
            tl += loss.item()
            n_batches += 1
            for k in comp_acc:
                comp_acc[k] += comp.get(k, torch.zeros((), device=device)).item()

        if n_batches == 0:
            print("No valid batches - stopping.")
            break
        avg_train = tl / n_batches
        train_comp = {k: v / n_batches for k, v in comp_acc.items()}

        # ---- validation ----
        model.student.eval()
        vl = 0.0
        n_val = 0
        val_comp = {k: 0.0 for k in ("recon_mask", "recon_noise", "consistency", "distill")}
        with torch.no_grad():
            for X in val_ld:
                X = X.to(device)
                loss, comp = compute_loss(X)
                if torch.isnan(loss):
                    continue
                vl += loss.item()
                n_val += 1
                for k in val_comp:
                    val_comp[k] += comp.get(k, torch.zeros((), device=device)).item()
        avg_val = vl / max(n_val, 1)
        val_comp = {k: v / max(n_val, 1) for k, v in val_comp.items()}

        # ---- drift / stability monitors on the fixed reference batch ----
        drift = 1.0
        latent_cos = 1.0
        with torch.no_grad():
            recon = model.reconstruct(ref_batch)
            cur_mse = valid_pixel_mse(recon, ref_batch, valid_mask).item()
            drift = cur_mse / max(initial_recon_mse, 1e-10)
            h_s = model.student_latent(ref_batch)
            h_t = model.teacher_latent(ref_batch)
            latent_cos = cosine_similarity(h_s, h_t)

        wandb.log({
            "epoch": epoch,
            "train/total": avg_train,
            "val/total": avg_val,
            "val/recon_mask": val_comp["recon_mask"],
            "val/recon_noise": val_comp["recon_noise"],
            "val/consistency": val_comp["consistency"],
            "val/distill": val_comp["distill"],
            "monitor/recon_mse_clean": cur_mse,
            "monitor/recon_mse_drift": drift,
            "monitor/latent_cos_to_teacher": latent_cos,
            "lr": opt.param_groups[0]["lr"],
            "time_sec": time.time() - t_start,
        })
        print(f"Epoch {epoch+1:3d}/{args.epochs}  train={avg_train:.4f}  "
              f"val={avg_val:.4f}  drift={drift:.3f}  latent_cos={latent_cos:.4f}  "
              f"{time.time()-t_start:.0f}s")

        # ---- drift gate ----
        if args.stop_on_drift and drift > args.drift_threshold:
            print(f"STOP: recon MSE drift {drift:.2f} > {args.drift_threshold} - "
                  f"autoencoder identity degraded. Stopping.")
            break

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/{args.exp_name}_stage{args.stage}_best.pth"
            torch.save({
                "args": vars(args),
                "epoch": epoch,
                "model_state_dict": model.student.state_dict(),
                "best_val_loss": best_val_loss,
                "recon_mse_drift": drift,
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stop at epoch {epoch+1} (best {best_val_loss:.4f})")
                break

    wandb.summary["best/val_loss"] = best_val_loss
    wandb.summary["initial/recon_mse"] = initial_recon_mse
    wandb.summary["final/recon_mse_drift"] = drift
    wandb.summary["final/latent_cos_to_teacher"] = latent_cos
    wandb.finish()
    print("DONE")


if __name__ == "__main__":
    main()
