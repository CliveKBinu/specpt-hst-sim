#!/usr/bin/env python3
"""Factorized Universal Encoder pretraining.

Trains a shared encoder (h_universal, anchored to the frozen AE by
reconstruction + latent distillation + robustness) together with a
task-specific early z branch (h_z) supervised by redshift.

Stages (presets, overridable):
  1: sim-only, shared encoder FROZEN, z branch only  (signal / implementation test)
  2: real-only, shared encoder FROZEN, z branch only  (early-feature diagnostic)
  3: joint sim+real, shared encoder UNFROZEN at low lr   (the actual factorization)

Loss:
  L = lambda_mask  * recon(masked -> clean)
    + lambda_noise * recon(noisy  -> clean)
    + lambda_cons  * consistency(h_noisy1, h_noisy2)
    + lambda_dist  * distill(h_shared, h_teacher)
    + lambda_z     * NMADLoss(z_pred, z_true)          (ramped over z_ramp_epochs)

Reference: docs/factorized_universal_encoder.md
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
from src.specpt.losses import NMADLoss
from src.specpt.self_supervised import (
    SpectralViews, valid_pixel_mse, latent_consistency, latent_distill,
    cosine_similarity,
)
from src.specpt.factorized import (
    build_factorized_model, downsample_valid_mask,
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
BAND_MASK = downsample_valid_mask(VALID_MASK, n_bands=487)

REGRID_AE_CKPT = "/home/ckb2084/research/specpt-hst-sim/checkpoints/autoencoder_regrid_autoencoder_best.pth"
REAL_DATA_PATH = "/home/ckb2084/research/SpecPT/data/grism_specPT_v5.pkl"
SIM_DATA_PATH = "/home/ckb2084/research/specpt-hst-sim/data/training_format/grism_training_sim_v3_regrid.parquet"


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


def prepare_sim(df, val_split, test_split, seed, n_subset=None):
    train, val, _test = split_by_grism_id(
        df, test_size=val_split + test_split,
        val_size=test_split / (val_split + test_split), random_state=seed,
    )
    train = train.rename(columns={"grism_id": "TARGETID"})
    val = val.rename(columns={"grism_id": "TARGETID"})
    if "spec" not in train.columns:
        train = train.rename(columns={"clean_flux_resampled": "spec"})
        val = val.rename(columns={"clean_flux_resampled": "spec"})
    if n_subset and len(train) > n_subset:
        train = train.sample(n=n_subset, random_state=seed).reset_index(drop=True)
    cols = ["TARGETID", "z", "spec", "SNR"]
    print(f"Sim split: train={len(train)} val={len(val)} subset={n_subset}")
    return train[cols].copy(), val[cols].copy()


class ZLabeledDataset(Dataset):
    """Z-score-normalized spectra with z labels (real or sim)."""

    def __init__(self, df):
        specs = [SpectrumNormalizer.zscore_normalize(s) for s in df["spec"].values]
        self.X = torch.from_numpy(np.stack(specs, axis=0).astype(np.float32))
        self.z = torch.from_numpy(df["z"].values.astype(np.float32))
        self.snr = df["SNR"].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.z[idx]


class SelfSupervisedRealDataset(Dataset):
    """Label-free z-score-normalized spectra (for recon anchors on real)."""

    def __init__(self, df):
        specs = [SpectrumNormalizer.zscore_normalize(s) for s in df["spec"].values]
        self.specs = torch.from_numpy(np.stack(specs, axis=0).astype(np.float32))

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        return self.specs[idx]


class JointSimRealLoader:
    """Interleaves labeled real + labeled sim batches."""

    def __init__(self, real_ds, sim_ds, batch_size, real_frac):
        self.real = real_ds
        self.sim = sim_ds
        self.batch_size = batch_size
        self.real_bs = max(1, int(batch_size * real_frac))
        self.sim_bs = batch_size - self.real_bs

    def __len__(self):
        return len(self.real) // self.real_bs

    def __iter__(self):
        real_loader = DataLoader(self.real, batch_size=self.real_bs, shuffle=True,
                                 num_workers=0, drop_last=True)
        sim_loader = DataLoader(self.sim, batch_size=self.sim_bs, shuffle=True,
                                num_workers=0, drop_last=True)
        sim_iter = iter(sim_loader)
        for real_batch in real_loader:
            try:
                sim_batch = next(sim_iter)
            except StopIteration:
                sim_iter = iter(sim_loader)
                sim_batch = next(sim_iter)
            X_r, z_r = real_batch
            X_s, z_s = sim_batch
            yield X_r, z_r, X_s, z_s


def compute_nmad(pv, tv):
    pv = np.asarray(pv).ravel()
    tv = np.asarray(tv).ravel()
    delz = (pv - tv) / (1 + tv)
    nmad = float(1.4826 * np.median(np.abs(delz - np.median(delz))))
    eta = float(100 * np.mean(np.abs(delz) > 0.15))
    return nmad, eta


def stage_defaults(stage):
    if stage == "1":
        return dict(data="sim", freeze_encoder=True, lmd_mask=0.0, lmd_noise=0.0,
                    lmd_consistency=0.0, lmd_distill=0.0, lmd_z=1.0, z_ramp=0)
    if stage == "2":
        return dict(data="real", freeze_encoder=True, lmd_mask=0.0, lmd_noise=0.0,
                    lmd_consistency=0.0, lmd_distill=0.0, lmd_z=1.0, z_ramp=0)
    if stage == "3":
        return dict(data="joint", freeze_encoder=False, lmd_mask=1.0, lmd_noise=0.5,
                    lmd_consistency=1.0, lmd_distill=1.0, lmd_z=1.0, z_ramp=10)
    raise ValueError(f"Unknown stage {stage}")


def main():
    parser = argparse.ArgumentParser(description="Factorized encoder pretraining")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--stage", choices=["1", "2", "3"], default="1")
    parser.add_argument("--data", choices=["sim", "real", "joint"], default=None)
    parser.add_argument("--freeze_encoder", choices=["yes", "no"], default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--z_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--ae_ckpt", default=REGRID_AE_CKPT)
    parser.add_argument("--init_ckpt", default=None,
                        help="init student weights from a USE/AE checkpoint")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--lambda_mask", type=float, default=None)
    parser.add_argument("--lambda_noise", type=float, default=None)
    parser.add_argument("--lambda_consistency", type=float, default=None)
    parser.add_argument("--lambda_distill", type=float, default=None)
    parser.add_argument("--lambda_z", type=float, default=None)
    parser.add_argument("--z_ramp_epochs", type=int, default=None)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--sim_subset_size", type=int, default=None)
    parser.add_argument("--real_frac", type=float, default=0.25)
    parser.add_argument("--mask_chunks", type=int, nargs=2, default=(2, 5))
    parser.add_argument("--mask_size", type=int, nargs=2, default=(30, 100))
    parser.add_argument("--noise_sigma", type=float, nargs=2, default=(0.01, 0.05))
    parser.add_argument("--stop_on_drift", action="store_true")
    parser.add_argument("--drift_threshold", type=float, default=2.0)
    args = parser.parse_args()

    d = stage_defaults(args.stage)
    data = args.data if args.data else d["data"]
    freeze = args.freeze_encoder == "yes" if args.freeze_encoder else d["freeze_encoder"]
    lmd = {
        "mask": args.lambda_mask if args.lambda_mask is not None else d["lmd_mask"],
        "noise": args.lambda_noise if args.lambda_noise is not None else d["lmd_noise"],
        "consistency": args.lambda_consistency if args.lambda_consistency is not None else d["lmd_consistency"],
        "distill": args.lambda_distill if args.lambda_distill is not None else d["lmd_distill"],
        "z": args.lambda_z if args.lambda_z is not None else d["lmd_z"],
    }
    z_ramp = args.z_ramp_epochs if args.z_ramp_epochs is not None else d["z_ramp"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Stage: {args.stage}  data={data}  "
          f"freeze_encoder={freeze}  lambda={lmd}  z_ramp={z_ramp}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- data ----
    sim_train, sim_val = None, None
    real_train, real_val = None, None
    if data in ("real", "joint"):
        print("Loading real data (SNR>=2.5)...")
        df_real = pd.read_pickle(REAL_DATA_PATH)
        df_real = df_real[df_real["SNR"] >= 2.5].reset_index(drop=True)
        real_train, real_val, _ = prepare_real(
            df_real, args.val_split, args.test_split, args.seed)
        print(f"Real: train={len(real_train)} val={len(real_val)}")
    if data in ("sim", "joint"):
        print("Loading sim data...")
        df_sim = pd.read_parquet(SIM_DATA_PATH)
        sim_train, sim_val = prepare_sim(
            df_sim, args.val_split, args.test_split, args.seed,
            args.sim_subset_size)
        print(f"Sim: train={len(sim_train)} val={len(sim_val)}")

    # ---- model ----
    print("Building factorized model (student init from AE, frozen teacher)...")
    model = build_factorized_model(args.ae_ckpt, init_ckpt=args.init_ckpt,
                                   z_dim=args.z_dim, band_mask=BAND_MASK,
                                   device=device)
    if freeze:
        model.set_shared_trainable(False)
        print("Shared encoder FROZEN (only z branch + z head train)")
    else:
        model.set_shared_trainable(True)
        print("Shared encoder TRAINABLE (with recon + distill anchor)")

    groups, wd = model.trainable_param_groups(args.encoder_lr, args.z_lr,
                                              args.weight_decay)
    if not groups:
        print("No trainable parameters - nothing to train.")
        sys.exit(1)
    opt = torch.optim.AdamW(groups, weight_decay=wd)
    print(f"Trainable: {model.n_trainable:,}/{model.n_total:,}  "
          f"(z_lr={args.z_lr}, encoder_lr={args.encoder_lr})")

    views = SpectralViews(mask_chunks=tuple(args.mask_chunks),
                          mask_size=tuple(args.mask_size),
                          noise_sigma=tuple(args.noise_sigma), seed=args.seed)
    valid_mask = torch.tensor(VALID_MASK, device=device)
    criterion = NMADLoss(normalization_factor="std")

    # ---- train loaders ----
    train_loader = None
    sim_val_loader = None
    if data == "sim":
        train_loader = DataLoader(ZLabeledDataset(sim_train), batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, drop_last=True)
        sim_val_loader = DataLoader(ZLabeledDataset(sim_val), batch_size=args.batch_size,
                                    shuffle=False, num_workers=0)
    elif data == "real":
        train_loader = DataLoader(ZLabeledDataset(real_train), batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, drop_last=True)
    elif data == "joint":
        train_loader = JointSimRealLoader(ZLabeledDataset(real_train),
                                          ZLabeledDataset(sim_train),
                                          args.batch_size, args.real_frac)

    real_val_loader = None
    if real_val is not None:
        real_val_loader = DataLoader(ZLabeledDataset(real_val), batch_size=args.batch_size,
                                     shuffle=False, num_workers=0)

    # ---- fixed reference batch for drift / stability (real val spectra) ----
    ref_loader = None
    if real_val is not None:
        ref_loader = DataLoader(SelfSupervisedRealDataset(real_val),
                                batch_size=args.batch_size, shuffle=False, num_workers=0)
        ref_batch = next(iter(ref_loader))[:16].to(device)
    else:
        ref_batch = next(iter(train_loader))[0][:16].to(device)

    model.eval()
    with torch.no_grad():
        ref_recon = model.reconstruct(ref_batch)
        initial_recon_mse = valid_pixel_mse(ref_recon, ref_batch, valid_mask).item()
    print(f"Initial clean recon MSE (16 ref samples): {initial_recon_mse:.6f}")

    run_name = f"{args.exp_name}_stage{args.stage}"
    wandb.init(project="specpt-hst-sim-z",
               entity="ckb2084-rochester-institute-of-technology",
               name=run_name, config=vars(args))

    start_epoch = 0
    best_val_nmad = 1e9
    best_ep = 0
    patience_counter = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.student.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "z_branch_state_dict" in ckpt:
            model.z_branch.load_state_dict(ckpt["z_branch_state_dict"], strict=True)
        if "z_head_state_dict" in ckpt:
            model.z_head.load_state_dict(ckpt["z_head_state_dict"], strict=True)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_nmad = 1e9
        print(f"Resumed weights from {args.resume} (epoch {start_epoch}); "
              f"early-stopping baseline reset")

    def train_mode():
        model.student.eval() if freeze else model.student.train()
        model.z_branch.train()
        model.z_head.train()

    def eval_mode():
        model.student.eval()
        model.z_branch.eval()
        model.z_head.eval()

    def labeled_loss(clean, z_true):
        """Returns (total, comp dict) for labeled batches."""
        eff_z = lmd["z"] * min(1.0, (epoch + 1) / max(z_ramp, 1)) if z_ramp > 0 else lmd["z"]
        comp = {}
        if eff_z > 0:
            pred = model.z_pred(clean).flatten()
            comp["z"] = eff_z * criterion(pred, z_true)
        anchors = (lmd["mask"] + lmd["noise"] + lmd["consistency"] + lmd["distill"]) > 0
        if anchors and not freeze:
            batch = views(clean)
            h_clean = model.shared_latent(batch["clean"])
            h_noisy1 = model.shared_latent(batch["noisy1"])
            h_noisy2 = model.shared_latent(batch["noisy2"])
            with torch.no_grad():
                h_teacher = model.teacher_latent(batch["clean"])
            recon_masked = model.reconstruct(batch["masked"])
            recon_noisy = model.reconstruct(batch["noisy1"])
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
                comp["distill"] = lmd["distill"] * latent_distill(
                    h_clean, h_teacher)
        total = sum(comp.values())
        return total, comp, eff_z

    def eval_nmad(loader, device):
        eval_mode()
        pv, tv = [], []
        with torch.no_grad():
            for X, z in loader:
                X = X.to(device)
                pv.append(model.z_pred(X).flatten().cpu().numpy())
                tv.append(z.numpy())
        pv = np.clip(np.concatenate(pv), 0, None)
        tv = np.concatenate(tv)
        return compute_nmad(pv, tv)

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_mode()
        tl = 0.0
        n_batches = 0
        comp_acc = {k: 0.0 for k in ("z", "recon_mask", "recon_noise",
                                     "consistency", "distill")}
        for batch in train_loader:
            if data == "joint":
                X_r, z_r, X_s, z_s = batch
                clean = torch.cat([X_r, X_s]).to(device)
                z_true = torch.cat([z_r, z_s]).flatten().to(device)
            else:
                clean, z_true = batch
                clean = clean.to(device)
                z_true = z_true.flatten().to(device)
            opt.zero_grad()
            loss, comp, _ = labeled_loss(clean, z_true)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss at epoch {epoch+1}")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in groups for p in g["params"]], max_norm=1.0)
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
        val_nmad, val_eta = 1e9, 100.0
        sim_val_nmad, sim_val_eta = None, None
        if real_val_loader is not None:
            val_nmad, val_eta = eval_nmad(real_val_loader, device)
        if sim_val_loader is not None:
            sim_val_nmad, sim_val_eta = eval_nmad(sim_val_loader, device)
        else:
            sim_val_nmad, sim_val_eta = val_nmad, val_eta

        # ---- drift / stability monitors ----
        drift = 1.0
        latent_cos = 1.0
        cur_mse = initial_recon_mse
        eval_mode()
        with torch.no_grad():
            recon = model.reconstruct(ref_batch)
            cur_mse = valid_pixel_mse(recon, ref_batch, valid_mask).item()
            drift = cur_mse / max(initial_recon_mse, 1e-10)
            h_s = model.shared_latent(ref_batch)
            h_t = model.teacher_latent(ref_batch)
            latent_cos = cosine_similarity(h_s, h_t)

        log = {
            "epoch": epoch,
            "train/total": avg_train,
            "train/z": train_comp["z"],
            "train/recon_mask": train_comp["recon_mask"],
            "train/recon_noise": train_comp["recon_noise"],
            "train/consistency": train_comp["consistency"],
            "train/distill": train_comp["distill"],
            "val/real_nmad": val_nmad,
            "val/real_eta": val_eta,
            "monitor/recon_mse_clean": cur_mse,
            "monitor/recon_mse_drift": drift,
            "monitor/latent_cos_to_teacher": latent_cos,
            "lr": opt.param_groups[0]["lr"],
            "time_sec": time.time() - t_start,
        }
        if sim_val_nmad is not None:
            log["val/sim_nmad"] = sim_val_nmad
            log["val/sim_eta"] = sim_val_eta
        wandb.log(log)
        print(f"Epoch {epoch+1:3d}/{args.epochs}  train={avg_train:.4f}  "
              f"real_nmad={val_nmad:.5f}  "
              + (f"sim_nmad={sim_val_nmad:.5f}  " if sim_val_nmad is not None else "")
              + f"drift={drift:.3f}  latent_cos={latent_cos:.4f}  "
              + f"{time.time()-t_start:.0f}s")

        # ---- drift gate ----
        if args.stop_on_drift and drift > args.drift_threshold:
            print(f"STOP: recon MSE drift {drift:.2f} > {args.drift_threshold} - "
                  f"autoencoder identity degraded. Stopping.")
            break

        primary = sim_val_nmad if real_val_loader is None else val_nmad
        if primary < best_val_nmad:
            best_val_nmad = primary
            best_ep = epoch
            patience_counter = 0
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/{args.exp_name}_stage{args.stage}_best.pth"
            torch.save({
                "args": vars(args),
                "epoch": epoch,
                "model_state_dict": model.student.state_dict(),
                "z_branch_state_dict": model.z_branch.state_dict(),
                "z_head_state_dict": model.z_head.state_dict(),
                "best_val_nmad": best_val_nmad,
                "recon_mse_drift": drift,
                "latent_cos_to_teacher": latent_cos,
            }, ckpt_path)
            print(f"  Saved {ckpt_path}  (best_nmad={best_val_nmad:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stop at epoch {epoch+1} (best nmad {best_val_nmad:.5f})")
                break

    wandb.summary["best/val_nmad"] = best_val_nmad
    wandb.summary["best/epoch"] = best_ep
    wandb.summary["final/recon_mse_drift"] = drift
    wandb.summary["final/latent_cos_to_teacher"] = latent_cos
    wandb.finish()
    print("DONE")


if __name__ == "__main__":
    main()
