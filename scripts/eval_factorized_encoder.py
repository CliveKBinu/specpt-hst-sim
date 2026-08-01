#!/usr/bin/env python3
"""Evaluate a Factorized Universal Encoder.

Reports, on held-out real 3D-HST test spectra (SNR>=2.5):
- Shared-latent reconstruction quality vs frozen AE baseline
- Shared-latent stability (masked/noisy vs clean) + teacher drift
- Redshift from three pathways:
    (a) frozen z_head direct predictions (z_branch -> z_head)
    (b) linear probe on h_z [128]
    (c) linear probe on h_universal [512]
- Anti-shrinkage diagnostics: R^2, prediction range, pred/target std ratio
- Bootstrap 95% CI on test NMAD for the z_head pathway
- SNR-bucketed NMAD / eta

Usage:
  python scripts/eval_factorized_encoder.py --exp_name fac_stage2 --ckpt checkpoints/fac_stage2_stage2_best.pth
"""
import sys, os, argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score

sys.path.insert(0, ".")
from src.specpt.model import SpectrumNormalizer
from src.specpt.losses import NMADLoss
from src.specpt.self_supervised import (
    SpectralViews, valid_pixel_mse, cosine_similarity, build_use_model,
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


class HSTDataset(Dataset):
    def __init__(self, df, with_z=True):
        specs = [SpectrumNormalizer.zscore_normalize(s) for s in df["spec"].values]
        self.X = torch.from_numpy(np.stack(specs, axis=0).astype(np.float32))
        self.with_z = with_z
        if with_z:
            self.z = torch.from_numpy(df["z"].values.astype(np.float32))
        self.snr = df["SNR"].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.with_z:
            return self.X[idx], self.z[idx]
        return self.X[idx]


def extract_shared(model, loader, device):
    model.student.eval()
    all_h = []
    with torch.no_grad():
        for X in loader:
            X = X.to(device)
            all_h.append(model.shared_latent(X).cpu().numpy())
    return np.concatenate(all_h).astype(np.float32)


def extract_z(model, loader, device):
    model.z_branch.eval()
    all_h = []
    with torch.no_grad():
        for X in loader:
            X = X.to(device)
            all_h.append(model.z_latent(X).cpu().numpy())
    return np.concatenate(all_h).astype(np.float32)


def compute_metrics(pv, tv):
    pv = np.asarray(pv).ravel()
    tv = np.asarray(tv).ravel()
    delz = (pv - tv) / (1 + tv)
    nmad = float(1.4826 * np.median(np.abs(delz - np.median(delz))))
    eta = float(100 * np.mean(np.abs(delz) > 0.15))
    rmse = float(np.sqrt(np.mean((pv - tv) ** 2)))
    r2 = float(r2_score(tv, pv))
    return nmad, eta, rmse, r2


def range_report(pv, tv):
    pv = np.asarray(pv).ravel()
    tv = np.asarray(tv).ravel()
    return {
        "pred_min": float(np.min(pv)),
        "pred_max": float(np.max(pv)),
        "true_min": float(np.min(tv)),
        "true_max": float(np.max(tv)),
        "pred_std": float(np.std(pv)),
        "true_std": float(np.std(tv)),
        "std_ratio": float(np.std(pv) / max(np.std(tv), 1e-12)),
    }


def snr_bucket_metrics(pv, tv, snr_values):
    bins = [0, 5, 10, 20, np.inf]
    labels = ["2.5-5", "5-10", "10-20", "20+"]
    pv, tv = np.asarray(pv).ravel(), np.asarray(tv).ravel()
    snr = np.asarray(snr_values).ravel()
    bucket_ids = np.digitize(snr, bins)
    results = {}
    for bid in range(1, len(bins)):
        mask = bucket_ids == bid
        if mask.sum() == 0:
            continue
        nmad, eta, rmse, r2 = compute_metrics(pv[mask], tv[mask])
        results[labels[bid - 1]] = {"n": int(mask.sum()), "nmad": nmad,
                                    "eta": eta, "rmse": rmse, "r2": r2}
    return results


def bootstrap_nmad_ci(pv, tv, n_boot=200, seed=42):
    pv = np.asarray(pv).ravel()
    tv = np.asarray(tv).ravel()
    rng = np.random.RandomState(seed)
    n = len(pv)
    nmads = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        delz = (pv[idx] - tv[idx]) / (1 + tv[idx])
        nmads.append(1.4826 * np.median(np.abs(delz - np.median(delz))))
    return float(np.percentile(nmads, 2.5)), float(np.percentile(nmads, 97.5))


def train_probe(h_train, z_train, h_val, z_val, h_test, z_test, args, device,
                label=""):
    """Linear(D,1)+Softplus head on frozen features (exp_035 recipe)."""
    head = nn.Sequential(nn.Linear(h_train.shape[1], 1), nn.Softplus()).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.head_lr,
                            weight_decay=args.head_wd)
    criterion = NMADLoss(normalization_factor="std")
    n_train = h_train.shape[0]
    best_val_nmad = 1e9
    best_ep = 0
    patience = 0
    for ep in range(1, args.head_epochs + 1):
        perm = torch.randperm(n_train, device=device)
        head.train()
        tl = 0.0
        for i in range(0, n_train, args.head_batch):
            idx = perm[i:i + args.head_batch]
            X = torch.from_numpy(h_train[idx.cpu().numpy()]).to(device)
            y = torch.from_numpy(z_train[idx.cpu().numpy()]).to(device)
            opt.zero_grad()
            pred = head(X).flatten()
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            tl += loss.item()
        head.eval()
        with torch.no_grad():
            pv = head(torch.from_numpy(h_val).to(device)).flatten().cpu().numpy()
        nmad, _, _, _ = compute_metrics(pv, z_val)
        if ep % 25 == 0 or nmad < best_val_nmad:
            print(f"  {label} head ep {ep:3d}  train={tl:.4f}  val_nmad={nmad:.5f}")
        if nmad < best_val_nmad:
            best_val_nmad = nmad
            best_ep = ep
            patience = 0
        else:
            patience += 1
            if patience >= args.head_patience:
                print(f"  {label} head early stop at ep {ep}")
                break
    head.eval()
    with torch.no_grad():
        pv_test = head(torch.from_numpy(h_test).to(device)).flatten().cpu().numpy()
    test_nmad, test_eta, test_rmse, test_r2 = compute_metrics(pv_test, z_test)
    return (best_val_nmad, best_ep, test_nmad, test_eta, test_rmse, test_r2,
            pv_test)


def main():
    parser = argparse.ArgumentParser(description="Evaluate factorized encoder")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--head_wd", type=float, default=1e-3)
    parser.add_argument("--head_epochs", type=int, default=300)
    parser.add_argument("--head_batch", type=int, default=64)
    parser.add_argument("--head_patience", type=int, default=30)
    parser.add_argument("--ae_ckpt", default=REGRID_AE_CKPT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Factorized checkpoint: {args.ckpt}")

    print("Loading real data (SNR>=2.5)...")
    df = pd.read_pickle(REAL_DATA_PATH)
    df = df[df["SNR"] >= 2.5].reset_index(drop=True)
    train, val, test = prepare_real(df, args.val_split, args.test_split, args.seed)
    print(f"Train={len(train)}  Val={len(val)}  Test={len(test)}")
    test_snr = test["SNR"].values.ravel()

    model = build_factorized_model(args.ae_ckpt, init_ckpt=args.ckpt,
                                   band_mask=BAND_MASK, device=device)
    model.eval()
    print(f"Loaded factorized model ({model.n_total:,} params)")

    tr_ds = HSTDataset(train, with_z=False)
    va_ds = HSTDataset(val, with_z=False)
    te_ds = HSTDataset(test, with_z=False)
    tr_ld = DataLoader(tr_ds, 64, shuffle=False, num_workers=0)
    va_ld = DataLoader(va_ds, 64, shuffle=False, num_workers=0)
    te_ld = DataLoader(te_ds, 64, shuffle=False, num_workers=0)
    z_train = train["z"].values.ravel().astype(np.float32)
    z_val = val["z"].values.ravel().astype(np.float32)
    z_test = test["z"].values.ravel().astype(np.float32)

    valid_mask = torch.tensor(VALID_MASK, device=device)
    views = SpectralViews(mask_chunks=(2, 5), mask_size=(30, 100),
                          noise_sigma=(0.01, 0.05), seed=args.seed)

    print("Extracting shared (h_universal) and z (h_z) latents...")
    t0 = time.time()
    sh_tr = extract_shared(model, tr_ld, device)
    sh_va = extract_shared(model, va_ld, device)
    sh_te = extract_shared(model, te_ld, device)
    z_tr = extract_z(model, tr_ld, device)
    z_va = extract_z(model, va_ld, device)
    z_te = extract_z(model, te_ld, device)
    print(f"Shared: {sh_tr.shape} {sh_va.shape} {sh_te.shape}  "
          f"z: {z_tr.shape} {z_va.shape} {z_te.shape}  ({time.time()-t0:.0f}s)")

    # ---- reconstruction quality vs frozen AE baseline ----
    print("Measuring reconstruction quality on test...")
    baseline = build_use_model(args.ae_ckpt, teacher=False, device=device)
    baseline.student.eval()
    mse_student, cos_student, mse_baseline, cos_baseline = [], [], [], []
    with torch.no_grad():
        for X in te_ld:
            X = X.to(device)
            r_s = model.reconstruct(X)
            r_b = baseline.reconstruct(X)
            mse_student.append(valid_pixel_mse(r_s, X, valid_mask).item())
            mse_baseline.append(valid_pixel_mse(r_b, X, valid_mask).item())
            for i in range(X.shape[0]):
                cos_student.append(F.cosine_similarity(r_s[i], X[i], dim=0).item())
                cos_baseline.append(F.cosine_similarity(r_b[i], X[i], dim=0).item())
    print(f"  Recon MSE   student={np.mean(mse_student):.6f}  "
          f"baseline={np.mean(mse_baseline):.6f}")
    print(f"  Recon Cos   student={np.mean(cos_student):.4f}  "
          f"baseline={np.mean(cos_baseline):.4f}")

    # ---- latent stability + drift ----
    print("Measuring latent stability (masked/noisy vs clean)...")
    stab_masked, stab_noisy, drift_cos = [], [], []
    n_stab = 0
    with torch.no_grad():
        for X in te_ld:
            X = X.to(device)
            b = min(32, X.shape[0])
            v = views(X[:b])
            h_c = model.shared_latent(v["clean"])
            h_m = model.shared_latent(v["masked"])
            h_n = model.shared_latent(v["noisy1"])
            h_t = model.teacher_latent(v["clean"])
            stab_masked.append(cosine_similarity(h_c, h_m))
            stab_noisy.append(cosine_similarity(h_c, h_n))
            drift_cos.append(cosine_similarity(h_c, h_t))
            n_stab += 1
            if n_stab >= 8:
                break
    print(f"  Latent cos (clean vs masked): {np.mean(stab_masked):.4f}")
    print(f"  Latent cos (clean vs noisy):  {np.mean(stab_noisy):.4f}")
    print(f"  Latent cos (student vs teacher): {np.mean(drift_cos):.4f}")

    # ---- redshift pathways ----
    # (a) frozen z_head direct
    print("Running frozen z_head (z_branch -> z_head) on test...")
    pv_direct = []
    model.z_branch.eval()
    model.z_head.eval()
    with torch.no_grad():
        for X in te_ld:
            X = X.to(device)
            pv_direct.append(model.z_pred(X).flatten().cpu().numpy())
    pv_direct = np.clip(np.concatenate(pv_direct), 0, None)
    d_nmad, d_eta, d_rmse, d_r2 = compute_metrics(pv_direct, z_test)
    lo, hi = bootstrap_nmad_ci(pv_direct, z_test)
    d_rng = range_report(pv_direct, z_test)
    print(f"  z_head direct: NMAD={d_nmad:.5f}  eta={d_eta:.2f}%  "
          f"RMSE={d_rmse:.4f}  R2={d_r2:.4f}  CI95=[{lo:.5f},{hi:.5f}]")
    print(f"  range: pred=[{d_rng['pred_min']:.3f},{d_rng['pred_max']:.3f}] "
          f"true=[{d_rng['true_min']:.3f},{d_rng['true_max']:.3f}] "
          f"std_ratio={d_rng['std_ratio']:.3f}")

    # (b) linear probe on h_z
    print("Training linear-probe head on h_z [128]...")
    (z_val_nmad, z_ep, z_nmad, z_eta, z_rmse, z_r2, _) = train_probe(
        z_tr, z_train, z_va, z_val, z_te, z_test, args, device, label="z")
    print(f"  h_z probe: Test NMAD={z_nmad:.5f}  eta={z_eta:.2f}%  "
          f"RMSE={z_rmse:.4f}  R2={z_r2:.4f}")

    # (c) linear probe on h_universal
    print("Training linear-probe head on h_universal [512]...")
    (s_val_nmad, s_ep, s_nmad, s_eta, s_rmse, s_r2, _) = train_probe(
        sh_tr, z_train, sh_va, z_val, sh_te, z_test, args, device, label="shared")
    print(f"  h_universal probe: Test NMAD={s_nmad:.5f}  eta={s_eta:.2f}%  "
          f"RMSE={s_rmse:.4f}  R2={s_r2:.4f}")

    # ---- AE baseline (same recipe) ----
    print("Training linear-probe head on frozen ORIGINAL AE latents (baseline)...")
    h_base_tr = extract_shared(baseline, tr_ld, device)
    h_base_va = extract_shared(baseline, va_ld, device)
    h_base_te = extract_shared(baseline, te_ld, device)
    (b_val_nmad, b_ep, b_nmad, b_eta, b_rmse, b_r2, _) = train_probe(
        h_base_tr, z_train, h_base_va, z_val, h_base_te, z_test, args,
        device, label="ae")
    print(f"AE baseline: Test NMAD={b_nmad:.5f}  eta={b_eta:.2f}%  "
          f"RMSE={b_rmse:.4f}  R2={b_r2:.4f}")

    buckets = snr_bucket_metrics(pv_direct, z_test, test_snr)
    for label, br in buckets.items():
        print(f"  SNR {label}: N={br['n']} NMAD={br['nmad']:.5f} eta={br['eta']:.2f}% "
              f"R2={br['r2']:.4f}")

    wandb.init(project="specpt-hst-sim-z",
               entity="ckb2084-rochester-institute-of-technology",
               name=f"{args.exp_name}_eval", config=vars(args))
    wandb.log({
        "test/recon_mse": np.mean(mse_student),
        "test/recon_cosine": np.mean(cos_student),
        "baseline/recon_mse": np.mean(mse_baseline),
        "baseline/recon_cosine": np.mean(cos_baseline),
        "stability/masked_cos": np.mean(stab_masked),
        "stability/noisy_cos": np.mean(stab_noisy),
        "stability/student_teacher_cos": np.mean(drift_cos),
        "zhead/test_nmad": d_nmad,
        "zhead/test_eta": d_eta,
        "zhead/test_rmse": d_rmse,
        "zhead/test_r2": d_r2,
        "zhead/nmad_ci95_lo": lo,
        "zhead/nmad_ci95_hi": hi,
        "zhead/pred_std_ratio": d_rng["std_ratio"],
        "zhead/pred_range": f"{d_rng['pred_min']:.3f}-{d_rng['pred_max']:.3f}",
        "zprobe/test_nmad": z_nmad,
        "zprobe/test_eta": z_eta,
        "zprobe/test_r2": z_r2,
        "sharedprobe/test_nmad": s_nmad,
        "sharedprobe/test_eta": s_eta,
        "sharedprobe/test_r2": s_r2,
        "ae_baseline/nmad": b_nmad,
        "ae_baseline/eta": b_eta,
        "ae_baseline/rmse": b_rmse,
        "ae_baseline/r2": b_r2,
        "n_test": len(test),
    })
    for label, br in buckets.items():
        wandb.log({f"test_snr_{label}_n": br["n"],
                   f"test_snr_{label}_nmad": br["nmad"],
                   f"test_snr_{label}_eta": br["eta"],
                   f"test_snr_{label}_r2": br["r2"]})
    wandb.finish()
    print("DONE")


if __name__ == "__main__":
    main()
