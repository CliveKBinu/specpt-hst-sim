#!/usr/bin/env python3
"""Evaluate a Universal Spectral Encoder (USE) frozen latent.

Runs on held-out real 3D-HST test spectra and reports:
- Reconstruction quality (valid-pixel MSE, cosine) vs frozen AE baseline
- Latent stability under masking / noise (cosine to clean-view latent)
- Latent drift vs frozen autoencoder teacher
- Linear-probe redshift head: NMAD, eta, RMSE, R^2
- SNR-bucketed NMAD (2.5-5, 5-10, 10-20, 20+)

Usage:
  python scripts/eval_universal_latent.py --exp_name use_stageC
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
from src.specpt.model import SpecPT, SpectrumNormalizer, Swish
from src.specpt.losses import NMADLoss
from src.specpt.self_supervised import (
    SpectralViews, valid_pixel_mse, cosine_similarity, build_use_model,
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


def extract_latents(model, loader, device):
    model.student.eval()
    all_h = []
    with torch.no_grad():
        for X in loader:
            X = X.to(device)
            all_h.append(model.student_latent(X).cpu().numpy())
    return np.concatenate(all_h).astype(np.float32)


def compute_metrics(pv, tv):
    pv = np.asarray(pv).ravel()
    tv = np.asarray(tv).ravel()
    delz = (pv - tv) / (1 + tv)
    nmad = float(1.4826 * np.median(np.abs(delz - np.median(delz))))
    eta = float(100 * np.mean(np.abs(delz) > 0.15))
    rmse = float(np.sqrt(np.mean((pv - tv) ** 2)))
    return nmad, eta, rmse


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
        nmad, eta, rmse = compute_metrics(pv[mask], tv[mask])
        results[labels[bid - 1]] = {"n": int(mask.sum()), "nmad": nmad,
                                    "eta": eta, "rmse": rmse}
    return results


def train_linear_probe(h_train, z_train, h_val, z_val, h_test, z_test,
                       args, device):
    """Linear(512,1)+Softplus redshift head on frozen latents (exp_035 recipe)."""
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
        nmad, eta, _ = compute_metrics(pv, z_val)
        if ep % 25 == 0 or nmad < best_val_nmad:
            print(f"  head ep {ep:3d}  train={tl:.4f}  val_nmad={nmad:.5f}")
        if nmad < best_val_nmad:
            best_val_nmad = nmad
            best_ep = ep
            patience = 0
        else:
            patience += 1
            if patience >= args.head_patience:
                print(f"  head early stop at ep {ep}")
                break
    head.eval()
    with torch.no_grad():
        pv_test = head(torch.from_numpy(h_test).to(device)).flatten().cpu().numpy()
    from sklearn.metrics import r2_score
    test_nmad, test_eta, test_rmse = compute_metrics(pv_test, z_test)
    test_r2 = float(r2_score(z_test, pv_test))
    return (best_val_nmad, best_ep, test_nmad, test_eta, test_rmse, test_r2,
            pv_test)


def main():
    parser = argparse.ArgumentParser(description="Evaluate USE frozen latent")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--ckpt", default=None,
                        help="USE checkpoint path (default: "
                             "checkpoints/{exp_name}_stage*_best.pth, latest stage found)")
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

    if args.ckpt is None:
        cands = sorted(
            f"checkpoints/{args.exp_name}_stage{s}_best.pth" for s in "ABC")
        existing = [p for p in cands if os.path.exists(p)]
        if not existing:
            print(f"No USE checkpoint found for {args.exp_name}")
            sys.exit(1)
        args.ckpt = existing[-1]
    print(f"USE checkpoint: {args.ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading real data (SNR>=2.5)...")
    df = pd.read_pickle(REAL_DATA_PATH)
    df = df[df["SNR"] >= 2.5].reset_index(drop=True)
    train, val, test = prepare_real(df, args.val_split, args.test_split, args.seed)
    print(f"Train={len(train)}  Val={len(val)}  Test={len(test)}")
    train_snr, val_snr, test_snr = (d["SNR"].values.ravel()
                                    for d in (train, val, test))

    use = build_use_model(args.ae_ckpt, teacher=True, device=device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    use.student.load_state_dict(ckpt["model_state_dict"], strict=True)
    use.student.eval()
    nt = sum(p.numel() for p in use.student.parameters())
    print(f"Loaded USE student ({nt:,} params)")

    tr_ds = HSTDataset(train)
    va_ds = HSTDataset(val)
    te_ds = HSTDataset(test)
    tr_ld = DataLoader(tr_ds, 64, shuffle=False, num_workers=0)
    va_ld = DataLoader(va_ds, 64, shuffle=False, num_workers=0)
    te_ld = DataLoader(te_ds, 64, shuffle=False, num_workers=0)
    z_train = train["z"].values.ravel().astype(np.float32)
    z_val = val["z"].values.ravel().astype(np.float32)
    z_test = test["z"].values.ravel().astype(np.float32)

    valid_mask = torch.tensor(VALID_MASK, device=device)
    views = SpectralViews(mask_chunks=(2, 5), mask_size=(30, 100),
                          noise_sigma=(0.01, 0.05), seed=args.seed)

    print("Extracting USE latents...")
    t0 = time.time()
    h_train = extract_latents(use, tr_ld, device)
    h_val = extract_latents(use, va_ld, device)
    h_test = extract_latents(use, te_ld, device)
    print(f"Latents: train={h_train.shape} val={h_val.shape} test={h_test.shape} "
          f"({time.time()-t0:.0f}s)")

    # ---- reconstruction quality vs frozen AE baseline ----
    print("Measuring reconstruction quality on test...")
    baseline = build_use_model(args.ae_ckpt, teacher=False, device=device)
    baseline.student.eval()
    mse_student, cos_student = [], []
    mse_baseline, cos_baseline = [], []
    with torch.no_grad():
        for X in te_ld:
            X = X.to(device)
            r_s = use.reconstruct(X)
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

    # ---- latent stability + drift on a test subset ----
    print("Measuring latent stability (masked/noisy vs clean)...")
    stab_masked, stab_noisy, drift_cos = [], [], []
    n_stab = 0
    with torch.no_grad():
        for X in te_ld:
            X = X.to(device)
            b = min(32, X.shape[0])
            Xb = X[:b]
            v = views(Xb)
            h_c = use.student_latent(v["clean"])
            h_m = use.student_latent(v["masked"])
            h_n = use.student_latent(v["noisy1"])
            h_t = use.teacher_latent(v["clean"])
            stab_masked.append(cosine_similarity(h_c, h_m))
            stab_noisy.append(cosine_similarity(h_c, h_n))
            drift_cos.append(cosine_similarity(h_c, h_t))
            n_stab += 1
            if n_stab >= 8:
                break
    print(f"  Latent cos (clean vs masked): {np.mean(stab_masked):.4f}")
    print(f"  Latent cos (clean vs noisy):  {np.mean(stab_noisy):.4f}")
    print(f"  Latent cos (student vs teacher): {np.mean(drift_cos):.4f}")

    # ---- linear-probe redshift head ----
    print("Training linear-probe redshift head on frozen USE latents...")
    (best_val_nmad, best_ep, test_nmad, test_eta, test_rmse, test_r2,
     pv_test) = train_linear_probe(h_train, z_train, h_val, z_val,
                                   h_test, z_test, args, device)
    print(f"\nHead best val NMAD={best_val_nmad:.5f} at ep {best_ep}")
    print(f"Test: NMAD={test_nmad:.5f}  eta={test_eta:.2f}%  RMSE={test_rmse:.4f}  "
          f"R2={test_r2:.4f}")

    buckets = snr_bucket_metrics(pv_test, z_test, test_snr)
    for label, br in buckets.items():
        print(f"  SNR {label}: N={br['n']} NMAD={br['nmad']:.5f} eta={br['eta']:.2f}%")

    # ---- baselines from frozen AE (same head recipe) ----
    print("Training linear-probe head on frozen ORIGINAL AE latents (baseline)...")
    h_base_train = extract_latents(baseline, tr_ld, device)
    h_base_val = extract_latents(baseline, va_ld, device)
    h_base_test = extract_latents(baseline, te_ld, device)
    (b_val_nmad, b_ep, b_nmad, b_eta, b_rmse, b_r2, _) = train_linear_probe(
        h_base_train, z_train, h_base_val, z_val, h_base_test, z_test, args, device)
    print(f"AE baseline: Test NMAD={b_nmad:.5f}  eta={b_eta:.2f}%  "
          f"RMSE={b_rmse:.4f}  R2={b_r2:.4f}")

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
        "head/best_val_nmad": best_val_nmad,
        "head/best_epoch": best_ep,
        "test/nmad": test_nmad,
        "test/eta": test_eta,
        "test/rmse": test_rmse,
        "test/r2": test_r2,
        "ae_baseline/nmad": b_nmad,
        "ae_baseline/eta": b_eta,
        "ae_baseline/rmse": b_rmse,
        "ae_baseline/r2": b_r2,
        "n_test": len(test),
    })
    for label, br in buckets.items():
        wandb.log({f"test_snr_{label}_n": br["n"],
                   f"test_snr_{label}_nmad": br["nmad"],
                   f"test_snr_{label}_eta": br["eta"]})
    wandb.finish()
    print("DONE")


if __name__ == "__main__":
    main()
