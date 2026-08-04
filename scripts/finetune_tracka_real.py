#!/usr/bin/env python3
"""Transfer learning: head-only fine-tuning of sim-trained Track A models on real 3D-HST.

Usage:
    python scripts/finetune_tracka_real.py \
        --config configs/tracka_control_z.yaml \
        --checkpoint checkpoints/tracka_control_z_best_model.pth \
        --exp-name tracka_control_z_transfer
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
import torch.nn as nn
import yaml

REPO_ROOT = Path("/home/ckb2084/research/specpt-hst-sim")
sys.path.insert(0, str(REPO_ROOT))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "augment", "/home/ckb2084/research/SpecPT/specpt/augment.py"
)
_aug = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_aug)
split_by_grism_id = _aug.split_by_grism_id

from torch.utils.data import DataLoader
from src.specpt.model import (
    SpecPT,
    EnhancedSpecPTForRedshift,
    SpectrumNormalizer,
)
from src.specpt.dataloader import HSTGrismDataset

TRAIN_WAVES = np.linspace(10800.0, 17100.0, 7781)
PAD = (TRAIN_WAVES < 11000) | (TRAIN_WAVES > 16500)
REAL_DATA_PATH = "/home/ckb2084/research/SpecPT/data/grism_specPT_v5.pkl"


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_real_data(val_split, test_split, seed):
    df = pd.read_pickle(REAL_DATA_PATH)
    df = df[df["SNR"] >= 2.5].reset_index(drop=True)
    print(f"Loaded {len(df)} real 3D-HST spectra (SNR >= 2.5)")

    def _spec_from_clean(row):
        safe = np.where(row["sensitivity_resampled"] == 0, 1e-8, row["sensitivity_resampled"])
        s = (row["clean_flux_resampled"] / safe).astype(np.float32)
        s[PAD] = np.nan
        return s

    df["spec"] = [_spec_from_clean(r) for _, r in df.iterrows()]
    df["TARGETID"] = df["grism_id"]

    train, val, test = split_by_grism_id(
        df,
        test_size=val_split + test_split,
        val_size=test_split / (val_split + test_split),
        random_state=seed,
    )
    cols = ["TARGETID", "z", "spec", "SNR"]
    train = train[cols].reset_index(drop=True)
    val = val[cols].reset_index(drop=True)
    test = test[cols].reset_index(drop=True)
    print(f"Split: train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


def load_model(config, checkpoint_path):
    ae_cfg = config.get("model", {})
    z_cfg = config.get("redshift_head", config.get("head", {}))

    sp = SpecPT(
        input_size=ae_cfg.get("input_size", 7781),
        d_model=ae_cfg.get("d_model", 512),
        nhead=ae_cfg.get("nhead", 8),
        num_encoder_layers=ae_cfg.get("num_encoder_layers", 3),
        num_decoder_layers=ae_cfg.get("num_decoder_layers", 3),
        dim_feedforward=ae_cfg.get("dim_feedforward", 2048),
        dropout=ae_cfg.get("dropout", 0.1),
    )

    model = EnhancedSpecPTForRedshift(
        sp,
        output_features=z_cfg.get("output_features", 1),
        num_mlp_blocks=z_cfg.get("num_mlp_blocks", 5),
        mlp_dim=z_cfg.get("mlp_dim", 512),
        dropout_rate=z_cfg.get("dropout_rate", 0.1),
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys: {missing[:5]}...")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys: {unexpected[:5]}...")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded checkpoint from {checkpoint_path}")
    print(f"  Model: {total_params:,} total params")
    return model


def freeze_backbone(model, unfreeze_encoder_layers=0):
    for p in model.parameters():
        p.requires_grad = False

    for p in model.mlp_blocks.parameters():
        p.requires_grad = True
    for p in model.prediction.parameters():
        p.requires_grad = True
    for p in model.attention.parameters():
        p.requires_grad = True

    unfrozen_enc_names = []
    if unfreeze_encoder_layers > 0:
        enc_layers = model.pretrained_model.transformer_encoder.layers
        n = len(enc_layers)
        for i in range(n - unfreeze_encoder_layers, n):
            for p in enc_layers[i].parameters():
                p.requires_grad = True
            unfrozen_enc_names.append(f"encoder.layers[{i}]")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    head_trainable = sum(p.numel() for p in model.mlp_blocks.parameters() if p.requires_grad)
    head_trainable += sum(p.numel() for p in model.prediction.parameters() if p.requires_grad)
    head_trainable += sum(p.numel() for p in model.attention.parameters() if p.requires_grad)
    enc_trainable = trainable - head_trainable
    print(f"  Frozen AE: {frozen:,} params")
    print(f"  Trainable head: {head_trainable:,} params")
    if unfrozen_enc_names:
        print(f"  Unfrozen encoder: {enc_trainable:,} params in {unfrozen_enc_names}")
    return unfrozen_enc_names


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
    total_loss = 0.0
    n = 0
    for X, Y, _, _ in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        y = model(X).flatten()
        loss = criterion(y, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        n += X.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    preds, trues = [], []
    for X, Y, _, _ in loader:
        X, Y = X.to(device), Y.to(device)
        y = model(X).flatten()
        loss = criterion(y, Y)
        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item() * X.size(0)
            n += 1
        preds.append(y.cpu().numpy())
        trues.append(Y.cpu().numpy())
    val_loss = total_loss / max(n, 1)
    pv = np.clip(np.concatenate(preds), 0, None)
    tv = np.concatenate(trues)
    delz = (pv - tv) / (1 + tv)
    nmad = 1.4826 * np.median(np.abs(delz - np.median(delz)))
    eta = 100 * np.mean(np.abs(delz) > 0.15)
    return val_loss, nmad, eta, pv, tv


def main():
    ap = argparse.ArgumentParser(description="Transfer learning: Track A on real 3D-HST")
    ap.add_argument("--config", required=True, help="Track A YAML config")
    ap.add_argument("--checkpoint", required=True, help="Path to sim-trained z checkpoint")
    ap.add_argument("--exp-name", required=True, help="Experiment name for output files")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--test-split", type=float, default=0.1)
    ap.add_argument("--unfreeze-encoder-layers", type=int, default=0,
                    help="Unfreeze last N encoder layers (0 = head-only)")
    ap.add_argument("--encoder-lr", type=float, default=1e-6,
                    help="Learning rate for unfrozen encoder layers")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Exp: {args.exp_name}")

    config = load_config(args.config)
    model = load_model(config, args.checkpoint)
    unfrozen = freeze_backbone(model, args.unfreeze_encoder_layers)
    model = model.to(device)

    train_df, val_df, test_df = prepare_real_data(args.val_split, args.test_split, args.seed)

    tr_ld = DataLoader(
        HSTGrismDataset(train_df), args.batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    va_ld = DataLoader(HSTGrismDataset(val_df), args.batch_size, shuffle=False, num_workers=0)
    te_ld = DataLoader(HSTGrismDataset(test_df), args.batch_size, shuffle=False, num_workers=0)

    from src.specpt.losses import NMADLoss
    criterion = NMADLoss(normalization_factor="std")

    head_params = [p for p in model.mlp_blocks.parameters() if p.requires_grad]
    head_params += [p for p in model.prediction.parameters() if p.requires_grad]
    head_params += [p for p in model.attention.parameters() if p.requires_grad]
    enc_params = [p for p in model.pretrained_model.parameters() if p.requires_grad]

    if enc_params:
        optimizer = torch.optim.AdamW(
            [
                {"params": head_params, "lr": args.lr},
                {"params": enc_params, "lr": args.encoder_lr},
            ],
            weight_decay=args.weight_decay,
        )
        print(f"  Optimizer: head lr={args.lr:.1e}, encoder lr={args.encoder_lr:.1e}")
    else:
        optimizer = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=args.weight_decay)
        print(f"  Optimizer: head lr={args.lr:.1e}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=15, min_lr=1e-7
    )

    import wandb
    wandb.init(
        project="specpt-hst-sim-z",
        entity="ckb2084-rochester-institute-of-technology",
        name=args.exp_name,
        config={
            "config": args.config,
            "checkpoint": args.checkpoint,
            "epochs": args.epochs,
            "lr": args.lr,
            "encoder_lr": args.encoder_lr,
            "unfreeze_encoder_layers": args.unfreeze_encoder_layers,
            "unfrozen_encoder_layers": unfrozen,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "seed": args.seed,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
        },
    )

    best_val_loss = float("inf")
    best_nmad = 1e9
    best_ep = 0
    wait = 0

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tl = train_one_epoch(model, tr_ld, criterion, optimizer, device)
        vl, vn, ve, _, _ = evaluate(model, va_ld, criterion, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        print(
            f"Ep {ep:3d}/{args.epochs}  loss={tl:.4f}  val_loss={vl:.4f}  "
            f"val_nmad={vn:.5f}  eta={ve:.2f}%  lr={cur_lr:.2e}  {dt:.0f}s"
        )
        wandb.log(
            {"train_loss": tl, "val_loss": vl, "val_nmad": vn, "val_eta": ve, "lr": cur_lr, "epoch": ep}
        )
        scheduler.step(vl)
        if vl < best_val_loss:
            best_val_loss = vl
            best_nmad = vn
            best_ep = ep
            wait = 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "val_loss": vl,
                    "val_nmad": vn,
                    "config": args.config,
                    "checkpoint": args.checkpoint,
                },
                f"checkpoints/{args.exp_name}_best_model.pth",
            )
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stop at ep {ep}")
                break

    print("\n--- Post-training evaluation ---")
    model.eval()
    ckpt = torch.load(
        f"checkpoints/{args.exp_name}_best_model.pth",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    te_loss, te_nmad, te_eta, te_pv, te_tv = evaluate(model, te_ld, criterion, device)
    te_rmse = np.sqrt(np.mean((te_pv - te_tv) ** 2))
    te_bias = float(np.mean(te_pv - te_tv))
    te_delz = (te_pv - te_tv) / (1 + te_tv)
    te_r2 = 1 - np.sum(te_delz**2) / np.sum((te_tv - np.mean(te_tv)) ** 2)
    pred_std = float(np.std(te_pv))
    true_std = float(np.std(te_tv))
    std_ratio = pred_std / max(true_std, 1e-10)

    print(f"Test: NMAD={te_nmad:.5f}  eta={te_eta:.2f}%  RMSE={te_rmse:.4f}  bias={te_bias:+.5f}")
    print(f"       R2={te_r2:.4f}  pred_std={pred_std:.4f}  true_std={true_std:.4f}  std_ratio={std_ratio:.4f}")

    snr_bins = [(2.5, 5), (5, 10), (10, 20), (20, 9999)]
    test_rows = test_df.to_dict("records")
    snr_results = {}
    for lo, hi in snr_bins:
        mask = [(lo <= r["SNR"] < hi) for r in test_rows]
        if sum(mask) < 10:
            continue
        p, t = te_pv[mask], te_tv[mask]
        d = (p - t) / (1 + t)
        b = 1.4826 * np.median(np.abs(d - np.median(d)))
        e = 100 * np.mean(np.abs(d) > 0.15)
        snr_results[f"SNR_{lo:.0f}_{hi:.0f}"] = {"n": int(sum(mask)), "nmad": float(b), "eta": float(e)}
        print(f"  SNR [{lo:.0f}, {hi:.0f}): n={sum(mask):5d}  NMAD={b:.5f}  eta={e:.2f}%")

    metrics = {
        "overall": {
            "nmad": float(te_nmad),
            "eta": float(te_eta),
            "rmse": float(te_rmse),
            "bias": te_bias,
            "r2": float(te_r2),
            "pred_std": pred_std,
            "true_std": true_std,
            "std_ratio": std_ratio,
        },
        "snr_bins": snr_results,
        "n_test": len(test_df),
        "best_epoch": best_ep,
        "best_val_nmad": float(best_nmad),
    }

    out_dir = f"outputs/real_3dhst_tracka/{args.exp_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{args.exp_name}_transfer_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.savetxt(
        f"{out_dir}/{args.exp_name}_transfer_predictions.csv",
        np.column_stack([te_pv, te_tv]),
        delimiter=",",
        header="predicted_z,true_z",
    )

    wandb.log(
        {
            "test_nmad": te_nmad,
            "test_eta": te_eta,
            "test_rmse": te_rmse,
            "test_bias": te_bias,
            "test_r2": float(te_r2),
            "test_pred_std": pred_std,
            "test_true_std": true_std,
            "test_std_ratio": std_ratio,
        }
    )

    print(f"\nSaved: {out_dir}/{args.exp_name}_transfer_metrics.json")
    print(f"Saved: {out_dir}/{args.exp_name}_transfer_predictions.csv")
    print(f"Best val NMAD: {best_nmad:.5f} at ep {best_ep}")
    print(f"Test NMAD: {te_nmad:.5f}")
    wandb.finish()
    print("DONE")


if __name__ == "__main__":
    main()
