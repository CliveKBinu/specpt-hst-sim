#!/usr/bin/env python3
"""Stage 2: Linear probe on RNC-trained frozen encoder + projection.

Loads Stage 1 checkpoint (encoder + projection state dicts),
freezes both, trains Linear(128, 1) head with NMADLoss.

Evaluates on real test set (primary metric: test NMAD).
"""
import sys, os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import time, argparse, wandb
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from src.specpt.model import SpecPT, SpectrumNormalizer, Swish
from src.specpt.dataloader import HSTGrismDataset
from src.specpt.losses import NMADLoss

sys.path.insert(0, '/home/ckb2084/research/SpecPT')
import importlib.util
_spec = importlib.util.spec_from_file_location('augment',
    '/home/ckb2084/research/SpecPT/specpt/augment.py')
_aug = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_aug)
split_by_grism_id = _aug.split_by_grism_id

TRAIN_WAVES = np.linspace(10800.0, 17100.0, 7781)
PAD = (TRAIN_WAVES < 11000) | (TRAIN_WAVES > 16500)

REGRID_AE_CKPT = '/home/ckb2084/research/specpt-hst-sim/checkpoints/autoencoder_regrid_autoencoder_best.pth'
REAL_DATA_PATH = '/home/ckb2084/research/SpecPT/data/grism_specPT_v5.pkl'


def _spec_from_clean(row):
    safe = np.where(row['sensitivity_resampled'] == 0, 1e-8, row['sensitivity_resampled'])
    s = (row['clean_flux_resampled'] / safe).astype(np.float32)
    s[PAD] = np.nan
    return s


def prepare_real(df, val_split, test_split, seed):
    specs = [_spec_from_clean(r) for _, r in df.iterrows()]
    df['spec'] = specs
    df['TARGETID'] = df['grism_id']
    train, val, test = split_by_grism_id(
        df, test_size=val_split + test_split,
        val_size=test_split / (val_split + test_split), random_state=seed,
    )
    cols = ['TARGETID', 'z', 'spec', 'SNR']
    return train[cols].copy(), val[cols].copy(), test[cols].copy()


def compute_metrics(z_true, z_pred):
    z_pred = z_pred.flatten()
    z_true = z_true.flatten()
    mask = ~np.isnan(z_pred) & ~np.isnan(z_true)
    z_pred, z_true = z_pred[mask], z_true[mask]
    delz = (z_pred - z_true) / (1 + z_true)
    nmad = 1.4826 * np.median(np.abs(delz - np.median(delz)))
    eta = 100 * np.mean(np.abs(delz) > 0.15)
    rmse = np.sqrt(np.mean(delz ** 2))
    bias = np.median(delz)
    r2 = 1 - np.sum((z_true - z_pred) ** 2) / np.sum((z_true - z_true.mean()) ** 2)
    return {'nmad': nmad, 'eta': eta, 'rmse': rmse, 'bias': bias, 'r2': r2, 'n': len(z_true)}


class ProjectionHead(nn.Module):
    def __init__(self, d_model=512, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            Swish(),
            nn.Linear(256, proj_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=-1)


class RNCEncoderWithProjection(nn.Module):
    def __init__(self, encoder_state_dict, projection_state_dict, freeze=True):
        super().__init__()
        sp = SpecPT(input_size=7781, d_model=512, nhead=8, num_encoder_layers=3,
                    num_decoder_layers=3, dim_feedforward=2048, dropout=0.1)
        sp.load_state_dict(encoder_state_dict, strict=True)
        self.encoder = sp
        self.proj = ProjectionHead(d_model=512, proj_dim=128)
        self.proj.load_state_dict(projection_state_dict)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.proj.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder.forward_conv(x)
        x = x.flatten(start_dim=1)
        x = self.encoder.proj_to_d_model(x)
        x = x.unsqueeze(0)
        x = self.encoder.transformer_encoder(x)
        x = x.squeeze(0)
        x = self.proj(x)
        return x


def train_stage2(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading Stage 1 checkpoint: {args.stage1_ckpt}")
    s1 = torch.load(args.stage1_ckpt, map_location='cpu', weights_only=False)
    backbone = RNCEncoderWithProjection(s1['encoder_state_dict'], s1['projection_state_dict'], freeze=True).to(device)
    head = nn.Sequential(nn.Linear(128, 1), nn.Softplus()).to(device)

    nt = sum(p.numel() for p in head.parameters())
    print(f"Trainable: {nt:,} (head only, encoder+projection frozen)")

    print("Loading real data...")
    df_real = pd.read_pickle(REAL_DATA_PATH)
    df_real = df_real[df_real['SNR'] >= 2.5].reset_index(drop=True)
    real_train, real_val, real_test = prepare_real(df_real, args.val_split, args.test_split, args.seed)
    print(f"Real: train={len(real_train)} val={len(real_val)} test={len(real_test)}")

    tr_ld = DataLoader(HSTGrismDataset(real_train), args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    va_ld = DataLoader(HSTGrismDataset(real_val),   args.batch_size, shuffle=False, num_workers=0)
    te_ld = DataLoader(HSTGrismDataset(real_test),  args.batch_size, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.1, patience=args.scheduler_patience, verbose=True)
    criterion = NMADLoss()

    run_name = f"{args.exp_name}_stage2"
    wandb.init(project='specpt-hst-sim-z', entity='ckb2084-rochester-institute-of-technology',
               name=run_name, config=vars(args))

    best_val_nmad = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        backbone.eval()
        head.train()
        train_loss = 0.0
        t0 = time.time()
        for X, z, _, _ in tr_ld:
            X, z = X.to(device), z.to(device)
            with torch.no_grad():
                features = backbone(X)
            pred = head(features)
            loss = criterion(pred, z)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(tr_ld)

        backbone.eval()
        head.eval()
        all_pred, all_z = [], []
        with torch.no_grad():
            for X, z, _, _ in va_ld:
                X, z = X.to(device), z.to(device)
                features = backbone(X)
                pred = head(features)
                all_pred.append(pred.cpu().numpy())
                all_z.append(z.cpu().numpy())

        all_pred = np.concatenate(all_pred).flatten()
        all_z = np.concatenate(all_z).flatten()
        metrics = compute_metrics(all_z, all_pred)
        val_nmad = metrics['nmad']
        val_eta = metrics['eta']

        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'val/real_nmad': val_nmad,
            'val/real_eta': val_eta,
            'lr': opt.param_groups[0]['lr'],
        })
        print(f"Epoch {epoch:>3}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_nmad={val_nmad:.4f}  val_eta={val_eta:.2f}%  "
              f"lr={opt.param_groups[0]['lr']:.2e}  {time.time() - t0:.0f}s")

        scheduler.step(val_nmad)

        if val_nmad < best_val_nmad:
            best_val_nmad = val_nmad
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = f"checkpoints/{args.exp_name}_stage2_best.pth"
            torch.save({
                'args': args,
                'epoch': epoch,
                'head_state_dict': head.state_dict(),
                'backbone_state_dict': s1['encoder_state_dict'],
                'projection_state_dict': s1['projection_state_dict'],
                'val_nmad': val_nmad,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stop at epoch {epoch} (best {best_epoch}, val_nmad {best_val_nmad:.4f})")
                break

    print(f"Stage 2 done. Best epoch {best_epoch}, val_nmad {best_val_nmad:.4f}")
    print("Running test evaluation...")
    backbone.eval()
    head.eval()
    all_pred, all_z = [], []
    with torch.no_grad():
        for X, z, _, _ in te_ld:
            X, z = X.to(device), z.to(device)
            features = backbone(X)
            pred = head(features)
            all_pred.append(pred.cpu().numpy())
            all_z.append(z.cpu().numpy())
    all_pred = np.concatenate(all_pred).flatten()
    all_z = np.concatenate(all_z).flatten()
    test_metrics = compute_metrics(all_z, all_pred)

    for k, v in test_metrics.items():
        wandb.summary[f'test/real_{k}'] = v
    wandb.summary['best/epoch'] = best_epoch
    wandb.summary['best/val_nmad'] = best_val_nmad

    print(f"Test: NMAD={test_metrics['nmad']:.4f}  η={test_metrics['eta']:.2f}%  "
          f"RMSE={test_metrics['rmse']:.4f}  R²={test_metrics['r2']:.4f}  n={test_metrics['n']}")
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--stage1_ckpt', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--scheduler_patience', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    args = parser.parse_args()
    train_stage2(args)
