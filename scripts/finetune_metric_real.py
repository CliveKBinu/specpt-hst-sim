#!/usr/bin/env python3
"""Metric learning on regridded autoencoder backbone + real 3D-HST.

Uses NTXent (SimCLR-style) contrastive loss with redshift-based
positive mining (delta=0.05 relative to 1+z). Inference via k-NN
retrieval (k=10, softmax-weighted average).
"""
import sys, os, numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
import time, argparse, wandb

sys.path.insert(0, '.')
from src.specpt.model import SpecPT, Swish
from src.specpt.dataloader import HSTGrismDataset
from torch.utils.data import DataLoader

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


def prepare_data(df, val_split, test_split, seed):
    specs = [_spec_from_clean(r) for _, r in df.iterrows()]
    df['spec'] = specs
    df['TARGETID'] = df['grism_id']
    train, val, test = split_by_grism_id(
        df, test_size=val_split + test_split,
        val_size=test_split / (val_split + test_split), random_state=seed,
    )
    cols = ['TARGETID', 'z', 'spec', 'SNR']
    return train[cols].copy(), val[cols].copy(), test[cols].copy()


class MetricLearningHead(nn.Module):
    """Projection head: backbone → 128-dim L2-normalized embedding."""
    def __init__(self, pretrained_model, embed_dim=128):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.proj_to_d_model = pretrained_model.proj_to_d_model
        self.encoder = pretrained_model.transformer_encoder
        self.projection = nn.Sequential(
            nn.Linear(512, 256), Swish(), nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pretrained_model.forward_conv(x)
        x = x.flatten(start_dim=1)
        x = self.proj_to_d_model(x)
        x = x.unsqueeze(0)
        x = self.encoder(x)
        x = x.squeeze(0)
        z = self.projection(x)
        return F.normalize(z, dim=-1)


def ntxent_loss(z, y, tau=0.07, delta=0.05):
    """NTXent loss with redshift-based positive mining.

    Positive pairs: |z_i - z_j| / (1 + z_i) < delta
    """
    y = y.flatten()
    B = z.size(0)
    sim = z @ z.T / tau
    rel_dist = torch.abs(y.unsqueeze(0) - y.unsqueeze(-1)) / (1 + y.unsqueeze(-1))
    pos_mask = (rel_dist < delta) & ~torch.eye(B, device=z.device, dtype=bool)
    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1) - exp_sim.diag()
    numer = (exp_sim * pos_mask.float()).sum(dim=1).clamp(min=1e-8)
    has_pos = pos_mask.any(dim=1)
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    loss = -torch.log(numer[has_pos] / denom[has_pos]).mean()
    return loss


@torch.no_grad()
def encode_dataset(model, loader, device):
    embeddings, redshifts = [], []
    for X, Y, _, _ in loader:
        z = model(X.to(device))
        embeddings.append(z.cpu().numpy())
        redshifts.append(Y.numpy())
    return np.concatenate(embeddings), np.concatenate(redshifts)


def knn_predict(z_query, z_ref, y_ref, k=10, tau=0.07):
    """Softmax-weighted k-NN from L2-normalized embeddings."""
    y_ref = y_ref.flatten()
    sim = z_query @ z_ref.T
    topk_sim, topk_idx = sim.topk(k, dim=-1)
    weights = F.softmax(topk_sim / tau, dim=-1)
    preds = (weights * y_ref[topk_idx]).sum(dim=-1)
    return preds


def compute_metrics(pv, tv):
    delz = (pv - tv) / (1 + tv)
    nmad = float(1.4826 * np.median(np.abs(delz - np.median(delz))))
    eta = float(100 * np.mean(np.abs(delz) > 0.15))
    rmse = float(np.sqrt(np.mean((pv - tv) ** 2)))
    return nmad, eta, rmse


def train_metric(args, train, val, test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tr_ld = DataLoader(HSTGrismDataset(train), args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    va_ld = DataLoader(HSTGrismDataset(val),   args.batch_size, shuffle=False, num_workers=0)
    te_ld = DataLoader(HSTGrismDataset(test),  args.batch_size, shuffle=False, num_workers=0)
    print(f"Train {len(train)}  Val {len(val)}  Test {len(test)}")
    print("Loading regridded autoencoder...")
    ckpt = torch.load(REGRID_AE_CKPT, map_location='cpu', weights_only=False)
    sp = SpecPT(input_size=7781, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                dim_feedforward=2048, dropout=0.1)
    sp.load_state_dict(ckpt['model_state_dict'], strict=True)
    model = MetricLearningHead(sp, embed_dim=args.embed_dim).to(device)
    print(f"Projection: {sum(p.numel() for p in model.projection.parameters()):,} params")
    if args.freeze_backbone:
        for p in sp.parameters():
            p.requires_grad = False
        print("Backbone FROZEN — head only")
    else:
        print("Backbone UNFROZEN — end-to-end training")
    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {nt:,}/{sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    model.eval()
    z_ref, y_ref = encode_dataset(model, tr_ld, device)
    z_val, y_val = encode_dataset(model, va_ld, device)
    ref_t = torch.from_numpy(z_ref).to(device)
    val_t = torch.from_numpy(z_val).to(device)
    yt = torch.from_numpy(y_ref).to(device)
    preds = knn_predict(val_t, ref_t, yt, k=args.k).cpu().numpy()
    diag_nmad, diag_eta, _ = compute_metrics(preds, y_val)
    print(f"Diag: val_nmad={diag_nmad:.4f}  eta={diag_eta:.2f}%")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.1, patience=20, min_lr=1e-7,
    )
    best_nmad = 1e9
    best_ep = 0
    patience = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        if args.freeze_backbone:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()
            model.encoder.eval()
        t0 = time.time()
        tl = 0.0
        for X, Y, _, _ in tr_ld:
            X, Y = X.to(device), Y.to(device)
            opt.zero_grad()
            z = model(X)
            loss = ntxent_loss(z, Y, tau=args.tau, delta=args.delta)
            if loss.item() == 0:
                continue
            loss.backward()
            opt.step()
            tl += loss.item() * X.size(0)
        tl /= len(tr_ld.dataset)
        model.eval()
        z_ref, y_ref = encode_dataset(model, tr_ld, device)
        z_val, y_val = encode_dataset(model, va_ld, device)
        ref_t = torch.from_numpy(z_ref).to(device)
        val_t = torch.from_numpy(z_val).to(device)
        yt = torch.from_numpy(y_ref).to(device)
        preds = knn_predict(val_t, ref_t, yt, k=args.k).cpu().numpy()
        nmad, eta, _ = compute_metrics(preds, y_val)
        cur_lr = opt.param_groups[0]['lr']
        print(f"Ep {ep:2d}/{args.epochs}  loss={tl:.4f}  val_nmad={nmad:.5f}  eta={eta:.2f}%  lr={cur_lr:.2e}  {time.time()-t0:.0f}s")
        wandb.log({'train_loss': tl, 'val_nmad': nmad, 'val_eta': eta, 'lr': cur_lr, 'epoch': ep})
        scheduler.step(nmad)
        if nmad < best_nmad:
            best_nmad = nmad
            best_ep = ep
            patience = 0
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': ep, 'model_state_dict': model.state_dict(),
                'val_nmad': nmad, 'exp_name': args.exp_name,
            }, f'checkpoints/{args.exp_name}_best_model.pth')
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stop at ep {ep}")
                break
    model.eval()
    z_ref, y_ref = encode_dataset(model, tr_ld, device)
    z_te, y_te = encode_dataset(model, te_ld, device)
    ref_t = torch.from_numpy(z_ref).to(device)
    te_t = torch.from_numpy(z_te).to(device)
    yt = torch.from_numpy(y_ref).to(device)
    preds = knn_predict(te_t, ref_t, yt, k=args.k).cpu().numpy()
    test_nmad, test_eta, test_rmse = compute_metrics(preds, y_te)
    print(f"\nTest:  NMAD={test_nmad:.5f}  eta={test_eta:.2f}%  RMSE={test_rmse:.4f}")
    print(f"Best val NMAD: {best_nmad:.5f} at ep {best_ep}")
    return best_nmad, best_ep, test_nmad, test_eta, test_rmse


def main():
    parser = argparse.ArgumentParser(description='Metric learning on regrid autoencoder + real 3D-HST')
    parser.add_argument('--exp_name', default='exp_043')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--freeze_backbone', action='store_true', default=False)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--delta', type=float, default=0.05)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()
    print(f"=== {args.exp_name}: metric learning embed_dim={args.embed_dim} tau={args.tau} delta={args.delta} k={args.k} ===")
    df = pd.read_pickle(REAL_DATA_PATH)
    df = df[df['SNR'] >= 2.5].reset_index(drop=True)
    print(f"Loaded {len(df)} real 3D-HST spectra (SNR >= 2.5)")
    train, val, test = prepare_data(df, args.val_split, args.test_split, args.seed)
    print(f"  Final: train={len(train)}  val={len(val)}  test={len(test)}")
    wandb.init(project='specpt-hst-sim-z', entity='ckb2084-rochester-institute-of-technology',
               name=args.exp_name, config=vars(args))
    best_val_nmad, best_ep, test_nmad, test_eta, test_rmse = train_metric(args, train, val, test)
    wandb.log({
        'best_val_nmad': best_val_nmad, 'best_epoch': best_ep,
        'test_nmad': test_nmad, 'test_eta': test_eta, 'test_rmse': test_rmse,
        'train_samples': len(train), 'val_samples': len(val), 'test_samples': len(test),
    })
    wandb.finish()
    print("DONE")


if __name__ == '__main__':
    main()
