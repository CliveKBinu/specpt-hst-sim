#!/usr/bin/env python3
"""Fine-tune on regridded autoencoder backbone + real 3D-HST.

Modes:    --mode no_augment / augment
Heads:    --head_type simple / linear / mlp / resnet / enhanced
Backbone: --freeze_backbone (default: frozen)
"""
import sys, os, numpy as np, pandas as pd, torch, torch.nn as nn, time, argparse, wandb

sys.path.insert(0, '.')
from src.specpt.model import SpecPT, EnhancedSpecPTForRedshift, SpectrumNormalizer, Swish, ImprovedResidualMLPBlock
from src.specpt.dataloader import HSTGrismDataset
from src.specpt.losses import NMADLoss
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/ckb2084/research/SpecPT')
import importlib.util
_spec = importlib.util.spec_from_file_location('augment',
    '/home/ckb2084/research/SpecPT/specpt/augment.py')
_aug = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_aug)
augment_spectra = _aug.augment_spectra
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


def _mask_pad(s):
    s = s.copy().astype(np.float32)
    s[PAD] = np.nan
    return s


def prepare_no_aug(df, val_split, test_split, seed):
    specs = [_spec_from_clean(r) for _, r in df.iterrows()]
    df['spec'] = specs
    df['TARGETID'] = df['grism_id']
    train, val, test = split_by_grism_id(
        df, test_size=val_split + test_split,
        val_size=test_split / (val_split + test_split), random_state=seed,
    )
    cols = ['TARGETID', 'z', 'spec', 'SNR']
    return train[cols].copy(), val[cols].copy(), test[cols].copy()


def prepare_aug(df, val_split, test_split, seed):
    train_raw, val_raw, test_raw = split_by_grism_id(
        df, test_size=val_split + test_split,
        val_size=test_split / (val_split + test_split), random_state=seed,
    )
    print(f"  Pre-split: train={len(train_raw)} val={len(val_raw)} test={len(test_raw)}")
    for sub in [val_raw, test_raw]:
        sub['spec'] = [_spec_from_clean(r) for _, r in sub.iterrows()]
        sub['TARGETID'] = sub['grism_id']
    aug = augment_spectra(train_raw, random_state=seed)
    aug['spec'] = aug['spec'].apply(_mask_pad)
    print(f"  Augmented: {len(aug)} copies from {len(train_raw)} originals")
    orig = pd.DataFrame({
        'TARGETID': train_raw['grism_id'].values,
        'z': train_raw['z'].values,
        'spec': [_spec_from_clean(r) for _, r in train_raw.iterrows()],
        'SNR': train_raw['SNR'].values,
    })
    train = pd.concat([orig, aug[['TARGETID', 'z', 'spec', 'SNR']]], ignore_index=True)
    val = val_raw[['TARGETID', 'z', 'spec', 'SNR']].copy()
    test = test_raw[['TARGETID', 'z', 'spec', 'SNR']].copy()
    return train, val, test


class SimpleRedshiftHead(nn.Module):
    """Simple redshift prediction head (no attention, no residual blocks)."""
    def __init__(self, pretrained_model, head_type='simple'):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.proj_to_d_model = pretrained_model.proj_to_d_model
        self.encoder = pretrained_model.transformer_encoder
        if head_type == 'linear':
            self.head = nn.Sequential(nn.Linear(512, 1), nn.Softplus())
        elif head_type == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(512, 256), Swish(), nn.Dropout(0.1),
                nn.Linear(256, 128), Swish(), nn.Dropout(0.1),
                nn.Linear(128, 1), nn.Softplus(),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(512, 256), Swish(), nn.Dropout(0.1),
                nn.Linear(256, 1), nn.Softplus(),
            )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pretrained_model.forward_conv(x)
        x = x.flatten(start_dim=1)
        x = self.proj_to_d_model(x)
        x = x.unsqueeze(0)
        x = self.encoder(x)
        x = x.squeeze(0)
        return self.head(x)


class ResNetRedshiftHead(nn.Module):
    """Residual MLP redshift head (no attention, just residual blocks)."""
    def __init__(self, pretrained_model, num_blocks=3):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.proj_to_d_model = pretrained_model.proj_to_d_model
        self.encoder = pretrained_model.transformer_encoder
        self.blocks = nn.Sequential(*[
            ImprovedResidualMLPBlock(512, 512, 0.1) for _ in range(num_blocks)
        ])
        self.prediction = nn.Sequential(
            nn.Linear(512, 256), Swish(), nn.Dropout(0.1),
            nn.Linear(256, 1), nn.Softplus(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pretrained_model.forward_conv(x)
        x = x.flatten(start_dim=1)
        x = self.proj_to_d_model(x)
        x = x.unsqueeze(0)
        x = self.encoder(x)
        x = x.squeeze(0)
        x = self.blocks(x)
        return self.prediction(x)


def train_linear_probe(args, train, val, test):
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
    if args.head_type in ('simple', 'linear', 'mlp'):
        model = SimpleRedshiftHead(sp, head_type=args.head_type).to(device)
        print(f"Head: {args.head_type} ({sum(p.numel() for p in model.head.parameters()):,} params)")
    elif args.head_type == 'resnet':
        model = ResNetRedshiftHead(sp, num_blocks=3).to(device)
        print(f"Head: resnet ({sum(p.numel() for p in model.parameters()):,} params)")
    else:
        model = EnhancedSpecPTForRedshift(sp, num_mlp_blocks=5, mlp_dim=512, dropout_rate=0.1).to(device)
        print(f"Head: enhanced ({sum(p.numel() for p in model.mlp_blocks.parameters()) + sum(p.numel() for p in model.prediction.parameters()):,} params)")
    if args.freeze_backbone:
        backbone = model.pretrained_model
        for p in backbone.parameters():
            p.requires_grad = False
        # enable all head-specific parameters
        for p in model.parameters():
            if p.requires_grad is False:
                p.requires_grad = True
        # re-freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False
        print("Backbone FROZEN — head only")
    else:
        for p in model.parameters():
            p.requires_grad = True
        print("Backbone UNFROZEN — end-to-end training")
    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {nt:,}/{sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    criterion = NMADLoss(normalization_factor='std')
    model.eval()
    for Xd, Yd, _, _ in va_ld:
        with torch.no_grad():
            zd = model(Xd.to(device)).flatten()
        print(f"Diag: loss={criterion(zd, Yd.to(device)).item():.4f}")
        break
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.1, patience=20, min_lr=1e-7
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
            y = model(X).flatten()
            loss = criterion(y, Y)
            loss.backward()
            opt.step()
            tl += loss.item() * X.size(0)
        tl /= len(tr_ld.dataset)
        model.eval()
        with torch.no_grad():
            ap, at = [], []
            for X, Y, _, _ in va_ld:
                y = model(X.to(device)).flatten()
                ap.append(y.cpu().numpy())
                at.append(Y.numpy())
        pv = np.clip(np.concatenate(ap), 0, None)
        tv = np.concatenate(at)
        delz = (pv - tv) / (1 + tv)
        nmad = 1.4826 * np.median(np.abs(delz - np.median(delz)))
        eta = 100 * np.mean(np.abs(delz) > 0.15)
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
                'val_nmad': nmad, 'mode': args.mode, 'exp_name': args.exp_name,
            }, f'checkpoints/{args.exp_name}_best_model.pth')
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stop at ep {ep}")
                break
    model.eval()
    with torch.no_grad():
        ap, at = [], []
        for X, Y, _, _ in te_ld:
            y = model(X.to(device)).flatten()
            ap.append(y.cpu().numpy())
            at.append(Y.numpy())
    pv = np.clip(np.concatenate(ap), 0, None)
    tv = np.concatenate(at)
    delz = (pv - tv) / (1 + tv)
    test_nmad = 1.4826 * np.median(np.abs(delz - np.median(delz)))
    test_eta = 100 * np.mean(np.abs(delz) > 0.15)
    test_rmse = np.sqrt(np.mean((pv - tv) ** 2))
    print(f"\nTest:  NMAD={test_nmad:.5f}  eta={test_eta:.2f}%  RMSE={test_rmse:.4f}")
    print(f"Best val NMAD: {best_nmad:.5f} at ep {best_ep}")
    return best_nmad, best_ep, test_nmad, test_eta, test_rmse


def main():
    parser = argparse.ArgumentParser(description='Fine-tune on regridded autoencoder + real 3D-HST')
    parser.add_argument('--mode', choices=['no_augment', 'augment'], default='no_augment')
    parser.add_argument('--exp_name', default='exp_037')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--freeze_backbone', action='store_true', default=False)
    parser.add_argument('--head_type', choices=['simple', 'linear', 'mlp', 'resnet', 'enhanced'], default='simple')
    args = parser.parse_args()
    print(f"=== {args.exp_name}: mode={args.mode} head={args.head_type} ===")
    df = pd.read_pickle(REAL_DATA_PATH)
    df = df[df['SNR'] >= 2.5].reset_index(drop=True)
    print(f"Loaded {len(df)} real 3D-HST spectra (SNR >= 2.5)")
    if args.mode == 'no_augment':
        train, val, test = prepare_no_aug(df, args.val_split, args.test_split, args.seed)
    else:
        train, val, test = prepare_aug(df, args.val_split, args.test_split, args.seed)
    print(f"  Final: train={len(train)}  val={len(val)}  test={len(test)}")
    wandb.init(project='specpt-hst-sim-z', entity='ckb2084-rochester-institute-of-technology',
               name=args.exp_name, config=vars(args))
    best_val_nmad, best_ep, test_nmad, test_eta, test_rmse = train_linear_probe(args, train, val, test)
    wandb.log({
        'best_val_nmad': best_val_nmad, 'best_epoch': best_ep,
        'test_nmad': test_nmad, 'test_eta': test_eta, 'test_rmse': test_rmse,
        'train_samples': len(train), 'val_samples': len(val), 'test_samples': len(test),
    })
    wandb.finish()
    print("DONE")


if __name__ == '__main__':
    main()
