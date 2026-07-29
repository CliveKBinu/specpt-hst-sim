#!/usr/bin/env python3
"""Stage 1: RNC (Rank-N-Contrast) encoder training.

Two modes:
  --freeze_encoder: only projection head learns (exp_050)
  (default, no flag): encoder + projection learn at differential LR (exp_051)

Saves checkpoint used by rnc_stage2.py.
"""
import sys, os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import time, argparse, wandb
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '.')
from src.specpt.model import SpecPT, SpectrumNormalizer, Swish
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
SIM_VAL_PATH = '/home/ckb2084/research/specpt-hst-sim/data/training_format/grism_training_sim_v3_regrid.parquet'


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


class EncoderOnly(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.proj_to_d_model = pretrained_model.proj_to_d_model
        self.encoder = pretrained_model.transformer_encoder

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pretrained_model.forward_conv(x)
        x = x.flatten(start_dim=1)
        x = self.proj_to_d_model(x)
        x = x.unsqueeze(0)
        x = self.encoder(x)
        return x.squeeze(0)


class ProjectionHead(nn.Module):
    def __init__(self, d_model=512, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            Swish(),
            nn.Linear(256, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class RnCLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        bs = features.shape[0]
        feat = features.reshape(-1, features.shape[-1])
        N = 2 * bs
        z = torch.cat([labels.reshape(-1), labels.reshape(-1)], dim=0)
        z_dist = torch.abs(z.unsqueeze(0) - z.unsqueeze(1))
        sim = -torch.cdist(feat, feat, p=2)
        exp_sim = torch.exp(sim / self.temperature)
        diag = torch.eye(N, dtype=torch.bool, device=feat.device)
        loss = 0.0
        for i in range(N):
            z_dist_i = z_dist[i]
            mask = z_dist_i[None, :] >= z_dist_i[:, None]
            mask = mask & ~diag[i:i+1, :]
            num = exp_sim[i].clone()
            den = (exp_sim[i][None, :] * mask).sum(dim=1)
            num[i] = 1.0
            den[i] = 1.0
            loss += -torch.log(num / (den + 1e-8)).sum()
        return loss / (N * (N - 1))


class SpectralAugment:
    def __init__(self, p=0.5, noise_sigma=(0.01, 0.05), mask_chunks=(1, 3), mask_size=(20, 50), shift_max=3):
        self.p = p
        self.noise_sigma = noise_sigma
        self.mask_chunks = mask_chunks
        self.mask_size = mask_size
        self.shift_max = shift_max

    def __call__(self, x):
        x = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x.copy())
        if torch.rand(1).item() < self.p:
            sigma = torch.empty(1).uniform_(*self.noise_sigma).item()
            x = x + torch.randn_like(x) * sigma
        if torch.rand(1).item() < self.p:
            n = torch.randint(self.mask_chunks[0], self.mask_chunks[1] + 1, (1,)).item()
            for _ in range(n):
                size = torch.randint(self.mask_size[0], self.mask_size[1] + 1, (1,)).item()
                start = torch.randint(0, len(x) - size, (1,)).item()
                x[start:start+size] = 0.0
        if torch.rand(1).item() < self.p:
            shift = torch.randint(-self.shift_max, self.shift_max + 1, (1,)).item()
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=0)
        return x


class TwoCropRealDataset(Dataset):
    def __init__(self, df, augment_fn):
        from src.specpt.model import SpectrumNormalizer
        specs = [SpectrumNormalizer.zscore_normalize(s) for s in df['spec'].values]
        self.specs = [torch.from_numpy(np.asarray(s, dtype=np.float32)).float() for s in specs]
        self.zs = torch.from_numpy(df['z'].values.astype(np.float32)).float()
        self.augment = augment_fn

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        X = self.specs[idx]
        return self.augment(X), self.augment(X), self.zs[idx]


def train_stage1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Freeze encoder: {args.freeze_encoder}")

    print("Loading real data...")
    df_real = pd.read_pickle(REAL_DATA_PATH)
    df_real = df_real[df_real['SNR'] >= 2.5].reset_index(drop=True)
    real_train, real_val, real_test = prepare_real(df_real, args.val_split, args.test_split, args.seed)
    print(f"Real: train={len(real_train)} val={len(real_val)} test={len(real_test)}")
    del real_test  # not used in Stage 1

    augment_fn = SpectralAugment(p=args.augment_p)
    train_ds = TwoCropRealDataset(real_train, augment_fn)
    val_ds = TwoCropRealDataset(real_val, augment_fn)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("Loading regridded autoencoder...")
    ckpt = torch.load(REGRID_AE_CKPT, map_location='cpu', weights_only=False)
    sp = SpecPT(input_size=7781, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                dim_feedforward=2048, dropout=0.1)
    sp.load_state_dict(ckpt['model_state_dict'], strict=True)
    encoder = EncoderOnly(sp).to(device)
    projection = ProjectionHead(d_model=512, proj_dim=args.proj_dim).to(device)

    if args.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        params = [p for p in projection.parameters() if p.requires_grad]
        print("Encoder frozen — training projection head only")
    else:
        enc_params = [p for p in encoder.parameters() if p.requires_grad]
        proj_params = [p for p in projection.parameters() if p.requires_grad]
        params = [
            {'params': enc_params, 'lr': args.enc_lr},
            {'params': proj_params, 'lr': args.proj_lr},
        ]
        print(f"Encoder unfrozen — enc LR {args.enc_lr}, proj LR {args.proj_lr}")

    nt = sum(p.numel() for p in (encoder.parameters() if args.freeze_encoder else list(encoder.parameters()) + list(projection.parameters())) if p.requires_grad)
    print(f"Trainable: {nt:,}")

    opt = torch.optim.AdamW(params, lr=args.proj_lr if args.freeze_encoder else args.proj_lr, weight_decay=args.weight_decay)
    rnc_loss_fn = RnCLoss(temperature=args.temperature)

    ref_indices = list(range(min(16, len(real_train))))
    ref_specs = [train_ds.specs[i] for i in ref_indices]
    ref_batch_raw = torch.stack(ref_specs, dim=0).to(device)
    sp.eval()
    with torch.no_grad():
        initial_recon = sp(ref_batch_raw)
        initial_recon_mse = F.mse_loss(initial_recon, ref_batch_raw).item()
    print(f"Initial recon MSE (16 samples): {initial_recon_mse:.6f}")

    run_name = f"{args.exp_name}_stage1"
    wandb.init(project='specpt-hst-sim-z', entity='ckb2084-rochester-institute-of-technology',
               name=run_name, config=vars(args))
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        encoder.train(not args.freeze_encoder)
        projection.train()
        train_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for X1, X2, z in train_ld:
            X1, X2, z = X1.to(device), X2.to(device), z.to(device)
            with torch.set_grad_enabled(not args.freeze_encoder):
                f1 = encoder(X1)
                f2 = encoder(X2)
            p1 = projection(f1)
            p2 = projection(f2)
            features = torch.stack([p1, p2], dim=1)
            loss = rnc_loss_fn(features, z)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_batches += 1

        encoder.eval()
        projection.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for X1, X2, z in val_ld:
                X1, X2, z = X1.to(device), X2.to(device), z.to(device)
                f1 = encoder(X1)
                f2 = encoder(X2)
                p1 = projection(f1)
                p2 = projection(f2)
                features = torch.stack([p1, p2], dim=1)
                loss = rnc_loss_fn(features, z)
                val_loss += loss.item()
                n_val += 1

        avg_train = train_loss / n_batches
        avg_val = val_loss / n_val
        wandb.log({
            'epoch': epoch,
            'train/rnc_loss': avg_train,
            'val/rnc_loss': avg_val,
            'lr': opt.param_groups[0]['lr'],
            'time_sec': time.time() - t0,
        })
        if epoch % 5 == 0:
            with torch.no_grad():
                recon = sp(ref_batch_raw)
                mse = F.mse_loss(recon, ref_batch_raw).item()
                drift = mse / max(initial_recon_mse, 1e-10)
            wandb.log({'recon_mse': mse, 'recon_mse_drift': drift, 'epoch': epoch})
        print(f"Epoch {epoch:>3}/{args.epochs}  train_rnc={avg_train:.4f}  val_rnc={avg_val:.4f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}  {time.time() - t0:.0f}s")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = f"checkpoints/{args.exp_name}_stage1_best.pth"
            torch.save({
                'args': args,
                'epoch': epoch,
                'projection_state_dict': projection.state_dict(),
                'encoder_state_dict': sp.state_dict(),
                'val_rnc_loss': avg_val,
                'train_rnc_loss': avg_train,
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stop at epoch {epoch} (best {best_epoch}, val_rnc {best_val_loss:.4f})")
                break

    wandb.summary['best/epoch'] = best_epoch
    wandb.summary['best/val_rnc_loss'] = best_val_loss
    wandb.finish()
    print(f"Stage 1 done. Best epoch {best_epoch}, val_rnc_loss {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--freeze_encoder', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--enc_lr', type=float, default=1e-5)
    parser.add_argument('--proj_lr', type=float, default=3e-3)
    parser.add_argument('--augment_p', type=float, default=0.5)
    args = parser.parse_args()
    train_stage1(args)
