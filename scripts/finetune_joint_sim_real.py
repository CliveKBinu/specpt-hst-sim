#!/usr/bin/env python3
"""Joint sim+real training with unfrozen encoder and sim reconstruction regularizer.

Modes: --lr_policy flat / differential
"""
import sys, os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import time, argparse, wandb

sys.path.insert(0, '.')
from src.specpt.model import SpecPT, SpectrumNormalizer, Swish
from src.specpt.dataloader import HSTGrismDataset
from src.specpt.losses import NMADLoss
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
SIM_DATA_PATH = '/home/ckb2084/research/specpt-hst-sim/data/training_format/grism_training_sim_v3_regrid.parquet'


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


def prepare_sim(df):
    df = df.rename(columns={'grism_id': 'TARGETID'})
    cols = ['TARGETID', 'z', 'spec', 'SNR']
    return df[cols].copy()


class SimpleRedshiftHead(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.proj_to_d_model = pretrained_model.proj_to_d_model
        self.encoder = pretrained_model.transformer_encoder
        self.decoder = pretrained_model.transformer_decoder
        self.linear1 = pretrained_model.linear1
        self.linear2 = pretrained_model.linear2
        self.head = nn.Sequential(
            nn.Linear(512, 256), Swish(), nn.Dropout(0.1),
            nn.Linear(256, 1), nn.Softplus(),
        )

    def encode(self, x):
        x = x.unsqueeze(1)
        x = self.pretrained_model.forward_conv(x)
        x = x.flatten(start_dim=1)
        x = self.proj_to_d_model(x)
        x = x.unsqueeze(0)
        x = self.encoder(x)
        return x.squeeze(0)

    def decode(self, encoded):
        x = encoded.unsqueeze(0)
        x = self.decoder(x, x)
        x = x.squeeze(0)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def forward(self, x, return_encoded=False):
        encoded = self.encode(x)
        pred = self.head(encoded)
        if return_encoded:
            return pred, encoded
        return pred


class JointSimRealLoader:
    def __init__(self, real_dataset, sim_dataset, batch_size, real_frac):
        self.real = real_dataset
        self.sim = sim_dataset
        self.batch_size = batch_size
        self.real_bs = max(1, int(batch_size * real_frac))
        self.sim_bs = batch_size - self.real_bs

    def __len__(self):
        return len(self.real) // self.real_bs

    def __iter__(self):
        real_loader = DataLoader(self.real, batch_size=self.real_bs, shuffle=True, num_workers=0, drop_last=True)
        sim_loader = DataLoader(self.sim, batch_size=self.sim_bs, shuffle=True, num_workers=0, drop_last=True)
        sim_iter = iter(sim_loader)
        for real_batch in real_loader:
            try:
                sim_batch = next(sim_iter)
            except StopIteration:
                sim_iter = iter(sim_loader)
                sim_batch = next(sim_iter)
            X_r, z_r, _, _ = real_batch
            X_s, z_s, _, _ = sim_batch
            yield X_r, z_r, X_s, z_s


def train_joint(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading real data...")
    df_real = pd.read_pickle(REAL_DATA_PATH)
    df_real = df_real[df_real['SNR'] >= 2.5].reset_index(drop=True)
    real_train, real_val, real_test = prepare_real(df_real, args.val_split, args.test_split, args.seed)
    print(f"Real: train={len(real_train)} val={len(real_val)} test={len(real_test)}")

    print("Loading sim data...")
    df_sim = pd.read_parquet(SIM_DATA_PATH)
    sim_all = prepare_sim(df_sim)
    sim_train = sim_all.sample(frac=0.9, random_state=args.seed).reset_index(drop=True)
    sim_val = sim_all.drop(sim_train.index).reset_index(drop=True)
    print(f"Sim: train={len(sim_train)} val={len(sim_val)}")

    real_train_ds = HSTGrismDataset(real_train)
    real_val_ds = HSTGrismDataset(real_val)
    real_test_ds = HSTGrismDataset(real_test)
    sim_train_ds = HSTGrismDataset(sim_train)
    sim_val_ds = HSTGrismDataset(sim_val)

    train_loader = JointSimRealLoader(
        real_train_ds, sim_train_ds,
        batch_size=args.batch_size, real_frac=args.real_frac,
    )
    val_loader = DataLoader(real_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(real_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("Loading regridded autoencoder...")
    ckpt = torch.load(REGRID_AE_CKPT, map_location='cpu', weights_only=False)
    sp = SpecPT(input_size=7781, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                dim_feedforward=2048, dropout=0.1)
    sp.load_state_dict(ckpt['model_state_dict'], strict=True)
    model = SimpleRedshiftHead(sp).to(device)
    for p in model.parameters():
        p.requires_grad = True
    print("Backbone UNFROZEN — end-to-end training")

    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {nt:,}/{sum(p.numel() for p in model.parameters()):,}")

    if args.lr_policy == 'flat':
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr, weight_decay=args.weight_decay)
    elif args.lr_policy == 'differential':
        conv_params, enc_params, head_params = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith('head.'):
                head_params.append(p)
            elif 'transformer_encoder' in name:
                enc_params.append(p)
            else:
                conv_params.append(p)
        opt = torch.optim.AdamW([
            {'params': conv_params, 'lr': args.lr},
            {'params': enc_params, 'lr': args.lr},
            {'params': head_params, 'lr': args.lr_head},
        ], weight_decay=args.weight_decay)
    print(f"LR policy: {args.lr_policy} (base_lr={args.lr})")

    criterion = NMADLoss(normalization_factor='std')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.1, patience=args.scheduler_patience, min_lr=1e-7
    )

    model.eval()
    for X, Y, _, _ in val_loader:
        with torch.no_grad():
            pred = model(X.to(device)).flatten()
            print(f"Init val NMADLoss: {criterion(pred, Y.to(device)).item():.4f}")
        break

    best_val_loss = float('inf')
    best_nmad = 1e9
    best_ep = 0
    patience = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_real_acc = 0.0
        loss_sim_acc = 0.0
        loss_recon_acc = 0.0
        n_real = 0
        n_sim = 0

        for X_r, z_r, X_s, z_s in train_loader:
            X_r, z_r = X_r.to(device), z_r.to(device).flatten()
            X_s, z_s = X_s.to(device), z_s.to(device).flatten()

            opt.zero_grad()

            all_X = torch.cat([X_r, X_s])
            all_preds, encoded_all = model(all_X, return_encoded=True)
            pred_r = all_preds[:len(z_r)].flatten()
            pred_s = all_preds[len(z_r):].flatten()

            l_real = args.loss_weight_real * criterion(pred_r, z_r)
            l_sim = args.loss_weight_sim * criterion(pred_s, z_s)

            if args.loss_weight_recon > 0:
                encoded_s = encoded_all[len(z_r):]
                recon_s = model.decode(encoded_s)
                l_recon = args.loss_weight_recon * F.mse_loss(recon_s, X_s)
            else:
                l_recon = torch.tensor(0.0, device=device)

            total = l_real + l_sim + l_recon
            total.backward()
            opt.step()

            loss_real_acc += l_real.item() * X_r.size(0)
            loss_sim_acc += l_sim.item() * X_s.size(0)
            loss_recon_acc += l_recon.item() * X_s.size(0)
            n_real += X_r.size(0)
            n_sim += X_s.size(0)

        loss_real_avg = loss_real_acc / max(n_real, 1)
        loss_sim_avg = loss_sim_acc / max(n_sim, 1)
        loss_recon_avg = loss_recon_acc / max(n_sim, 1)

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X, Y, _, _ in val_loader:
                y = model(X.to(device)).flatten()
                val_preds.append(y.cpu().numpy())
                val_targets.append(Y.numpy())

        pv = np.clip(np.concatenate(val_preds), 0, None)
        tv = np.concatenate(val_targets)
        delz = (pv - tv) / (1 + tv)
        val_nmad = 1.4826 * np.median(np.abs(delz - np.median(delz)))
        val_eta = 100 * np.mean(np.abs(delz) > 0.15)
        cur_lr = opt.param_groups[0]['lr']

        print(f"Ep {ep:2d}/{args.epochs}  real_l={loss_real_avg:.4f}  sim_l={loss_sim_avg:.4f}  "
              f"recon={loss_recon_avg:.6f}  val_nmad={val_nmad:.5f}  eta={val_eta:.2f}%  "
              f"lr={cur_lr:.2e}  {time.time()-t0:.0f}s")
        wandb.log({
            'train/real_nmad_loss': loss_real_avg,
            'train/sim_nmad_loss': loss_sim_avg,
            'train/sim_recon_mse': loss_recon_avg,
            'val/real_nmad': val_nmad,
            'val/real_eta': val_eta,
            'lr': cur_lr,
            'epoch': ep,
        })

        sim_recon_mse = 0.0
        n_sim_val = 0
        with torch.no_grad():
            for X, _, _, _ in DataLoader(sim_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0):
                X = X.to(device)
                recon_val = model.decode(model.encode(X))
                sim_recon_mse += F.mse_loss(recon_val, X).item() * X.size(0)
                n_sim_val += X.size(0)
        if n_sim_val > 0:
            wandb.log({'val/sim_recon_mse': sim_recon_mse / n_sim_val, 'epoch': ep})

        scheduler.step(val_nmad)

        if val_nmad < best_val_loss:
            best_val_loss = val_nmad
            best_nmad = val_nmad
            best_ep = ep
            patience = 0
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': ep, 'model_state_dict': model.state_dict(),
                'val_nmad': val_nmad, 'exp_name': args.exp_name,
            }, f'checkpoints/{args.exp_name}_best_model.pth')
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stop at ep {ep}")
                break

    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for X, Y, _, _ in test_loader:
            y = model(X.to(device)).flatten()
            test_preds.append(y.cpu().numpy())
            test_targets.append(Y.numpy())
    pv = np.clip(np.concatenate(test_preds), 0, None)
    tv = np.concatenate(test_targets)
    delz = (pv - tv) / (1 + tv)
    test_nmad = 1.4826 * np.median(np.abs(delz - np.median(delz)))
    test_eta = 100 * np.mean(np.abs(delz) > 0.15)
    test_rmse = np.sqrt(np.mean((pv - tv) ** 2))
    print(f"\nTest:  NMAD={test_nmad:.5f}  eta={test_eta:.2f}%  RMSE={test_rmse:.4f}")

    counts = len(real_train), len(real_val), len(real_test), len(sim_train), len(sim_val)
    return best_val_loss, best_nmad, best_ep, test_nmad, test_eta, test_rmse, counts


def main():
    parser = argparse.ArgumentParser(description='Joint sim+real fine-tuning')
    parser.add_argument('--exp_name', default='exp_048_joint_sim_real')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_head', type=float, default=3e-4)
    parser.add_argument('--lr_policy', choices=['flat', 'differential'], default='flat')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--real_frac', type=float, default=0.25)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--scheduler_patience', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--loss_weight_real', type=float, default=1.0)
    parser.add_argument('--loss_weight_sim', type=float, default=0.5)
    parser.add_argument('--loss_weight_recon', type=float, default=0.1)
    parser.add_argument('--no_recon', action='store_true', default=False,
                        help='Disable reconstruction loss (γ=0)')
    args = parser.parse_args()
    if args.no_recon:
        args.loss_weight_recon = 0.0
    print(f"=== {args.exp_name}: lr_policy={args.lr_policy} real_frac={args.real_frac} "
          f"α={args.loss_weight_real} β={args.loss_weight_sim} γ={args.loss_weight_recon} ===")
    wandb.init(project='specpt-hst-sim-z', entity='ckb2084-rochester-institute-of-technology',
               name=args.exp_name, config=vars(args))
    result = train_joint(args)
    best_val_loss, best_val_nmad, best_ep, test_nmad, test_eta, test_rmse, counts = result
    real_tr, real_va, real_te, sim_tr, sim_va = counts
    wandb.log({
        'best/val_nmad': best_val_nmad,
        'best/val_loss': best_val_loss,
        'best/epoch': best_ep,
        'test/real_nmad': test_nmad,
        'test/real_eta': test_eta,
        'test/real_rmse': test_rmse,
        'train/real_samples': real_tr,
        'train/sim_samples': sim_tr,
        'val/real_samples': real_va,
        'test/real_samples': real_te,
        'val/sim_samples': sim_va,
    })
    wandb.finish()
    print("DONE")


if __name__ == '__main__':
    main()
