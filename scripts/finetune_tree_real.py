#!/usr/bin/env python3
"""Random Forest on frozen regridded autoencoder latents + real 3D-HST.

Extracts 512-d frozen encoder features once, caches them, then fits
RandomForestRegressor. No hyperparameter tuning (sklearn defaults).
"""
import sys, os, numpy as np, pandas as pd, torch, time, argparse, wandb, json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

sys.path.insert(0, '.')
from src.specpt.model import SpecPT
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
CACHE_DIR = 'outputs/tree_features'


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


def extract_latents(model, loader, device):
    model.eval()
    all_z, all_y = [], []
    with torch.no_grad():
        for X, Y, _, _ in loader:
            x = X.to(device).unsqueeze(1)
            x = model.forward_conv(x)
            x = x.flatten(start_dim=1)
            x = model.proj_to_d_model(x)
            x = x.unsqueeze(0)
            x = model.transformer_encoder(x)
            x = x.squeeze(0)
            all_z.append(x.cpu().numpy())
            all_y.append(Y.numpy())
    return np.concatenate(all_z).astype(np.float32), np.concatenate(all_y)


def compute_metrics(pv, tv):
    delz = (pv - tv) / (1 + tv)
    nmad = float(1.4826 * np.median(np.abs(delz - np.median(delz))))
    eta = float(100 * np.mean(np.abs(delz) > 0.15))
    rmse = float(np.sqrt(np.mean((pv - tv) ** 2)))
    return nmad, eta, rmse


def main():
    parser = argparse.ArgumentParser(description='Random Forest on frozen encoder latents')
    parser.add_argument('--exp_name', default='exp_044')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--recompute', action='store_true', default=False,
                        help='Recompute cached latents')
    parser.add_argument('--n_jobs', type=int, default=8,
                        help='Parallel jobs for RF fitting')
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"=== {args.exp_name}: RF on frozen encoder latents ===")

    df = pd.read_pickle(REAL_DATA_PATH)
    df = df[df['SNR'] >= 2.5].reset_index(drop=True)
    print(f"Loaded {len(df)} real 3D-HST spectra (SNR >= 2.5)")

    train, val, test = prepare_data(df, args.val_split, args.test_split, args.seed)
    print(f"  Train={len(train)}  Val={len(val)}  Test={len(test)}")

    cache_ok = all(os.path.exists(os.path.join(CACHE_DIR, f'{split}_latent.npy'))
                   for split in ['train', 'val', 'test'])
    if cache_ok and not args.recompute:
        print("Loading cached latents...")
        z_train = np.load(os.path.join(CACHE_DIR, 'train_latent.npy'))
        y_train = np.load(os.path.join(CACHE_DIR, 'train_z.npy'))
        z_val   = np.load(os.path.join(CACHE_DIR, 'val_latent.npy'))
        y_val   = np.load(os.path.join(CACHE_DIR, 'val_z.npy'))
        z_test  = np.load(os.path.join(CACHE_DIR, 'test_latent.npy'))
        y_test  = np.load(os.path.join(CACHE_DIR, 'test_z.npy'))
    else:
        print("Loading regridded autoencoder...")
        ckpt = torch.load(REGRID_AE_CKPT, map_location='cpu', weights_only=False)
        sp = SpecPT(input_size=7781, d_model=512, nhead=8, num_encoder_layers=3,
                    num_decoder_layers=3, dim_feedforward=2048, dropout=0.1)
        sp.load_state_dict(ckpt['model_state_dict'], strict=True)
        for p in sp.parameters():
            p.requires_grad = False
        sp.to(device)
        print("Extracting latents (encoder only, no head)...")
        t0 = time.time()
        for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
            ld = DataLoader(HSTGrismDataset(split_data), 64, shuffle=False, num_workers=0)
            z, y = extract_latents(sp, ld, device)
            np.save(os.path.join(CACHE_DIR, f'{split_name}_latent.npy'), z)
            np.save(os.path.join(CACHE_DIR, f'{split_name}_z.npy'), y)
            print(f"  {split_name}: {z.shape}  ({time.time()-t0:.0f}s)")
        z_train = np.load(os.path.join(CACHE_DIR, 'train_latent.npy'))
        y_train = np.load(os.path.join(CACHE_DIR, 'train_z.npy'))
        z_val   = np.load(os.path.join(CACHE_DIR, 'val_latent.npy'))
        y_val   = np.load(os.path.join(CACHE_DIR, 'val_z.npy'))
        z_test  = np.load(os.path.join(CACHE_DIR, 'test_latent.npy'))
        y_test  = np.load(os.path.join(CACHE_DIR, 'test_z.npy'))

    print(f"\nLatent dim: {z_train.shape[1]},  Train samples: {z_train.shape[0]}")
    print(f"z range: [{y_train.min():.3f}, {y_train.max():.3f}]")

    wandb.init(project='specpt-hst-sim-z',
               entity='ckb2084-rochester-institute-of-technology',
               name=args.exp_name,
               config={
                   'model': 'RandomForestRegressor',
                   'n_estimators': args.n_estimators,
                   'n_jobs': args.n_jobs,
                   'feature_dim': z_train.shape[1],
                   'freeze_backbone': True,
                   'seed': args.seed,
                   'val_split': args.val_split,
                   'test_split': args.test_split,
                   'train_samples': len(train),
                   'val_samples': len(val),
                   'test_samples': len(test),
                   'regrid_ae_ckpt': os.path.basename(REGRID_AE_CKPT),
               })

    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    rf.fit(z_train, y_train)
    fit_time = time.time() - t0
    print(f"RF fit: {fit_time:.1f}s  n_estimators={args.n_estimators}")

    pv_val = rf.predict(z_val)
    pv_test = rf.predict(z_test)
    pv_train = rf.predict(z_train)

    val_nmad, val_eta, val_rmse = compute_metrics(pv_val, y_val)
    test_nmad, test_eta, test_rmse = compute_metrics(pv_test, y_test)
    train_nmad, train_eta, train_rmse = compute_metrics(pv_train, y_train)

    print(f"\nTrain: NMAD={train_nmad:.5f}  η={train_eta:.2f}%  RMSE={train_rmse:.4f}")
    print(f"Val:   NMAD={val_nmad:.5f}  η={val_eta:.2f}%  RMSE={val_rmse:.4f}")
    print(f"Test:  NMAD={test_nmad:.5f}  η={test_eta:.2f}%  RMSE={test_rmse:.4f}")
    print(f"R2 Test: {r2_score(y_test, pv_test):.4f}")

    wandb.log({
        'val_nmad': val_nmad,
        'val_eta': val_eta,
        'val_rmse': val_rmse,
        'test_nmad': test_nmad,
        'test_eta': test_eta,
        'test_rmse': test_rmse,
        'train_nmad': train_nmad,
        'train_r2': r2_score(y_train, pv_train),
        'test_r2': r2_score(y_test, pv_test),
        'fit_time_sec': fit_time,
        'n_features': z_train.shape[1],
    })

    preds = np.stack([y_test, pv_test]).T
    preds_path = os.path.join(CACHE_DIR, f'{args.exp_name}_test_preds.npy')
    np.save(preds_path, preds)
    wandb.save(preds_path)

    wandb.log({'best_val_nmad': val_nmad})
    wandb.finish()
    print("DONE")


if __name__ == '__main__':
    main()
