#!/usr/bin/env python3
"""B1+B3: RF on PRE-attention frozen latents + SNR-bucketed NMAD.

Extracts 512-d latents from proj_to_d_model (BEFORE transformer_encoder),
fits RandomForestRegressor, and logs per-SNR-bucket metrics (B3).

Decision: if pre-attention NMAD ~ post-attention NMAD (exp_045, 0.2077),
the 3-layer MHA is decorative on real data.
"""
import sys, os, numpy as np, pandas as pd, torch, time, argparse, wandb
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
CACHE_DIR = 'outputs/tree_features/pre_attn'


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


def extract_pre_attn_latents(model, loader, device):
    model.eval()
    all_z, all_y = [], []
    with torch.no_grad():
        for X, Y, _, _ in loader:
            x = X.to(device).unsqueeze(1)
            x = model.forward_conv(x)
            x = x.flatten(start_dim=1)
            x = model.proj_to_d_model(x)
            all_z.append(x.cpu().numpy())
            all_y.append(Y.numpy())
    return np.concatenate(all_z).astype(np.float32), np.concatenate(all_y).ravel()


def compute_metrics(pv, tv):
    pv = np.asarray(pv).ravel()
    tv = np.asarray(tv).ravel()
    delz = (pv - tv) / (1 + tv)
    nmad = float(1.4826 * np.median(np.abs(delz - np.median(delz))))
    eta = float(100 * np.mean(np.abs(delz) > 0.15))
    rmse = float(np.sqrt(np.mean((pv - tv) ** 2)))
    return nmad, eta, rmse


def snr_bucket_metrics(pv, tv, snr_values, bins=None, labels=None):
    if bins is None:
        bins = [0, 5, 10, 20, np.inf]
    if labels is None:
        labels = ['2.5-5', '5-10', '10-20', '20+']
    pv, tv = np.asarray(pv).ravel(), np.asarray(tv).ravel()
    snr = np.asarray(snr_values).ravel()
    bucket_ids = np.digitize(snr, bins)
    results = {}
    for bid in range(1, len(bins)):
        mask = bucket_ids == bid
        if mask.sum() == 0:
            continue
        nmad, eta, rmse = compute_metrics(pv[mask], tv[mask])
        results[labels[bid - 1]] = {
            'n': int(mask.sum()), 'nmad': nmad, 'eta': eta, 'rmse': rmse,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description='Pre-attention RF + SNR analysis')
    parser.add_argument('--exp_name', default='exp_046_pre_attn_RF')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--recompute', action='store_true', help='Recompute cached latents')
    parser.add_argument('--n_jobs', type=int, default=8, help='RF parallel jobs')
    args = parser.parse_args()
    skip_extraction = os.environ.get('SKIP_EXTRACTION', '').lower() in ('1', 'true', 'yes')

    os.makedirs(CACHE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"=== {args.exp_name}: Pre-attention RF + SNR analysis ===")

    df = pd.read_pickle(REAL_DATA_PATH)
    df = df[df['SNR'] >= 2.5].reset_index(drop=True)
    print(f"Loaded {len(df)} real 3D-HST spectra (SNR >= 2.5)")

    train, val, test = prepare_data(df, args.val_split, args.test_split, args.seed)
    print(f"  Train={len(train)}  Val={len(val)}  Test={len(test)}")

    train_snr, val_snr, test_snr = [d['SNR'].values.ravel() for d in (train, val, test)]

    cache_ok = all(os.path.exists(os.path.join(CACHE_DIR, f'{split}_latent.npy'))
                   for split in ['train', 'val', 'test'])
    if (cache_ok and not args.recompute) or skip_extraction:
        print("Loading cached pre-attention latents...")
        z_train = np.load(os.path.join(CACHE_DIR, 'train_latent.npy'))
        z_val   = np.load(os.path.join(CACHE_DIR, 'val_latent.npy'))
        z_test  = np.load(os.path.join(CACHE_DIR, 'test_latent.npy'))
        y_train = np.load(os.path.join(CACHE_DIR, 'train_z.npy')).ravel()
        y_val   = np.load(os.path.join(CACHE_DIR, 'val_z.npy')).ravel()
        y_test  = np.load(os.path.join(CACHE_DIR, 'test_z.npy')).ravel()
    else:
        print("Loading regridded autoencoder (frozen)...")
        ckpt = torch.load(REGRID_AE_CKPT, map_location='cpu', weights_only=False)
        sp = SpecPT(input_size=7781, d_model=512, nhead=8, num_encoder_layers=3,
                    num_decoder_layers=3, dim_feedforward=2048, dropout=0.1)
        sp.load_state_dict(ckpt['model_state_dict'], strict=True)
        for p in sp.parameters():
            p.requires_grad = False
        sp.to(device)
        print("Extracting pre-attention latents (forward_conv → proj_to_d_model only)...")
        t0 = time.time()
        for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
            ld = DataLoader(HSTGrismDataset(split_data), 64, shuffle=False, num_workers=0)
            z, y = extract_pre_attn_latents(sp, ld, device)
            np.save(os.path.join(CACHE_DIR, f'{split_name}_latent.npy'), z)
            np.save(os.path.join(CACHE_DIR, f'{split_name}_z.npy'), y)
            print(f"  {split_name}: {z.shape}  ({time.time()-t0:.0f}s)")
        z_train = np.load(os.path.join(CACHE_DIR, 'train_latent.npy'))
        z_val   = np.load(os.path.join(CACHE_DIR, 'val_latent.npy'))
        z_test  = np.load(os.path.join(CACHE_DIR, 'test_latent.npy'))
        y_train = np.load(os.path.join(CACHE_DIR, 'train_z.npy')).ravel()
        y_val   = np.load(os.path.join(CACHE_DIR, 'val_z.npy')).ravel()
        y_test  = np.load(os.path.join(CACHE_DIR, 'test_z.npy')).ravel()

    print(f"\nLatent dim: {z_train.shape[1]},  Train: {z_train.shape[0]}")
    print(f"z range: [{y_train.min():.3f}, {y_train.max():.3f}]")

    wandb.init(project='specpt-hst-sim-z',
               entity='ckb2084-rochester-institute-of-technology',
               name=args.exp_name,
               config={
                   'model': 'PreAttnRandomForest',
                   'tap_point': 'pre_attention',
                   'n_estimators': args.n_estimators,
                   'n_jobs': args.n_jobs,
                   'feature_dim': z_train.shape[1],
                   'freeze_backbone': True,
                   'train_samples': len(train),
                   'val_samples': len(val),
                   'test_samples': len(test),
               })

    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=args.n_estimators,
                                n_jobs=args.n_jobs, random_state=args.seed)
    rf.fit(z_train, y_train)
    fit_time = time.time() - t0
    print(f"RF fit: {fit_time:.1f}s  n_estimators={args.n_estimators}")

    pv_val, pv_test, pv_train = rf.predict(z_val), rf.predict(z_test), rf.predict(z_train)

    val_nmad, val_eta, val_rmse = compute_metrics(pv_val, y_val)
    test_nmad, test_eta, test_rmse = compute_metrics(pv_test, y_test)
    train_nmad, train_eta, train_rmse = compute_metrics(pv_train, y_train)

    print(f"\nTrain: NMAD={train_nmad:.5f}  η={train_eta:.2f}%  RMSE={train_rmse:.4f}")
    print(f"Val:   NMAD={val_nmad:.5f}  η={val_eta:.2f}%  RMSE={val_rmse:.4f}")
    print(f"Test:  NMAD={test_nmad:.5f}  η={test_eta:.2f}%  RMSE={test_rmse:.4f}")
    print(f"R2 Test: {r2_score(y_test, pv_test):.4f}")

    wandb.log({
        'val_nmad': val_nmad, 'val_eta': val_eta, 'val_rmse': val_rmse,
        'test_nmad': test_nmad, 'test_eta': test_eta, 'test_rmse': test_rmse,
        'train_nmad': train_nmad, 'train_eta': train_eta, 'train_rmse': train_rmse,
        'train_r2': r2_score(y_train, pv_train),
        'test_r2': r2_score(y_test, pv_test),
        'fit_time_sec': fit_time, 'n_features': z_train.shape[1],
    })

    # B3: SNR-bucketed NMAD
    for split_name, pv, tv, snr in [
        ('train', pv_train, y_train, train_snr),
        ('val',   pv_val,   y_val,   val_snr),
        ('test',  pv_test,  y_test,  test_snr),
    ]:
        buckets = snr_bucket_metrics(pv, tv, snr)
        for label, br in buckets.items():
            key = f'{split_name}_snr_{label}'
            print(f"  {key}: N={br['n']} NMAD={br['nmad']:.5f} η={br['eta']:.2f}%")
            wandb.log({
                f'{key}_n': br['n'], f'{key}_nmad': br['nmad'],
                f'{key}_eta': br['eta'], f'{key}_rmse': br['rmse'],
            })

    preds = np.column_stack([pv_test, y_test])
    preds_path = os.path.join(CACHE_DIR, f'{args.exp_name}_test_preds.npy')
    np.save(preds_path, preds)
    wandb.save(preds_path)

    wandb.log({'best_val_nmad': val_nmad})
    wandb.finish()
    print("DONE")


if __name__ == '__main__':
    main()
