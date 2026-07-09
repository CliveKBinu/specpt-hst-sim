#!/usr/bin/env python3
"""Stage 1 linear probe — absolutely minimal."""
import sys, os, numpy as np, pandas as pd, torch, torch.nn as nn, time
sys.path.insert(0, '.')
import yaml
from src.specpt.model import SpecPT, EnhancedSpecPTForRedshift, SpectrumNormalizer
from src.specpt.dataloader import HSTGrismDataset
from src.specpt.losses import NMADLoss
from torch.utils.data import DataLoader

TRAIN_WAVES = np.linspace(10311.4, 17464.6, 7781)
PAD = (TRAIN_WAVES < 11000) | (TRAIN_WAVES > 16500)

print("=== STAGE 1 LINEAR PROBE ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Data
df = pd.read_pickle('/home/ckb2084/research/SpecPT/data/grism_specPT_v5.pkl')
df = df[df['SNR'] >= 2.5].reset_index(drop=True)
safe = df['sensitivity_resampled'].apply(lambda a: np.where(a==0,1e-8,a))
df['fs'] = df['clean_flux_resampled'] / safe
specs = []
for _, r in df.iterrows():
    g = np.interp(TRAIN_WAVES, np.asarray(r['wavelength_resampled'],np.float64),
                  np.asarray(r['fs'],np.float64), left=np.nan, right=np.nan)
    g[PAD] = np.nan; specs.append(g.astype(np.float32))
df['spec'] = specs; df = df.rename(columns={'grism_id':'TARGETID'})
print(f"Spectra: {len(df)}")

# Split
rng = np.random.default_rng(42)
idx = rng.permutation(len(df))
tr, va, te = df.iloc[idx[:int(len(df)*0.70)]].copy(), df.iloc[idx[int(len(df)*0.70):int(len(df)*0.85)]].copy(), df.iloc[idx[int(len(df)*0.85):]].copy()
print(f"Train {len(tr)} Val {len(va)} Test {len(te)}")
ds = lambda d: HSTGrismDataset(d, normalize_fn=SpectrumNormalizer.zscore_normalize)
tr_ld = DataLoader(ds(tr), 128, shuffle=True, num_workers=0, drop_last=True)
va_ld = DataLoader(ds(va), 128, shuffle=False, num_workers=0)
te_ld = DataLoader(ds(te), 128, shuffle=False, num_workers=0)

# Model
print("Loading model...")
with open('configs/exp_032.yaml') as f: cfg = yaml.safe_load(f)['model']
ae = torch.load('/home/ckb2084/research/galax_spec/pretrained_weights/SpecPT_DESI_combined_autoencoder_150_new.pth', map_location='cpu', weights_only=False)
sp = SpecPT(input_size=7781, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048, dropout=0.1)
sp.load_state_dict(ae, strict=False)
model = EnhancedSpecPTForRedshift(sp, num_mlp_blocks=cfg['num_mlp_blocks'], mlp_dim=cfg['mlp_dim'], dropout_rate=cfg.get('dropout',0.1))
ckpt = torch.load('/home/ckb2084/research/specpt-hst-sim/checkpoints/exp_032_best_model.pth', map_location='cpu', weights_only=False)
m, u = model.load_state_dict(ckpt['model_state_dict'], strict=False)
print(f"Missing: {len(m)}, Unexpected: {len(u)}")
model = model.to(device)

# Freeze
for p in model.parameters(): p.requires_grad = False
for p in model.prediction.parameters(): p.requires_grad = True
nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {nt:,}/{sum(p.numel() for p in model.parameters()):,}")

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-3)
criterion = NMADLoss(normalization_factor='std')

# Diag
model.eval()
for Xd, Yd, _, _ in va_ld:
    with torch.no_grad():
        zd = model(Xd.to(device)).flatten()
    print(f"Diag: loss={criterion(zd, Yd.to(device)).item():.4f}")
    break

# Train
best_nmad = 1e9; patience = 0
for ep in range(1, 31):
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
    model.encoder.eval()
    t0 = time.time(); tl = 0.0
    for b, (X, Y, _, _) in enumerate(tr_ld):
        X, Y = X.cuda(), Y.cuda()
        opt.zero_grad()
        y = model(X).flatten()
        if torch.isnan(y).any():
            print(f"  NaN! ep={ep} batch={b}")
            raise SystemExit(1)
        if torch.isnan(Y).any():
            print(f"  Y has NaN! ep={ep} batch={b}")
            raise SystemExit(1)
        loss = criterion(y, Y)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  loss={loss.item()} ep={ep} batch={b}  y=[{y.min().item():.4f},{y.max().item():.4f}]  "
                  f"Y=[{Y.min().item():.4f},{Y.max().item():.4f}]  "
                  f"std_y={torch.std(Y).item():.6f}")
            raise SystemExit(1)
        loss.backward()
        opt.step()
        tl += loss.item() * X.size(0)
    tl /= len(tr_ld.dataset)

    model.eval()
    with torch.no_grad():
        ap, at = [], []
        for X, Y, _, _ in va_ld:
            y = model(X.to(device)).flatten()
            ap.append(y.cpu().numpy()); at.append(Y.numpy())
    pv = np.clip(np.concatenate(ap), 0, None); tv = np.concatenate(at)
    delz = (pv-tv)/(1+tv)
    nmad = 1.4826 * np.median(np.abs(delz - np.median(delz)))
    eta = 100 * np.mean(np.abs(delz) > 0.15)
    print(f"Ep {ep:2d}/30  loss={tl:.4f}  val_nmad={nmad:.5f}  eta={eta:.2f}%  {time.time()-t0:.0f}s")

    if nmad < best_nmad:
        best_nmad = nmad
        os.makedirs('checkpoints/finetune_real', exist_ok=True)
        torch.save({'epoch':ep,'model_state_dict':model.state_dict(),'nmad':nmad},
                   f'checkpoints/finetune_real/stage1_best.pth')
        patience = 0
    else:
        patience += 1
        if patience >= 5:
            print(f"Early stop at {ep}")
            break

print(f"Best NMAD: {best_nmad:.5f}")
print("DONE")
