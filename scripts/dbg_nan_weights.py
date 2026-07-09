#!/usr/bin/env python3
"""Minimal reproduction: load exp_032, forward one batch, check weights."""
import sys, os, torch, numpy as np, pandas as pd
sys.path.insert(0, '.')
import yaml
from src.specpt.model import SpecPT, EnhancedSpecPTForRedshift, SpectrumNormalizer
from src.specpt.dataloader import HSTGrismDataset
from src.specpt.losses import NMADLoss

TRAIN_WAVES = np.linspace(10311.4, 17464.6, 7781)
PADDING_MASK = (TRAIN_WAVES < 11000) | (TRAIN_WAVES > 16500)

def load_data():
    df = pd.read_pickle('/home/ckb2084/research/SpecPT/data/grism_specPT_v5.pkl')
    df = df[df['SNR'] >= 2.5].reset_index(drop=True)
    df = df.iloc[:256].copy()
    safe_sens = df['sensitivity_resampled'].apply(lambda a: np.where(a==0,1e-8,a))
    df['flux_sensitivity'] = df['clean_flux_resampled'] / safe_sens
    specs = []
    for _, row in df.iterrows():
        ref = np.asarray(row['wavelength_resampled'], np.float64)
        s = np.asarray(row['flux_sensitivity'], np.float64)
        g = np.interp(TRAIN_WAVES, ref, s, left=np.nan, right=np.nan)
        g[PADDING_MASK] = np.nan
        specs.append(g.astype(np.float32))
    df['spec'] = specs
    df = df.rename(columns={'grism_id':'TARGETID'})
    return df

def build_model():
    with open('configs/exp_032.yaml') as f:
        mc = yaml.safe_load(f)['model']
    ae = torch.load('/home/ckb2084/research/galax_spec/pretrained_weights/SpecPT_DESI_combined_autoencoder_150_new.pth', map_location='cpu', weights_only=False)
    sp = SpecPT(input_size=7781, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048, dropout=0.1)
    sp.load_state_dict(ae, strict=False)
    model = EnhancedSpecPTForRedshift(sp, num_mlp_blocks=mc['num_mlp_blocks'], mlp_dim=mc['mlp_dim'], dropout_rate=mc.get('dropout',0.1))
    ckpt = torch.load('checkpoints/exp_032_best_model.pth', map_location='cpu', weights_only=False)
    m, u = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if m: print(f'Missing keys: {len(m)}')
    if u: print(f'Unexpected keys: {len(u)}')
    return model

def check_nan(msg):
    bad = [n for n,p in model.named_parameters() if torch.isnan(p).any()]
    if bad: print(f'  NAN: {msg}: {bad[:5]}')
    else: print(f'  CLEAN: {msg}')
    return len(bad) == 0

df = load_data()
ds = HSTGrismDataset(df, normalize_fn=SpectrumNormalizer.zscore_normalize)
dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)

model = build_model().cuda()
criterion = NMADLoss(normalization_factor='std')

# Phase 1: eval diagnostic (exact same as fine-tune script)
model.eval()
X_diag, Y_diag = next(iter(dl))
X_diag = X_diag.cuda()
Y_diag = Y_diag.cuda()
with torch.no_grad():
    x = X_diag.unsqueeze(1)
    x = model.pretrained_model.forward_conv(x)
    x = x.flatten(start_dim=1)
    x = model.proj_to_d_model(x)
    x = x.unsqueeze(0)
    x = model.encoder(x)
    x = x.squeeze(0)
    attn, _ = model.attention(x, x, x)
    x = attn + x
    x = model.mlp_blocks(x)
    z_pred = model.prediction(x)
loss_diag = criterion(z_pred.flatten(), Y_diag)
print(f'Diag loss={loss_diag.item():.4f}')
check_nan('after diag loss')

# Phase 2: model.train()
model.train()
check_nan('after model.train()')

# Phase 3: freeze BN
for m in model.modules():
    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
        m.eval()
check_nan('after BN freeze')

# Phase 4: train forward
for X_train, Y_train in dl:
    X_train = X_train.cuda()
    Y_train = Y_train.cuda()
    check_nan('before forward')
    with torch.no_grad():
        # Exact same manual forward as train epoch
        x = X_train.unsqueeze(1)
        x = model.pretrained_model.forward_conv(x)
        check_nan('after conv')
        x = x.flatten(start_dim=1)
        x = model.proj_to_d_model(x)
        check_nan('after proj')
        x = x.unsqueeze(0)
        x = model.encoder(x)
        check_nan('after encoder')
        x = x.squeeze(0)
        attn, _ = model.attention(x, x, x)
        check_nan('after attention')
        x = attn + x
        x = model.mlp_blocks(x)
        check_nan('after mlp')
        y_pred = model.prediction(x).flatten()
        check_nan('after pred')
        print(f'  pred range=[{y_pred.min().item():.4f},{y_pred.max().item():.4f}]')
    break

# Phase 5: also test how _apply_freeze_policy changes things
model2 = build_model().cuda()
check_nan('model2 after load')
for p in model2.parameters(): p.requires_grad = False
for p in model2.prediction.parameters(): p.requires_grad = True
check_nan('model2 after freeze policy')
model2.train()
for m in model2.modules():
    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
        m.eval()
# Forward
X_test, Y_test = next(iter(dl))
X_test = X_test.cuda()
with torch.no_grad():
    p = model2(X_test).flatten()
check_nan('model2 after frozen train forward')
print(f'  model2 pred range=[{p.min().item():.4f},{p.max().item():.4f}]')
