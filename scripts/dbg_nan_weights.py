#!/usr/bin/env python3
"""Minimal reproduction: load exp_032, forward one batch, check weights."""
import sys, os, torch, numpy as np, pandas as pd
sys.path.insert(0, '.')
import yaml
from src.specpt.model import SpecPT, EnhancedSpecPTForRedshift, SpectrumNormalizer
from src.specpt.dataloader import HSTGrismDataset

TRAIN_WAVES = np.linspace(10311.4, 17464.6, 7781)
PADDING_MASK = (TRAIN_WAVES < 11000) | (TRAIN_WAVES > 16500)

def load_data():
    df = pd.read_pickle('/home/ckb2084/research/SpecPT/data/grism_specPT_v5.pkl')
    df = df[df['SNR'] >= 2.5].reset_index(drop=True)
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
    return df[:256]

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

df = load_data()
ds = HSTGrismDataset(df, normalize_fn=SpectrumNormalizer.zscore_normalize)
dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)

model = build_model().cuda()
model.eval()

# Check weights right after loading
for n, p in model.named_parameters():
    if torch.isnan(p).any():
        print(f'NAN WEIGHT (post-load): {n}')
print('Post-load: all clean?', all(not torch.isnan(p).any() for p in model.parameters()))

# Forward in eval mode
for X, Y, _, _ in dl:
    X, Y = X.cuda(), Y.cuda()
    with torch.no_grad():
        p = model(X).flatten()
    print(f'Eval forward: NaN in pred={torch.isnan(p).sum().item()}/{p.numel()}, range=[{p.min().item():.4f},{p.max().item():.4f}]')
    break

# Check weights after eval forward
for n, p in model.named_parameters():
    if torch.isnan(p).any():
        print(f'NAN WEIGHT (post-eval): {n}')
print('Post-eval: all clean?', all(not torch.isnan(p).any() for p in model.parameters()))

# Switch to train mode
model.train()
for X, Y, _, _ in dl:
    X, Y = X.cuda(), Y.cuda()
    # Check weights right at start of forward
    for n, p in model.named_parameters():
        if torch.isnan(p).any():
            print(f'NAN WEIGHT (pre-train-forward): {n}')
    with torch.no_grad():
        # Run each layer manually, checking weights after each
        x = X.unsqueeze(1)
        x = model.pretrained_model.forward_conv(x)
        for n, p in model.named_parameters():
            if torch.isnan(p).any():
                print(f'NAN WEIGHT (after conv): {n}')
        x = x.flatten(start_dim=1)
        x = model.proj_to_d_model(x)
        for n, p in model.named_parameters():
            if torch.isnan(p).any():
                print(f'NAN WEIGHT (after proj): {n}')
        x = x.unsqueeze(0)
        x = model.encoder(x)
        for n, p in model.named_parameters():
            if torch.isnan(p).any():
                print(f'NAN WEIGHT (after encoder): {n}')
        x = x.squeeze(0)
        attn, _ = model.attention(x, x, x)
        for n, p in model.named_parameters():
            if torch.isnan(p).any():
                print(f'NAN WEIGHT (after attention): {n}')
        x = attn + x
        x = model.mlp_blocks(x)
        for n, p in model.named_parameters():
            if torch.isnan(p).any():
                print(f'NAN WEIGHT (after mlp): {n}')
        p = model.prediction(x).flatten()
        print(f'Train forward: NaN in pred={torch.isnan(p).sum().item()}/{p.numel()}')
    break
