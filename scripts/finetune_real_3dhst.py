#!/usr/bin/env python3
"""Fine-tune exp_032 on real 3D-HST grism data using the proven reproduction pattern."""
import argparse, json, math, os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import yaml
from src.specpt.model import SpecPT, EnhancedSpecPTForRedshift, SpectrumNormalizer
from src.specpt.dataloader import HSTGrismDataset
from src.specpt.training.eval import compute_metrics, compute_per_bin_metrics
from src.specpt.losses import NMADLoss

TRAIN_WAVES = np.linspace(10311.4, 17464.6, 7781)
PADDING_MASK = (TRAIN_WAVES < 11000) | (TRAIN_WAVES > 16500)
BEST_VAL_NMAD = 1e9

def load_fix_data(path, min_snr=2.5):
    df = pd.read_pickle(path)
    df = df[df["SNR"] >= min_snr].reset_index(drop=True)
    safe = df["sensitivity_resampled"].apply(lambda a: np.where(a==0,1e-8,a))
    df["flux_sens"] = df["clean_flux_resampled"] / safe
    specs = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Interpolating"):
        wl = np.asarray(r["wavelength_resampled"], np.float64)
        s = np.asarray(r["flux_sens"], np.float64)
        g = np.interp(TRAIN_WAVES, wl, s, left=np.nan, right=np.nan)
        g[PADDING_MASK] = np.nan
        specs.append(g.astype(np.float32))
    df["spec"] = specs
    df = df.rename(columns={"grism_id":"TARGETID"})
    return df

def split_data(df, tf=0.70, vf=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    ntr = int(len(df)*tf); nva = int(len(df)*vf)
    return (df.iloc[idx[:ntr]].copy(), df.iloc[idx[ntr:ntr+nva]].copy(),
            df.iloc[idx[ntr+nva:]].copy())

def build_model(ckpt_path, ae_path, cfg_path, device):
    with open(cfg_path) as f: cfg = yaml.safe_load(f)["model"]
    ae = torch.load(ae_path, map_location="cpu", weights_only=False)
    sp = SpecPT(input_size=7781, d_model=512, nhead=8, num_encoder_layers=3,
                num_decoder_layers=3, dim_feedforward=2048, dropout=0.1)
    sp.load_state_dict(ae, strict=False)
    model = EnhancedSpecPTForRedshift(sp, num_mlp_blocks=cfg["num_mlp_blocks"],
                                       mlp_dim=cfg["mlp_dim"], dropout_rate=cfg.get("dropout",0.1))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m, u = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if m: print(f"  Missing keys: {len(m)}")
    if u: print(f"  Unexpected keys: {len(u)}")
    return model.to(device)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_p, all_t, all_id = [], [], []
    for X, Y, _, tid in loader:
        X = X.to(device)
        p = model(X).flatten()
        all_p.append(p.cpu().numpy())
        all_t.append(Y.numpy())
        all_id.extend(tid)
    preds = np.clip(np.concatenate(all_p), 0, None)
    trues = np.concatenate(all_t)
    met = compute_metrics(trues, preds)
    met["per_z"] = compute_per_bin_metrics(trues, preds, bins=[0,0.5,1,1.5,2,3])
    return met, (trues, preds, all_id)

def make_plots(t, p, s, per_z, per_s, met, stage):
    dz = (p-t)/(1+t); fig = plt.figure(figsize=(5.7,7))
    gs = GridSpec(2,1,height_ratios=[3,1],hspace=0.01)
    ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1],sharex=ax)
    ml = np.ceil(max(t.max(),p.max()))+0.1
    ax.scatter(t,p,marker=".",s=10,c="b",alpha=0.6); ax.plot([-0.05,ml],[-0.05,ml],"k",zorder=9)
    xl = np.linspace(0,ml)
    for ls in [(xl,xl+(1+xl)*0.15,"k--"),(xl,xl-(1+xl)*0.15,"k--")]:
        ax.plot(ls[0],ls[1],ls[2],zorder=9)
    ax.text(0,ml-0.1,f"NMAD={met['nmad']:.4f}"); ax.set_ylabel("Pred z")
    ax.set_title(f"Stage {stage} 3D-HST Real"); plt.tight_layout()
    plt.close(fig)
    return {"scatter": fig}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1,2])
    ap.add_argument("--init", default="/home/ckb2084/research/specpt-hst-sim/checkpoints/exp_032_best_model.pth")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    args = ap.parse_args()
    stage = args.stage

    s1d = {"ep":30,"lr":3e-4,"wd":1e-3,"pat":5}
    s2d = {"ep":40,"lr":1e-5,"wd":1e-4,"pat":7}
    hc = s2d if stage==2 else s1d
    C = {
        "data_path": "/home/ckb2084/research/SpecPT/data/grism_specPT_v5.pkl",
        "init_ckpt": args.init,
        "ae_path": "/home/ckb2084/research/galax_spec/pretrained_weights/SpecPT_DESI_combined_autoencoder_150_new.pth",
        "model_cfg": str(REPO_ROOT / "configs/exp_032.yaml"),
        "out_dir": "/home/ckb2084/research/specpt-hst-sim/outputs/real_3dhst",
        "ckpt_dir": "/home/ckb2084/research/specpt-hst-sim/checkpoints/finetune_real",
        "wandb_id": "ejfhtjlk",
        "wandb_entity": "ckb2084-rochester-institute-of-technology",
        "wandb_project": "specpt-hst-sim",
        "bs": 128, "nw": 0, "min_snr": 2.5,
        "ep": args.epochs or hc["ep"], "lr": args.lr or hc["lr"],
        "wd": args.wd or hc["wd"], "patience": args.patience or hc["pat"],
    }

    print(f"STAGE {stage}: {'Linear Probe' if stage==1 else 'Partial Freeze'}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_fix_data(C["data_path"], C["min_snr"])
    tr, va, te = split_data(df)
    ds = lambda d: HSTGrismDataset(d, normalize_fn=SpectrumNormalizer.zscore_normalize)
    dl = lambda d,s=False: DataLoader(ds(d), batch_size=C["bs"], shuffle=s, num_workers=0)
    tr_ld = DataLoader(ds(tr), batch_size=C["bs"], shuffle=True, num_workers=0, drop_last=True)
    va_ld = DataLoader(ds(va), batch_size=C["bs"], shuffle=False, num_workers=0)
    te_ld = DataLoader(ds(te), batch_size=C["bs"], shuffle=False, num_workers=0)

    model = build_model(C["init_ckpt"], C["ae_path"], C["model_cfg"], device)

    # Freeze policy
    for p in model.parameters(): p.requires_grad = False
    if stage == 1:
        for p in model.prediction.parameters(): p.requires_grad = True
    elif stage == 2:
        for p in model.encoder.layers[-1].parameters(): p.requires_grad = True
        for p in model.attention.parameters(): p.requires_grad = True
        for p in model.mlp_blocks.parameters(): p.requires_grad = True
        for p in model.prediction.parameters(): p.requires_grad = True
        for p in model.pretrained_model.transformer_decoder.parameters(): p.requires_grad = True

    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {ntr:,} / {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=C["lr"], weight_decay=C["wd"])
    criterion = NMADLoss(normalization_factor="std")

    # Diagnostic
    model.eval()
    for Xd, Yd, _, _ in va_ld:
        with torch.no_grad():
            zd = model(Xd.to(device)).flatten()
        print(f"Diag: loss={criterion(zd, Yd.to(device)).item():.4f}, "
              f"z=[{zd.min().item():.4f},{zd.max().item():.4f}]")
        break

    # Train
    best_ckpt = f"{C['ckpt_dir']}/stage{stage}_best_model.pth"
    os.makedirs(C["ckpt_dir"], exist_ok=True)
    patience = 0

    best_nmad = 1e9
    for ep in range(1, C["ep"]+1):
        model.train()
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
        model.encoder.eval()

        total_loss = 0
        t0 = time.time()
        for X, Y, _, _ in tr_ld:
            X, Y = X.cuda(), Y.cuda()
            opt.zero_grad()
            y = model(X).flatten()
            if torch.isnan(y).any():
                print(f"  NaN! epoch {ep}")
                return
            loss = criterion(y, Y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * X.size(0)
        train_loss = total_loss / len(tr_ld.dataset)

        met, _ = evaluate(model, va_ld, device)
        print(f"Ep {ep:2d}/{C['ep']}  loss={train_loss:.4f}  "
              f"val_nmad={met['nmad']:.5f}  val_eta={met['eta']:.2f}%  "
              f"{time.time()-t0:.0f}s")

        if met["nmad"] < best_nmad:
            best_nmad = met["nmad"]
            torch.save({"epoch":ep,"model_state_dict":model.state_dict(),"best_val_nmad":met["nmad"]}, best_ckpt)
            patience = 0
        else:
            patience += 1
            if patience >= C["patience"]:
                print(f"  Early stop at {ep}")
                break

    print(f"Best val NMAD: {best_nmad:.5f}")

    # Test
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    met, (t, p, ids) = evaluate(model, te_ld, device)
    print(f"Test NMAD={met['nmad']:.5f}  eta={met['eta']:.2f}%  n={len(t)}")

    # Save outputs
    pd.DataFrame({"TARGETID":ids,"z_true":t,"z_pred":p}).to_csv(
        f"{C['ckpt_dir']}/stage{stage}_test_preds.csv", index=False)
    with open(f"{C['ckpt_dir']}/stage{stage}_test_metrics.json","w") as f:
        json.dump(met, f, indent=2, default=str)

    # W&B
    import wandb
    run = wandb.init(id=C["wandb_id"], resume="allow", project=C["wandb_project"], entity=C["wandb_entity"])
    wandb.log({f"real_stage{stage}_nmad":met["nmad"],f"real_stage{stage}_eta":met["eta"],
               f"real_stage{stage}_n":len(t)})
    wandb.finish()
    print("DONE")

if __name__ == "__main__":
    main()
