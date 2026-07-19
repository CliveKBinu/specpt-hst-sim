#!/usr/bin/env python3
"""Phase 1: Train tiny MLP/ResNet on frozen exp_032 encoder features (real 3D-HST)."""

import argparse, json, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import yaml
from src.specpt.model import SpecPT, EnhancedSpecPTForRedshift, SpectrumNormalizer
from src.specpt.losses import NMADLoss

TRAIN_WAVES = np.linspace(10311.4, 17464.6, 7781)
PADDING_MASK = (TRAIN_WAVES < 11000) | (TRAIN_WAVES > 16500)
SEED = 42
NORMALIZER = SpectrumNormalizer(method="median")


class TinyMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=1, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h = self.dropout(torch.relu(self.fc1(x)))
        h = self.dropout(self.fc2(h))
        return torch.relu(x + h)


class TinyResNet(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=1, n_blocks=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = torch.relu(self.proj(x))
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


MODELS = {"mlp": TinyMLP, "resnet": TinyResNet}

# --- Per-bin helpers (self-contained, no specpt.training.eval dependency) ---
def _compute_metrics(t, p):
    dz = np.abs(p - t) / (1 + t)
    nmad = float(np.median(dz))
    eta = float((dz > 0.15).mean() * 100)
    bias = float(np.median(p - t))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    return {"nmad": nmad, "eta": eta, "bias": bias, "rmse": rmse}


def _per_bin(t, p, bins, values=None, label=None):
    results = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if values is not None:
            m = (values >= lo) & (values < hi)
        else:
            m = (t >= lo) & (t < hi)
        n = int(m.sum())
        if n < 2:
            results.append({"bin": f"[{lo},{hi})", "count": n, "nmad": 0, "eta": 0})
            continue
        mt, mp = t[m], p[m]
        dz = np.abs(mp - mt) / (1 + mt)
        results.append({"bin": f"[{lo},{hi})", "count": n,
                        "nmad": float(np.median(dz)),
                        "eta": float((dz > 0.15).mean() * 100)})
    return results


def load_fix_data(path, min_snr=2.5):
    df = pd.read_pickle(path)
    df = df[df.SNR >= min_snr].reset_index(drop=True)
    safe = df["sensitivity_resampled"].apply(lambda a: np.where(a==0, 1e-8, a))
    df["flux_sens"] = df["clean_flux_resampled"] / safe
    specs = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Interpolating"):
        wl = np.asarray(r["wavelength_resampled"], np.float64)
        s = np.asarray(r["flux_sens"], np.float64)
        g = np.interp(TRAIN_WAVES, wl, s, left=np.nan, right=np.nan)
        g[PADDING_MASK] = np.nan
        specs.append(g)
    return df, specs


def build_model(ckpt_path, ae_path, model_cfg_path, device):
    with open(model_cfg_path) as f:
        cfg = yaml.safe_load(f)["model"]
    ae_state = torch.load(ae_path, map_location="cpu", weights_only=False)
    specpt = SpecPT(cfg["input_size"], cfg["d_model"], cfg["nhead"],
                    cfg["num_encoder_layers"], cfg["num_decoder_layers"],
                    cfg["dim_feedforward"], cfg["dropout"])
    specpt.load_state_dict(ae_state, strict=False)
    model = EnhancedSpecPTForRedshift(
        pretrained_model=specpt, d_model=cfg["d_model"],
        nhead=cfg["nhead"], num_mlp_blocks=cfg.get("num_mlp_blocks", 12),
        mlp_dim=cfg.get("mlp_dim", 1024), dropout_rate=cfg.get("dropout", 0.1))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_state = ckpt["model_state_dict"]
    # Strip prefix if needed
    cleaned = {}
    for k, v in ckpt_state.items():
        if k.startswith("pretrained_model."):
            cleaned[k] = v
        else:
            cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model.to(device)


def extract(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading exp_032 checkpoint...")
    model = build_model(args.init_ckpt, args.ae_path, args.model_cfg, device)
    print(f"Loading real 3D-HST data...")
    df, raw_specs = load_fix_data(args.real_data, args.min_snr)
    z_vals = df.z.values.astype(np.float32)
    ids = df.grism_id if "grism_id" in df.columns else df.TARGETID.values
    snr_vals = df.SNR.values.astype(np.float32)
    n = len(df)
    print(f"  {n} spectra loaded")

    # Split (same as Stage 1/2)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_idx, val_idx, test_idx = perm[:n_train], perm[n_train:n_train+n_val], perm[n_train+n_val:]
    print(f"  Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    def _idx_to_data(spec_list, idx):
        return [spec_list[i] for i in idx], z_vals[idx], ids[idx] if isinstance(ids, np.ndarray) else [ids[i] for i in idx], snr_vals[idx]

    train_data = _idx_to_data(raw_specs, train_idx)
    val_data = _idx_to_data(raw_specs, val_idx)
    test_data = _idx_to_data(raw_specs, test_idx)

    class SpecDataset(torch.utils.data.Dataset):
        def __init__(self, specs_arr, z_arr, id_arr, snr_arr):
            self.specs = specs_arr
            self.z = z_arr
            self.ids = id_arr
            self.snr = snr_arr
        def __len__(self):
            return len(self.z)
        def __getitem__(self, i):
            s = NORMALIZER(self.specs[i]).astype(np.float32)
            return torch.from_numpy(s), torch.tensor(self.z[i], dtype=torch.float32), str(self.ids[i])

    train_ds = SpecDataset(*train_data)
    val_ds = SpecDataset(*val_data)
    test_ds = SpecDataset(*test_data)

    def _extract_features(dataset, name):
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
        feats, zs, ids_list = [], [], []
        with torch.no_grad():
            for x, y, sid in tqdm(loader, desc=f"Extracting {name}"):
                x = x.to(device)
                h = x.unsqueeze(1)
                h = model.pretrained_model.forward_conv(h)
                h = h.flatten(start_dim=1)
                h = model.proj_to_d_model(h)
                h = h.unsqueeze(0)
                h = model.encoder(h)
                h = h.squeeze(0)
                feats.append(h.cpu().numpy())
                zs.append(y.cpu().numpy())
                ids_list.extend(sid)
        return np.concatenate(feats).astype(np.float32), np.concatenate(zs), np.array(ids_list)

    print("Extracting features...")
    train_f, train_z, train_ids = _extract_features(train_ds, "train")
    val_f, val_z, val_ids = _extract_features(val_ds, "val")
    test_f, test_z, test_ids = _extract_features(test_ds, "test")

    # Save cache with SNR
    out = {"train_features": train_f, "train_z": train_z, "train_ids": train_ids,
           "train_snr": train_data[3].astype(np.float32),
           "val_features": val_f, "val_z": val_z, "val_ids": val_ids,
           "val_snr": val_data[3].astype(np.float32),
           "test_features": test_f, "test_z": test_z, "test_ids": test_ids,
           "test_snr": test_data[3].astype(np.float32)}
    os.makedirs(args.ckpt_dir, exist_ok=True)
    np.savez_compressed(args.cache_path, **out)
    print(f"Features cached: {args.cache_path}")
    return out


def load_cache(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def train_model(model_cls, feats, z, val_feats, val_z, cfg, device, model_type):
    model = model_cls().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=0)

    train_t = torch.from_numpy(feats).to(device)
    train_tz = torch.from_numpy(z).to(device)
    val_t = torch.from_numpy(val_feats).to(device)
    val_tz = torch.from_numpy(val_z).to(device)
    bs = cfg["batch_size"]
    best_nmad = 1e9
    patience = 0
    history = {"train_loss": [], "val_nmad": [], "lr": []}
    metric_fn = NMADLoss(normalization_factor="std")

    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(train_t))
        total_loss = 0
        for i in range(0, len(train_t), bs):
            idx = perm[i:i + bs]
            xb, yb = train_t[idx], train_tz[idx]
            optimizer.zero_grad()
            yp = model(xb)
            loss = criterion(yp, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        avg_loss = total_loss / len(train_t)

        model.eval()
        with torch.no_grad():
            vp = model(val_t)
            nmad = metric_fn(vp, val_tz).item()
        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(avg_loss)
        history["val_nmad"].append(nmad)
        history["lr"].append(lr)
        dt = time.time() - t0
        print(f"  {model_type} Ep {ep:2d}/{cfg['epochs']}  loss={avg_loss:.4f}  "
              f"val_nmad={nmad:.5f}  lr={lr:.2e}  {dt:.0f}s")

        if nmad < best_nmad:
            best_nmad = nmad
            model_fn = f"{cfg['ckpt_dir']}/tiny_{model_type}_best.pth"
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                        "best_val_nmad": nmad, "history": history}, model_fn)
            patience = 0
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print(f"  Early stop at {ep}")
                break
    return model, history, best_nmad


def evaluate(model, test_feats, test_z, test_ids, test_snr, cfg, device, model_type, history=None):
    model.eval()
    test_t = torch.from_numpy(test_feats).to(device)
    with torch.no_grad():
        preds = model(test_t).cpu().numpy()
    trues = test_z

    met = _compute_metrics(trues, preds)
    met["per_z"] = _per_bin(trues, preds, bins=[0, 0.5, 1, 1.5, 2, 3])
    if test_snr is not None:
        met["per_snr"] = _per_bin(trues, preds, bins=[0, 2.5, 5, 10, 20, 1e9],
                                  values=test_snr)
    met["n"] = len(trues)
    print(f"  {model_type} test: NMAD={met['nmad']:.5f} eta={met['eta']:.2f}%")

    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    pdf = pd.DataFrame({"TARGETID": [str(s) for s in test_ids],
                        "z_true": trues, "z_pred": preds})
    if test_snr is not None:
        pdf["snr"] = test_snr
    pdf.to_csv(f"{cfg['ckpt_dir']}/tiny_{model_type}_test_preds.csv", index=False)
    with open(f"{cfg['ckpt_dir']}/tiny_{model_type}_test_metrics.json", "w") as f:
        json.dump(met, f, indent=2, default=str)

    # Training curves
    if history:
        plot_training_curves(history, cfg["output_dir"], model_type)
    # Diagnostic plots
    plot_diagnostics(trues, preds, cfg["output_dir"], model_type, test_snr=test_snr)
    return met, (trues, preds, test_ids)


def plot_training_curves(history, out_dir, model_type):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].plot(history["val_nmad"], label="Val NMAD")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("NMAD")
    for ax in axes: ax.legend(); ax.grid()
    fig.savefig(f"{out_dir}/tiny_{model_type}_training_curves.png", dpi=150)
    plt.close(fig)


def plot_diagnostics(trues, preds, out_dir, model_type, test_snr=None):
    dz = np.abs(preds - trues) / (1 + trues)
    outlier_mask = dz > 0.15

    # 1) Standard scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(trues, preds, s=2, alpha=0.3, c="steelblue")
    lims = [min(trues.min(), preds.min()), max(trues.max(), preds.max())]
    ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("z true"); ax.set_ylabel("z pred")
    ax.set_title(f"{model_type} (NMAD={np.median(dz):.4f})")
    ax.axis("equal")
    fig.savefig(f"{out_dir}/tiny_{model_type}_scatter_standard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) SNR-colored scatter
    if test_snr is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(trues, preds, s=2, c=np.log10(np.clip(test_snr, 1, 1e3)),
                        cmap="plasma", alpha=0.5)
        ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
        plt.colorbar(sc, ax=ax, label="log10(SNR)")
        ax.set_xlabel("z true"); ax.set_ylabel("z pred")
        ax.set_title(f"{model_type} (SNR colored)")
        ax.axis("equal")
        fig.savefig(f"{out_dir}/tiny_{model_type}_scatter_snr_colored.png",
                     dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3) Per-z-bin
    zbins = [0, 0.5, 1, 1.5, 2, 3]
    zlabels = [f"[{zbins[i]},{zbins[i+1]})" for i in range(len(zbins)-1)]
    z_nmads, z_etas = [], []
    for i in range(len(zbins)-1):
        lo, hi = zbins[i], zbins[i+1]
        m = (trues >= lo) & (trues < hi)
        if m.sum() > 1:
            z_nmads.append(np.median(np.abs(preds[m]-trues[m])/(1+trues[m])))
            z_etas.append((np.abs(preds[m]-trues[m])/(1+trues[m])>0.15).mean()*100)
        else:
            z_nmads.append(0); z_etas.append(0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(zlabels, z_nmads, color="steelblue")
    axes[0].set_ylabel("NMAD"); axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(zlabels, z_etas, color="coral")
    axes[1].set_ylabel("Eta (%)"); axes[1].tick_params(axis="x", rotation=45)
    for ax in axes: ax.grid(axis="y")
    fig.suptitle(f"{model_type} per-z-bin")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/tiny_{model_type}_per_z_bin.png", dpi=150)
    plt.close(fig)

    # 4) Per-SNR-bin
    if test_snr is not None:
        s_bins = [0, 2.5, 5, 10, 20, 1e9]
        s_labels = ["[2.5,5)", "[5,10)", "[10,20)", "[20,∞)"]
        s_nmads, s_etas = [], []
        for i in range(len(s_bins)-1):
            lo, hi = s_bins[i], s_bins[i+1]
            m = (test_snr >= lo) & (test_snr < hi)
            if m.sum() > 1:
                s_nmads.append(np.median(np.abs(preds[m]-trues[m])/(1+trues[m])))
                s_etas.append((np.abs(preds[m]-trues[m])/(1+trues[m])>0.15).mean()*100)
            else:
                s_nmads.append(0); s_etas.append(0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(s_labels, s_nmads, color="steelblue")
        axes[0].set_ylabel("NMAD"); axes[0].tick_params(axis="x", rotation=45)
        axes[1].bar(s_labels, s_etas, color="coral")
        axes[1].set_ylabel("Eta (%)"); axes[1].tick_params(axis="x", rotation=45)
        for ax in axes: ax.grid(axis="y")
        fig.suptitle(f"{model_type} per-SNR-bin")
        fig.tight_layout()
        fig.savefig(f"{out_dir}/tiny_{model_type}_per_snr_bin.png", dpi=150)
        plt.close(fig)

    # 5) Outlier diagnostic
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(dz, bins=50, color="steelblue", alpha=0.7)
    axes[0].axvline(0.15, color="red", ls="--", lw=1, label="eta threshold")
    axes[0].set_xlabel("|dz|/(1+z)"); axes[0].set_ylabel("N"); axes[0].legend()
    axes[1].scatter(trues, dz, s=2, alpha=0.3, c="steelblue")
    axes[1].axhline(0.15, color="red", ls="--", lw=1)
    axes[1].set_xlabel("z true"); axes[1].set_ylabel("|dz|/(1+z)")
    axes[2].hist(trues[~outlier_mask], bins=40, alpha=0.5, label="Inliers", color="steelblue")
    axes[2].hist(trues[outlier_mask], bins=40, alpha=0.5, label="Outliers", color="coral")
    axes[2].set_xlabel("z true"); axes[2].set_ylabel("N"); axes[2].legend()
    for ax in axes: ax.grid(True, alpha=0.3)
    fig.suptitle(f"{model_type} outlier diagnostic")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/tiny_{model_type}_outlier_diagnostic.png", dpi=150)
    plt.close(fig)


def run_model(args, model_type, cache):
    MODEL_CLS = lambda: MODELS[model_type](input_dim=512, hidden_dim=256, output_dim=1)
    cfg = {"lr": args.lr, "weight_decay": args.wd, "epochs": args.epochs,
           "patience": args.patience, "batch_size": args.batch_size,
           "ckpt_dir": args.ckpt_dir, "output_dir": args.output_dir}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining {model_type} on {device}...")
    model, history, best_nmad = train_model(
        MODEL_CLS, cache["train_features"], cache["train_z"],
        cache["val_features"], cache["val_z"], cfg, device, model_type)

    # Reload best
    ckpt_fn = f"{cfg['ckpt_dir']}/tiny_{model_type}_best.pth"
    ckpt = torch.load(ckpt_fn, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    met, _ = evaluate(model, cache["test_features"], cache["test_z"],
                      cache["test_ids"], cache.get("test_snr"), cfg, device, model_type,
                      history=ckpt["history"])
    met["best_val_nmad"] = ckpt["best_val_nmad"]
    met["history"] = ckpt.get("history", {})
    return met


def compare(cache, args):
    """Generate comparison between Stage 1 (linear probe), tiny MLP, and tiny ResNet."""
    results = {}
    for model_type in ["linear", "mlp", "resnet"]:
        p = f"{args.ckpt_dir}/tiny_{model_type}_test_preds.csv"
        if os.path.exists(p):
            df = pd.read_csv(p)
            met = _compute_metrics(df.z_true.values, df.z_pred.values)
            results[model_type] = {"nmad": met["nmad"], "eta": met["eta"]}
    if not results:
        print("No results to compare")
        return

    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"linear": "#2E86AB", "mlp": "#A23B72", "resnet": "#F18F01"}
    models = list(results.keys())
    axes[0].bar(models, [results[m]["nmad"] for m in models],
                color=[colors.get(m, "steelblue") for m in models])
    axes[0].set_ylabel("NMAD"); axes[0].grid(axis="y")
    axes[1].bar(models, [results[m]["eta"] for m in models],
                color=[colors.get(m, "steelblue") for m in models])
    axes[1].set_ylabel("Eta (%)"); axes[1].grid(axis="y")
    for ax in axes: ax.tick_params(axis="x", rotation=30)
    fig = axes[0].get_figure()
    fig.suptitle("Model Comparison on Real 3D-HST")
    fig.tight_layout()
    fig.savefig(f"{args.output_dir}/tiny_comparison_nmad_bar.png", dpi=150)
    plt.close(fig)
    print(f"Comparison: {json.dumps(results, default=str, indent=2)}")


def log_wandb(args, results):
    import wandb
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = wandb.init(id=args.wandb_id, resume="allow",
                         project=args.wandb_project, entity=args.wandb_entity)
    for model_type, met in results.items():
        prefix = f"tiny_real_{model_type}"
        wandb.log({f"{prefix}_nmad": met["nmad"], f"{prefix}_eta": met["eta"],
                   f"{prefix}_best_val_nmad": met.get("best_val_nmad", 0),
                   f"{prefix}_n": met["n"]})
        # Log images
        for fname in ["scatter_standard", "scatter_snr_colored", "per_z_bin",
                       "per_snr_bin", "outlier_diagnostic", "training_curves"]:
            ip = f"{args.output_dir}/tiny_{model_type}_{fname}.png"
            if os.path.exists(ip):
                wandb.log({f"{prefix}_{fname}": wandb.Image(ip)})
        pred_path = f"{args.ckpt_dir}/tiny_{model_type}_test_preds.csv"
        if os.path.exists(pred_path):
            pdf = pd.read_csv(pred_path, nrows=5)  # preview only
            table = wandb.Table(dataframe=pd.read_csv(pred_path))
            wandb.log({f"{prefix}_predictions": table})
    # Comparison image
    comp_path = f"{args.output_dir}/tiny_comparison_nmad_bar.png"
    if os.path.exists(comp_path):
        wandb.log({"tiny_real_comparison": wandb.Image(comp_path)})
    wandb.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["extract", "train", "all", "compare"])
    ap.add_argument("--model", default="mlp", choices=["mlp", "resnet"])
    ap.add_argument("--real-data", default="/home/ckb2084/research/SpecPT/data/grism_specPT_v5.pkl")
    ap.add_argument("--init-ckpt", default="/home/ckb2084/research/specpt-hst-sim/checkpoints/exp_032_best_model.pth")
    ap.add_argument("--ae-path", default="/home/ckb2084/research/galax_spec/pretrained_weights/SpecPT_DESI_combined_autoencoder_150_new.pth")
    ap.add_argument("--model-cfg", default="configs/exp_032.yaml")
    ap.add_argument("--cache-path", default="checkpoints/finetune_real/exp_032_features.npz")
    ap.add_argument("--ckpt-dir", default="checkpoints/finetune_real")
    ap.add_argument("--output-dir", default="outputs/real_3dhst")
    ap.add_argument("--min-snr", type=float, default=2.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--wandb-id", default="ejfhtjlk")
    ap.add_argument("--wandb-entity", default="ckb2084-rochester-institute-of-technology")
    ap.add_argument("--wandb-project", default="specpt-hst-sim-z")
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.command == "extract":
        extract(args)

    elif args.command == "train":
        if not os.path.exists(args.cache_path):
            extract(args)
        cache = load_cache(args.cache_path)
        results = {}
        for mt in (["mlp", "resnet"] if args.model == "all" else [args.model]):
            results[mt] = run_model(args, mt, cache)
            print(f"  {mt}: Test NMAD={results[mt]['nmad']:.5f}  Eta={results[mt]['eta']:.2f}%")
        log_wandb(args, results)

    elif args.command == "all":
        if not os.path.exists(args.cache_path):
            extract(args)
        cache = load_cache(args.cache_path)
        results = {}
        for mt in ["mlp", "resnet"]:
            results[mt] = run_model(args, mt, cache)
            print(f"  {mt}: Test NMAD={results[mt]['nmad']:.5f}  Eta={results[mt]['eta']:.2f}%")
        # Compare with Stage 1 linear probe (existing CSV or these results)
        compare(cache, args)
        log_wandb(args, results)

    elif args.command == "compare":
        cache = load_cache(args.cache_path) if os.path.exists(args.cache_path) else None
        compare(cache, args)


if __name__ == "__main__":
    main()
