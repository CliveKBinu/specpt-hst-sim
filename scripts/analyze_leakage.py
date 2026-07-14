#!/usr/bin/env python3
"""Data leakage analysis for augmented grism spectroscopy datasets.

Reports cross-split ID overlap, near-duplicate spectra, and distribution
balance for train/val/test splits.

Usage:
    python scripts/analyze_leakage.py \\
        --data /path/to/grism_specPT_augumented_v2_more_data.pkl \\
        --out outputs/leakage \\
        --seed 42
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.specpt.dataloader import _patch_pickle_compat, split_data


def parse_args():
    p = argparse.ArgumentParser(description="Data leakage analysis for augmented spectroscopy datasets.")
    p.add_argument("--data", required=True, help="Path to pickle/parquet file")
    p.add_argument("--out", default="outputs/leakage", help="Output directory")
    p.add_argument("--rows-limit", type=int, default=0,
                   help="Load only first N rows for a quick smoke test (0 = all)")
    p.add_argument("--dup-batch-mb", type=int, default=512,
                   help="Target memory per cosine similarity batch chunk (MB)")
    p.add_argument("--dup-thresholds", type=float, nargs="+",
                   default=[0.99, 0.999, 0.9999, 0.99999],
                   help="Cosine similarity thresholds for near-dup detection")
    p.add_argument("--self-dup-sample", type=int, default=5000,
                   help="Sample size for within-set near-dup check. 0 = skip")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def log(msg):
    print(msg, flush=True)


def peek_schema(df):
    n_rows = len(df)
    log(f"  Shape: {n_rows} rows x {len(df.columns)} cols")
    log(f"  Columns: {list(df.columns)}")
    for c in df.columns:
        first = df[c].iloc[0]
        is_arr = isinstance(first, (np.ndarray, list))
        details = f"dtype={df[c].dtype}"
        if not is_arr:
            details += f", NaN={df[c].isna().sum()}"
            if df[c].dtype in ("object", "category"):
                details += f", unique={df[c].nunique()}"
        log(f"    {c}: {details}")
        if is_arr:
            arr = np.asarray(first)
            log(f"      -> array shape={arr.shape}, dtype={arr.dtype}, "
                f"NaN={np.isnan(arr).sum()}/{len(arr)}")
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    log(f"  Memory (deep): {mem_mb:.0f} MB")
    return {"n_rows": n_rows, "n_cols": len(df.columns), "memory_mb": round(mem_mb, 1)}


def auto_detect_columns(df):
    id_col = next((c for c in ["grism_id", "TARGETID", "id", "source_id", "galaxy_id"]
                   if c in df.columns), None)
    if id_col is None:
        for c in df.columns:
            if df[c].dtype in ("object", "category"):
                id_col = c
                break
    n_unq = df[id_col].nunique() if id_col else "?"

    z_col = next((c for c in ["z", "redshift", "z_true", "z_truth"] if c in df.columns), None)
    spec_col = next((c for c in ["spec", "clean_flux_resampled", "flux", "flam"]
                     if c in df.columns), None)
    if spec_col is None:
        for c in df.columns:
            if isinstance(df[c].iloc[0], (np.ndarray, list)):
                spec_col = c
                break
    snr_col = next((c for c in ["SNR", "snr", "line_peak_snr", "snr_median"] if c in df.columns), None)

    log(f"\n  Auto-detected columns:")
    log(f"    id_col   = {id_col}  (unique: {n_unq})")
    log(f"    z_col    = {z_col}")
    log(f"    spec_col = {spec_col}")
    log(f"    snr_col  = {snr_col}")
    return id_col, z_col, spec_col, snr_col


def load_data(args):
    _patch_pickle_compat()
    log(f"[1] Loading data from {args.data}")
    t0 = time.time()
    ext = os.path.splitext(args.data)[1]
    if ext == ".parquet":
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_pickle(args.data)
    dt = time.time() - t0
    log(f"    Loaded {len(df)} rows in {dt:.0f}s")
    if args.rows_limit and args.rows_limit < len(df):
        df = df.iloc[:args.rows_limit]
        log(f"    [LIMIT] Restricted to {len(df)} rows")
    return df


def safe_array(sdf, col):
    """Extract array column from DataFrame as float32 2D numpy array, NaN->0."""
    if col is None or len(sdf) == 0:
        return None
    arr = np.stack(sdf[col].values).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    return arr


def analyze_augmentation(df, id_col):
    log(f"\n[2] Augmentation factor analysis")
    counts = df[id_col].value_counts()
    result = {
        "total_rows": len(df),
        "unique_ids": int(len(counts)),
        "copies_per_id": {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "median": float(counts.median()),
            "mean": float(counts.mean()),
            "p25": float(counts.quantile(0.25)),
            "p75": float(counts.quantile(0.75)),
            "p95": float(counts.quantile(0.95)),
        },
        "multiplicity_bins": {
            "singletons (1x)": int((counts == 1).sum()),
            "2-3 copies": int(((counts >= 2) & (counts <= 3)).sum()),
            "4-9 copies": int(((counts >= 4) & (counts <= 9)).sum()),
            "10+ copies": int((counts >= 10).sum()),
        },
    }
    log(f"    Total rows: {result['total_rows']}")
    log(f"    Unique IDs: {result['unique_ids']}")
    log(f"    Copies/ID:  min={result['copies_per_id']['min']}, "
        f"max={result['copies_per_id']['max']}, "
        f"median={result['copies_per_id']['median']:.1f}, "
        f"mean={result['copies_per_id']['mean']:.1f}")
    for k, v in result['multiplicity_bins'].items():
        log(f"      {k}: {v}")
    return result, counts


def analyze_split_overlap(df, id_col, seed):
    log(f"\n[3] Simulating data split (seed={seed})")
    train, val, test = split_data(df, val_split=0.1, test_split=0.1, seed=seed)
    splits = {"train": train, "val": val, "test": test}
    for name, sdf in splits.items():
        log(f"    {name}: {len(sdf)} rows")

    train_ids = set(train[id_col].unique())
    val_ids = set(val[id_col].unique())
    test_ids = set(test[id_col].unique())

    def overlap_stats(subset_ids, subset_df, name):
        leaked_ids = subset_ids & train_ids
        n_leaked_rows = int(subset_df[id_col].isin(leaked_ids).sum())
        pct = 100.0 * n_leaked_rows / len(subset_df) if len(subset_df) > 0 else 0.0
        return {
            "n_ids_in_train": len(leaked_ids),
            "n_rows_leaked": n_leaked_rows,
            "pct_rows_leaked": round(pct, 2),
        }

    overlap = {
        "val_vs_train": overlap_stats(val_ids, val, "val"),
        "test_vs_train": overlap_stats(test_ids, test, "test"),
    }

    log(f"\n  --- Cross-split ID Overlap ---")
    log(f"  Train: {len(train)} rows, {len(train_ids)} unique IDs")
    for key, stats in overlap.items():
        split_name = key.split("_")[0]
        split_len = len(splits[split_name])
        log(f"  {key}: {stats['n_ids_in_train']} IDs also in train -> "
            f"{stats['n_rows_leaked']}/{split_len} "
            f"rows ({stats['pct_rows_leaked']:.1f}%)")

    return splits, overlap


def analyze_distribution(splits, z_col, snr_col, out_dir):
    if z_col is None and snr_col is None:
        return {}
    log(f"\n[4] Distribution analysis")
    result = {}
    colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}

    for col, label in [(z_col, "redshift"), (snr_col, "SNR")]:
        if col is None:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        col_result = {}
        for name in ["train", "val", "test"]:
            vals = splits[name][col].dropna().values
            col_result[name] = {
                "count": len(vals),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
            }
            ax.hist(vals, bins=80, alpha=0.5, label=f"{name} (n={len(vals)})",
                    color=colors[name], density=True)
        from scipy.stats import ks_2samp
        for pair_name, a_name, b_name in [("train_vs_val", "train", "val"),
                                           ("train_vs_test", "train", "test")]:
            a = splits[a_name][col].dropna().values
            b = splits[b_name][col].dropna().values
            ks = ks_2samp(a, b)
            col_result[f"ks_{pair_name}"] = {
                "statistic": round(ks.statistic, 5),
                "pvalue": float(ks.pvalue),
            }
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(f"{label.capitalize()} distribution per split")
        ax.legend()
        fig.savefig(out_dir / f"{label.replace(' ', '_')}_dist.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        result[label] = col_result
        log(f"    {label}: train(vs_val KS={col_result.get('ks_train_vs_val', {}).get('statistic', '?'):.4f}, "
            f"vs_test KS={col_result.get('ks_train_vs_test', {}).get('statistic', '?'):.4f})")
    return result


def analyze_grouped_split(df, id_col, z_col, seed, out_dir):
    log(f"\n[5] Grouped-split baseline (GroupShuffleSplit by {id_col})")
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, temp_idx = next(gss.split(df, groups=df[id_col]))
    temp = df.iloc[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp, groups=temp[id_col]))
    train_g = df.iloc[train_idx]
    val_g = temp.iloc[val_idx]
    test_g = temp.iloc[test_idx]

    train_ids = set(train_g[id_col].unique())
    val_ids = set(val_g[id_col].unique())
    test_ids = set(test_g[id_col].unique())
    result = {
        "split_sizes": {"train": len(train_g), "val": len(val_g), "test": len(test_g)},
        "unique_ids": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "id_overlap_val_train": len(val_ids & train_ids),
        "id_overlap_test_train": len(test_ids & train_ids),
    }
    log(f"    Grouped split: train={len(train_g)}, val={len(val_g)}, test={len(test_g)}")
    log(f"    Cross-split ID overlap (should be 0): val_vs_train={result['id_overlap_val_train']}, "
        f"test_vs_train={result['id_overlap_test_train']}")

    if z_col is not None:
        for name, sdf in [("train", train_g), ("val", val_g), ("test", test_g)]:
            log(f"    {name}: z_mean={sdf[z_col].mean():.3f}, z_std={sdf[z_col].std():.3f}")

    return result


def _normalize(A):
    A = A.astype(np.float32)
    norm = np.linalg.norm(A, axis=1, keepdims=True)
    return np.where(norm > 1e-8, A / norm, 0.0).astype(np.float32)


def batched_cosine_sim(A, B, batch_mb=512):
    n, m = len(A), len(B)
    max_sim = np.full(n, -1.0, dtype=np.float32)
    max_idx = np.full(n, -1, dtype=np.int64)
    A_norm = _normalize(A)
    B_norm = _normalize(B)

    try:
        import torch
        A_t = torch.from_numpy(A_norm)
        B_t = torch.from_numpy(B_norm)
        if torch.cuda.is_available():
            A_t = A_t.cuda()
            B_t = B_t.cuda()
        use_torch = True
    except ImportError:
        use_torch = False

    feat_dim = B.shape[1]
    rows_per_chunk = max(1, int(batch_mb * 1024 * 1024 / (4 * feat_dim)))
    rows_per_chunk = min(rows_per_chunk, n)

    t0 = time.time()
    if use_torch:
        for i in range(0, n, rows_per_chunk):
            chunk = A_t[i:i + rows_per_chunk]
            sim = torch.mm(chunk, B_t.T)
            vals, idxs = sim.max(dim=1)
            max_sim[i:i + len(chunk)] = vals.cpu().numpy()
            max_idx[i:i + len(chunk)] = idxs.cpu().numpy()
    else:
        for i in range(0, n, rows_per_chunk):
            chunk = A_norm[i:i + rows_per_chunk]
            sim = chunk @ B_norm.T
            max_sim[i:i + len(chunk)] = sim.max(axis=1)
            max_idx[i:i + len(chunk)] = sim.argmax(axis=1)
    dt = time.time() - t0
    use_gpu = use_torch and torch.cuda.is_available()
    log(f"    Computed {n}x{m} cosine similarity in {dt:.1f}s using "
        f"{'GPU' if use_gpu else 'CPU'} "
        f"(chunk={rows_per_chunk}, {n // rows_per_chunk + 1} chunks)")
    return max_sim, max_idx


def batched_cosine_self(A, sample_size, batch_mb=512):
    n = len(A)
    if sample_size >= n:
        sample_idx = np.arange(n)
    else:
        np.random.seed(42)
        sample_idx = np.random.choice(n, sample_size, replace=False)
    A_norm = _normalize(A)

    try:
        import torch
        A_t = torch.from_numpy(A_norm)
        if torch.cuda.is_available():
            A_t = A_t.cuda()
        use_torch = True
    except ImportError:
        use_torch = False

    feat_dim = A.shape[1]
    rows_per_chunk = max(1, int(batch_mb * 1024 * 1024 / (4 * feat_dim)))
    rows_per_chunk = min(rows_per_chunk, len(sample_idx))

    max_sim = {}
    t0 = time.time()
    if use_torch:
        for i in range(0, len(sample_idx), rows_per_chunk):
            chunk_idx = sample_idx[i:i + rows_per_chunk]
            chunk = A_t[chunk_idx]
            sim = torch.mm(chunk, A_t.T)
            sim_np = sim.cpu().numpy() if torch.cuda.is_available() else sim.numpy()
            for j, ii in enumerate(chunk_idx):
                row = sim_np[j].copy()
                row[ii] = -1.0
                max_sim[int(ii)] = float(row.max())
    else:
        for i in range(0, len(sample_idx), rows_per_chunk):
            chunk_idx = sample_idx[i:i + rows_per_chunk]
            chunk = A_norm[chunk_idx]
            sim = chunk @ A_norm.T
            for j, ii in enumerate(chunk_idx):
                row = sim[j].copy()
                row[ii] = -1.0
                max_sim[int(ii)] = float(row.max())
    dt = time.time() - t0
    log(f"    Computed self-similarity on {len(sample_idx)} samples in {dt:.1f}s")
    return np.array(list(max_sim.values()))


def analyze_near_dup(splits, spec_col, args, out_dir):
    if spec_col is None:
        log(f"\n[6] Near-duplicate analysis: SKIPPED (no spec column found)")
        return {}
    log(f"\n[6] Near-duplicate spectral analysis")
    result = {}

    t0 = time.time()
    spec_train = safe_array(splits["train"], spec_col) if len(splits["train"]) > 0 else None
    spec_val = safe_array(splits["val"], spec_col) if len(splits["val"]) > 0 else None
    spec_test = safe_array(splits["test"], spec_col) if len(splits["test"]) > 0 else None
    log(f"    Extracted spectra: train={spec_train.shape if spec_train is not None else 'none'}, "
        f"val={spec_val.shape if spec_val is not None else 'none'}, "
        f"test={spec_test.shape if spec_test is not None else 'none'} "
        f"in {time.time() - t0:.1f}s")

    for name_a, spec_a, name_b, spec_b in [
        ("val", spec_val, "train", spec_train),
        ("test", spec_test, "train", spec_train),
    ]:
        if spec_a is None or spec_b is None:
            continue
        key = f"{name_a}_vs_{name_b}"
        log(f"\n  --- {key} ---")
        max_sim, _ = batched_cosine_sim(spec_a, spec_b, batch_mb=args.dup_batch_mb)
        thresholds = result[key] = {}
        for thresh in args.dup_thresholds:
            count = int((max_sim >= thresh).sum())
            pct = 100.0 * count / len(max_sim)
            thresholds[f"cos>={thresh}"] = {"count": count, "pct": round(pct, 3)}
            log(f"    cos >= {thresh}: {count}/{len(max_sim)} ({pct:.3f}%)")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(max_sim, bins=100, range=(0, 1), alpha=0.7)
        for thresh in args.dup_thresholds:
            ax.axvline(thresh, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Max cosine similarity to train")
        ax.set_ylabel("Count")
        ax.set_title(f"Near-dup detection: {name_a} vs {name_b}")
        fig.savefig(out_dir / f"near_dup_{key}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Within-set self-similarity (sample-based)
    if args.self_dup_sample > 0:
        for name, spec in [("val", spec_val), ("test", spec_test)]:
            if spec is None:
                continue
            sample = min(args.self_dup_sample, len(spec))
            if sample < 100:
                continue
            key = f"within_{name}"
            log(f"\n  --- {key} (sample={sample}) ---")
            max_sim = batched_cosine_self(spec, sample, batch_mb=args.dup_batch_mb)
            thresholds = result[key] = {}
            for thresh in args.dup_thresholds:
                count = int((max_sim >= thresh).sum())
                pct = 100.0 * count / len(max_sim)
                thresholds[f"cos>={thresh}"] = {"count": count, "pct": round(pct, 3)}
                log(f"    cos >= {thresh}: {count}/{sample} ({pct:.3f}%)")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(max_sim, bins=80, range=(0, 1), alpha=0.7)
            for thresh in args.dup_thresholds:
                ax.axvline(thresh, color="red", linestyle="--", alpha=0.5)
            ax.set_xlabel("Max cosine similarity within set (excl. self)")
            ax.set_ylabel("Count")
            ax.set_title(f"Self-similarity: {name} (sample={sample})")
            fig.savefig(out_dir / f"near_dup_within_{name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    return result


def plot_aug_factor(counts, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = counts.values
    capped = np.clip(vals, 1, 50)
    ax.hist(capped, bins=50, alpha=0.7)
    ax.set_xlabel("Copies per ID (capped at 50)")
    ax.set_ylabel("Number of IDs")
    ax.set_title("Augmentation multiplicity")
    fig.savefig(out_dir / "aug_factor.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_id_overlap(overlap, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    keys = list(overlap.keys())
    pcts = [overlap[k]["pct_rows_leaked"] for k in keys]
    ax.bar(keys, pcts, color=["#ff7f0e", "#2ca02c"])
    ax.set_ylabel("% rows leaking across split")
    ax.set_title("Cross-split ID leakage")
    for i, (k, p) in enumerate(zip(keys, pcts)):
        ax.text(i, p + 0.5, f"{p:.1f}%", ha="center")
    fig.savefig(out_dir / "id_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    # 1. Load data
    df = load_data(args)

    # 2. Schema
    log(f"\n--- Schema Probe ---")
    peek_schema(df)

    # 3. Auto-detect columns
    id_col, z_col, spec_col, snr_col = auto_detect_columns(df)
    if id_col is None:
        log("ERROR: Could not detect ID column. Aborting.")
        sys.exit(1)

    # 4. Augmentation factor
    aug_result, counts = analyze_augmentation(df, id_col)
    plot_aug_factor(counts, out_dir)

    # 5. Split overlap (reproduces actual training split)
    splits, overlap = analyze_split_overlap(df, id_col, args.seed)
    plot_id_overlap(overlap, out_dir)

    # 6. Distribution balance
    dist_result = analyze_distribution(splits, z_col, snr_col, out_dir)

    # 7. Grouped-split baseline
    grouped_result = analyze_grouped_split(df, id_col, z_col, args.seed, out_dir)

    # 8. Near-duplicate spectral analysis
    dup_result = analyze_near_dup(splits, spec_col, args, out_dir)

    # 9. Assemble report
    report = {
        "config": {
            "data": args.data,
            "rows_limit": args.rows_limit or "all",
            "seed": args.seed,
            "dup_thresholds": args.dup_thresholds,
            "self_dup_sample": args.self_dup_sample,
        },
        "augmentation": aug_result,
        "split_overlap": overlap,
        "distributions": dist_result,
        "grouped_split_baseline": grouped_result,
        "near_duplicate": dup_result,
    }

    # Summary text
    summary_lines = [
        "=" * 60,
        "DATA LEAKAGE ANALYSIS SUMMARY",
        "=" * 60,
        f"File: {args.data}",
        f"Rows: {aug_result['total_rows']} | Unique IDs: {aug_result['unique_ids']}",
        f"Augmentation factor: {aug_result['copies_per_id']['mean']:.1f}x (median {aug_result['copies_per_id']['median']:.1f}x)",
        "",
        "--- RANDOM SPLIT (train/val/test = 80/10/10, seed={}) ---".format(args.seed),
    ]
    for key, stats in overlap.items():
        summary_lines.append(
            f"  {key}: {stats['n_rows_leaked']}/{stats['pct_rows_leaked']:.1f}% rows "
            f"({stats['n_ids_in_train']} IDs) leaked across split"
        )

    summary_lines.append("")
    if dup_result:
        summary_lines.append("--- Near-Duplicate Detection ---")
        for key, thresholds in dup_result.items():
            summary_lines.append(f"  {key}:")
            for t_str, t_stats in thresholds.items():
                summary_lines.append(f"    {t_str}: {t_stats['count']} samples ({t_stats['pct']}%)")

    summary_lines.append("")
    summary_lines.append(f"Grouped split baseline ID overlap: val_vs_train={grouped_result.get('id_overlap_val_train', '?')}")
    summary_lines.append("=" * 60)

    # Save
    report_path = out_dir / "leakage_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log(f"\nSaved JSON report to {report_path}")

    summary_path = out_dir / "leakage_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    log(f"Saved summary to {summary_path}")

    log("\n" + "\n".join(summary_lines))


if __name__ == "__main__":
    main()
