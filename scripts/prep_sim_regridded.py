#!/usr/bin/env python3
"""
Prepare re-gridded sim training data on the real 3D-HST wavelength grid.

Mirrors notebooks/01_data_exploration_and_training_format.ipynb but replaces
the target grid:

    OLD: np.linspace(10311.4, 17464.6, 7781)  -> 0.9199 A/pix  (sim-native)
    NEW: np.linspace(10800.0, 17100.0, 7781) -> 0.8080 A/pix  (real 3D-HST)

Both grids use 7781 pixels so input_size is unchanged and the frozen DESI
autoencoder checkpoint still loads. The 0.81 A/pix dispersion matches the
real grism_specPT_v5 data, so features land at the same pixel positions the
frozen conv layers learned — eliminating the 187-pixel H-alpha shift.

Quality cuts match notebook 01 / exp_032 Q1 data:
    line_peak_snr in [2.5, 15], z_quality >= 1

NOTE: The local HDF5 (data/hst_grism_combined.h5) is a different dataset than
the notebook's source (F:\personal_projects\HST_GRISM_Sim\output\hst_grism_combined.h5).
The local copy has much higher line_peak_snr values (median ~50 vs all <= 15 in
the notebook). With --snr-max 15, the local HDF5 produces ~14,492 spectra instead
of 72,361. Use --snr-max 100 (or omit) to keep all high-SNR spectra.

The G141 reliable-range mask (11000-16500 A) is applied AFTER interpolation
so NaN distributions match between sim and real data (z-score parity).

Output schema matches grism_training_sim_v2_Q1.parquet exactly:
    grism_id (str), z (float32), SNR (float32),
    spec (list<float32>), clean_flux_resampled (list<float32>)

Usage:
    python scripts/prep_sim_regridded.py \
        --src  /home/ckb2084/research/specpt-hst-sim/data/hst_grism_combined.h5 \
        --out  /home/ckb2084/research/specpt-hst-sim/data/training_format/grism_training_sim_v3_regrid.parquet

    # Override mask range or disable trimming:
    python scripts/prep_sim_regridded.py --no-trim
    python scripts/prep_sim_regridded.py --wave-min 10900 --wave-max 17000

    # simv4a release (pre-selected, flat HDF5, catalog_snr as SNR, no cuts):
    python scripts/prep_sim_regridded.py \
        --src  /path/to/simv4a_uniform_z_selection/track_a.h5 \
        --out  /path/to/data/training_format/grism_training_simv4a.parquet \
        --flat --snr-column catalog_snr --no-filter --passthrough-1d
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

N_PIXELS = 7781
NEW_GRID_START = 10800.0
NEW_GRID_END = 17100.0

DEFAULT_WAVE_MIN = 11000.0
DEFAULT_WAVE_MAX = 16500.0

SNR_MIN = 2.5
SNR_MAX = 15.0
Z_QUALITY_MIN = 1

TRACK = "track_a"

# 1D scalar columns to exclude from --passthrough-1d (handled separately or
# spectral arrays). Line *_FLUX / *_ERR columns are 1D scalars and ARE passed
# through when --passthrough-1d is set.
PASSTHROUGH_EXCLUDE = {
    "wave", "flam", "cont", "line", "ferr", "sensitivity", "contam",
    "redshift", "z_quality", "h_mag", "snr_median", "t_exp", "read_noise",
    "grism_id", "catalog_snr",
    "sensitivity_resampled", "contam_resampled", "clean_flux_resampled",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True,
                   help="Path to hst_grism_combined.h5 source HDF5.")
    p.add_argument("--out", required=True,
                   help="Output parquet path (.parquet).")
    p.add_argument("--track", default=TRACK,
                   help=f"HDF5 group name (default: {TRACK}).")
    p.add_argument("--flat", action="store_true",
                   help="Treat source HDF5 as flat (no group). Auto-detected "
                        "when --track group is absent and 'flam' is at root.")
    p.add_argument("--wave-min", type=float, default=DEFAULT_WAVE_MIN,
                   help=f"Blue trim edge in A (default: {DEFAULT_WAVE_MIN}).")
    p.add_argument("--wave-max", type=float, default=DEFAULT_WAVE_MAX,
                   help=f"Red trim edge in A (default: {DEFAULT_WAVE_MAX}).")
    p.add_argument("--no-trim", dest="trim", action="store_false",
                   help="Disable G141 post-interp NaN masking.")
    p.add_argument("--snr-min", type=float, default=SNR_MIN,
                   help="Minimum SNR filter (default: 2.5)")
    p.add_argument("--snr-max", type=float, default=SNR_MAX,
                   help="Maximum SNR filter (default: 15.0)")
    p.add_argument("--z-quality-min", type=int, default=Z_QUALITY_MIN)
    p.add_argument("--snr-column", choices=["line_peak_snr", "catalog_snr"],
                   default="line_peak_snr",
                   help="Column to store as 'SNR' in the output parquet "
                        "(default: line_peak_snr, computed from arrays). "
                        "Use 'catalog_snr' for pre-selected releases.")
    p.add_argument("--no-filter", action="store_true",
                   help="Disable ALL filtering (snr-min/max and z-quality). "
                        "Passes every spectrum through.")
    p.add_argument("--passthrough-1d", action="store_true",
                   help="Include all extra 1D scalar columns from the source "
                        "HDF5 (e.g. selection_weight, SED params, line "
                        "fluxes) as additional parquet columns.")
    return p.parse_args()


def main():
    args = parse_args()

    import h5py

    new_wavelengths = np.linspace(NEW_GRID_START, NEW_GRID_END, N_PIXELS)
    print(f"Target grid: {NEW_GRID_START}-{NEW_GRID_END} A, {N_PIXELS} px, "
          f"{new_wavelengths[1] - new_wavelengths[0]:.4f} A/pix")
    print(f"Trim: {args.trim} ({args.wave_min}-{args.wave_max} A)" if args.trim else "Trim: disabled")
    print(f"Source: {args.src}  (group=/{args.track}/)")
    print(f"Output: {args.out}")

    if args.trim:
        trim_mask = (new_wavelengths < args.wave_min) | (new_wavelengths > args.wave_max)
        n_trim = int(trim_mask.sum())
        print(f"Trim mask: {n_trim}/{N_PIXELS} pixels ({100 * n_trim / N_PIXELS:.1f}%) -> NaN")
    else:
        trim_mask = None

    print("Reading HDF5 ...")
    with h5py.File(args.src, "r") as f:
        if args.flat or (args.track not in f and "flam" in f):
            g = f
            print("  Flat HDF5 layout (no group)")
        else:
            g = f[args.track]
            print(f"  Using /{args.track}/ group")
        n_spec = int(g["redshift"].shape[0])
        print(f"  {n_spec:,} spectra")
        grism_id = [x.decode() if isinstance(x, bytes) else str(x) for x in g["grism_id"][:]]
        redshift = g["redshift"][:].astype(np.float32)
        z_quality = g["z_quality"][:].astype(np.int8)
        snr_med = g["snr_median"][:].astype(np.float32)
        flam_2d = g["flam"][:]
        cont_2d = g["cont"][:]
        ferr_2d = g["ferr"][:]
        wave = list(g["wave"][:])

        catalog_snr = None
        if args.snr_column == "catalog_snr":
            if "catalog_snr" not in g:
                print("ERROR: --snr-column catalog_snr requested but "
                      "'catalog_snr' is not in the HDF5")
                sys.exit(1)
            catalog_snr = g["catalog_snr"][:].astype(np.float32)

        extras = {}
        if args.passthrough_1d:
            for key in g.keys():
                if key in PASSTHROUGH_EXCLUDE:
                    continue
                ds = g[key]
                if ds.ndim == 1 and ds.shape[0] == n_spec:
                    extras[key] = ds[:]

    ratio = (flam_2d - cont_2d) / np.where(ferr_2d > 0, ferr_2d, np.inf)
    line_peak_snr = ratio.max(axis=1).astype(np.float32)
    del ratio, ferr_2d

    print(f"line_peak_snr: min={line_peak_snr.min():.2f}  "
          f"median={np.median(line_peak_snr):.2f}  max={line_peak_snr.max():.2f}")

    keep = np.ones(n_spec, dtype=bool)
    if args.no_filter:
        print("Filtering disabled (--no-filter): passing all spectra")
    else:
        if args.snr_min is not None:
            before = int(keep.sum())
            if args.snr_column == "catalog_snr":
                keep &= catalog_snr >= args.snr_min
                lbl = "catalog_snr"
            else:
                keep &= line_peak_snr >= args.snr_min
                lbl = "line_peak_snr"
            print(f"Filter {lbl} >= {args.snr_min}: {before} -> {int(keep.sum())}")
        if args.snr_max is not None:
            before = int(keep.sum())
            if args.snr_column == "catalog_snr":
                keep &= catalog_snr <= args.snr_max
                lbl = "catalog_snr"
            else:
                keep &= line_peak_snr <= args.snr_max
                lbl = "line_peak_snr"
            print(f"Filter {lbl} <= {args.snr_max}: {before} -> {int(keep.sum())}")
        if args.z_quality_min is not None:
            before = int(keep.sum())
            keep &= z_quality >= args.z_quality_min
            print(f"Filter z_quality >= {args.z_quality_min}: {before} -> {int(keep.sum())}")

    idx = np.where(keep)[0]
    n_keep = len(idx)
    print(f"Retained: {n_keep:,} / {n_spec:,} spectra")

    print("Interpolating onto new grid ...")
    specs = [None] * n_keep
    nan_counts = np.zeros(n_keep, dtype=np.int64)
    for k, i in enumerate(tqdm(idx, desc="Interpolating", unit="spec")):
        wl = np.asarray(wave[i], dtype=np.float64)
        fl = np.asarray(flam_2d[i], dtype=np.float64)
        g = np.interp(new_wavelengths, wl, fl, left=np.nan, right=np.nan)
        if trim_mask is not None:
            g[trim_mask] = np.nan
        g = g.astype(np.float32)
        nan_counts[k] = int(np.isnan(g).sum())
        specs[k] = g

    del flam_2d, cont_2d, wave

    print(f"NaN per spectrum: mean={nan_counts.mean():.1f}  "
          f"median={np.median(nan_counts):.0f}  "
          f"min={nan_counts.min()}  max={nan_counts.max()}  "
          f"({100 * nan_counts.mean() / N_PIXELS:.1f}% on average)")

    print("Building wide DataFrame ...")
    wide_df = pd.DataFrame({
        "grism_id": [grism_id[i] for i in idx],
        "z": redshift[idx].astype(np.float32),
        "SNR": (catalog_snr[idx] if catalog_snr is not None
                else line_peak_snr[idx]).astype(np.float32),
        "spec": specs,
        "clean_flux_resampled": specs,
    })
    for key, arr in extras.items():
        wide_df[key] = arr[idx]

    print(f"DataFrame shape: {wide_df.shape}")
    print(f"Columns: {list(wide_df.columns)}")
    print(f"spec[0] dtype: {wide_df['spec'].iloc[0].dtype}, shape: {wide_df['spec'].iloc[0].shape}")
    print(f"NaN in spec[0]: {int(np.isnan(wide_df['spec'].iloc[0]).sum())} / {N_PIXELS}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"Writing parquet -> {args.out}")
    wide_df.to_parquet(args.out, index=False)

    size_gb = os.path.getsize(args.out) / 1e9
    print(f"Wrote {n_keep:,} spectra, {size_gb:.2f} GB")
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())