# ⚡ SpecPT-HST-Sim

**Autonomous optimization engine for photometric redshift estimation on HST grism simulated data.**

[![Status](https://img.shields.io/badge/orchestrator-active-brightgreen)]()
[![Agent](https://img.shields.io/badge/agent-hermes-purple)]()
[![W&B](https://img.shields.io/badge/wandb-specpt--hst--sim-orange)]()

---

## 🎯 Mission

> Improve SpecPT redshift estimation by systematically exploring model capacity,
> hyperparameters, and training strategies. Every experiment is one controlled
> change, logged, and measured against the target.

| Metric | Target | Best | Gap |
|--------|--------|------|-----|
| NMAD (synthetic) | < 0.020 | **0.00785** (exp_032) | **✅ ACHIEVED** |
| NMAD (real 3D-HST) | < 0.100 | **0.20767** (exp_045_RF_fixed) | **0.10767** — encoder-level approach needed |
| Catastrophic outliers | < 1% | **12.73%** (exp_034) | **11.73%** |
| ECE | < 0.1 | — | — |

---

## 🔬 Active Experiments

| Experiment | Approach | Test NMAD | Test η | Status |
|------------|----------|-----------|--------|--------|
| exp_056 | EXACT exp_032 config on simv4a (DESI AE, zscore norm, random split) — isolates the data file | — | — | running (tigris job 83811) |
| exp_055 | Binned redshift head (24 log(1+z) bins, z≤4.0) on known-good v2_Q1 — attacks catastrophic outliers while holding NMAD | — | — | running (tigris job 83810) |
| exp_054 | Frozen simv4a-adapted AE + exp_032 head on simv4a — **FAILED** (test NMAD 0.3939, AE adaptation does not recover simv4a z-signal) | 0.3939 | 67.45% | completed — see leaderboard |
| exp_053 | Frozen regridded AE + exp_032 head on simv4a — **FAILED** (test NMAD 0.39077, simv4a does not transfer) | 0.39077 | 67.53% | completed — see leaderboard |

> ℹ️ **Track A (AE capacity sweep).** Retrain the SpecPT autoencoder from scratch on regridded sim data (`grism_training_sim_v3_regrid.parquet`) with reduced transformer capacity, then freeze it and train a redshift head on top. Each AE→redshift pair is chained with SLURM `afterok` on **tigris** (both AE and redshift run on GH200; dependencies can't span clusters). Grouped split by TARGETID prevents leakage. NOTE: transformer capacity is only a small slice of total AE size — the decoder `linear2` (~970M params) dominates, so total params only shrink 1.12B→1.00B across the three configs. If no real-data transfer gain, the next test is a decoder bottleneck (Track B).

> ℹ️ **Grid alignment.** The simulation data has been regridded from 10311–17465 Å to the real 3D-HST grid (10800–17100 Å, 0.81 Å/pix). This eliminates the 187-pixel feature shift that caused catastrophic real-data eval failure. See [`scripts/prep_sim_regridded.py`](scripts/prep_sim_regridded.py) and track [`EXPERIMENTS.md`](EXPERIMENTS.md) for details.

> ℹ️ **Outlier Analysis.** See [`notebooks/02_outlier_analysis.ipynb`](notebooks/02_outlier_analysis.ipynb) for deep-dive analysis of what makes outlier spectra different. All checkpoints now saved with experiment-specific names (`exp_NNN_best_model.pth`) to prevent cross-experiment contamination.

> ⚠️ **Autoencoder frozen (frozen runs).** Experiments exp_001 and exp_002 failed because they modified the autoencoder architecture. The autoencoder is a pretrained, frozen model by default. New `freeze_backbone: false` option allows end-to-end training (exp_034).

---

## 🧪 Real 3D-HST Evaluation

Transfer learning from regridded sim autoencoder backbone to real 3D-HST grism data. The regridded autoencoder reconstructs real spectra well (97% cosine similarity), but redshift prediction remains challenging due to the domain gap.

### Summary

Key finding: **All six axes of downstream methods on the frozen AE encoder are exhausted.** No head architecture, loss function, tree method, contrastive learning, or safe unfreezing technique has produced test R² > 0 or NMAD < 0.20 on the *final 512-d latent*. The FUSE program then showed a partial way out: a z-supervised early conv-map branch extracts a modest additional real-z signal (h_z probe 0.202, R² +0.08) beyond the 512-d latent — the projection discards some z — but the direct head shrinks (prediction collapse) and unfreezing with anchors doesn't reorganize the encoder. The encoder-level change is still the bottleneck.

| Exp | Head | Backbone | Augment | Test NMAD | Test η | Best Ep | Early Stop |
|-----|------|----------|---------|-----------|--------|---------|------------|
| **exp_045_RF_fixed** | **Random Forest** | **Frozen** | **No** | **0.20767** | **49.82%** | **n/a** | **n/a** |
| exp_051_RNC_unfrozen_v2 | RNC (unfrozen, LR 1e-5) | Unfrozen | No | 0.23935 | 52.24% | 1 | ❌ AE drift 126x |
| exp_050_RNC_frozen_v2 | RNC (frozen) | Frozen | No | 0.24224 | 52.24% | 1 | ❌ RNC flat |
| exp_035 | Linear (simple) | Frozen | No | 0.24883 | 54.64% | 19 | 49 |
| exp_041 | MLP (3-layer 512→256→128) | Frozen | No | 0.26045 | 56.61% | 292 | 300 (none) |
| exp_047_huber_linear | Linear (simple, HuberNMADLoss) | Frozen | No | 0.26528 | 57.77% | 12 | 42 |
| exp_042 | ResNet (3 residual blocks) | Frozen | No | 0.26646 | 57.69% | 32 | 62 |
| exp_040 | Simple (2-layer) | Unfrozen | No | 0.27395 | 58.55% | 1 | 31 |
| exp_043 | Metric Learning (NTXent + k-NN) | Frozen | No | 0.27399 | 58.35% | 1 | 31 |
| exp_039 | Simple (2-layer) | Unfrozen | No | 0.27547 | 58.80% | 1 | 31 |
| exp_037 | Enhanced (5 blocks) | Unfrozen | No | 0.28494 | 59.85% | 1 | 31 |
| exp_038 | Enhanced (5 blocks) | Unfrozen | Yes | 0.32759 | 64.27% | 58 | 88 |
| exp_036 | Linear (simple) | Frozen | Yes | 0.33644 | 67.27% | 1 | 31 |

### Key Learnings

1. **Frozen backbone (linear probe) works best.** Test NMAD 0.249 vs 0.274+ for unfrozen approaches.
2. **Augmentation hurts.** Both frozen+augment (0.336) and unfrozen+augment (0.328) are worse than baseline.
3. **End-to-end training overfits.** Training loss drops to 0.52 while val NMAD degrades from 0.24 → 0.28.
4. **Simple and complex heads perform similarly** when backbone is unfrozen (NMAD 0.27-0.28 across all head types).
5. **Head architecture on frozen features is exhausted.** MLP (exp_041 NMAD 0.260), ResNet (exp_042 NMAD 0.266), and metric learning (exp_043 NMAD 0.274) all lose to the simple linear probe (exp_035 NMAD 0.249). The encoder is the bottleneck.
6. **Metric learning (NTXent + k-NN) never converged.** Train loss 1.67, best epoch 1 — the contrastive embedding objective does not extract redshift-discriminative structure from the frozen encoder's 512-d latent space.
7. **exp_044 (RF) metrics were invalidated by a shape bug.** `y` arrays saved as 2-d `(n, 1)` from HSTGrismDataset, passed to `compute_metrics` where broadcasting created `n×n` pairwise residual matrices. Fixed and re-run as exp_045_RF_fixed.
8. **RF beats linear probe on frozen features — but for the wrong reason.** exp_045_RF_fixed achieves NMAD 0.2077 (16.5% improvement) but predictions are shrunken to [0.73, 1.80] vs true [0.01, 3.47]. Test R² = 0.094 — predictions don't track z-variance. The improvement is from RF's implicit Bayesian shrinkage (variance reduction), not from learning non-linear z-discriminative structure. This partially reopens the head-architecture axis for tree-based methods but the domain gap remains the dominant bottleneck.
9. **MHA is decorative on real data.** exp_046_pre_attn_RF (pre-attention RF, NMAD 0.20844) within 0.4% of exp_045 (post-attention, 0.20767). The 3-layer transformer_encoder adds zero discriminative value for real-data redshift prediction.
10. **Catastrophic η is a low-SNR tail problem.** B3 SNR breakdown from exp_046: SNR<5 (45% of test) → NMAD 0.287 η 65%; SNR 5-10 → 0.180 η 48%; SNR 10-20 → 0.099 η 29%; SNR 20+ → 0.082 η 26%. The 49.8% global η is driven by low-SNR samples where there is fundamentally no signal to predict redshift.
11. **Loss-space/metric-space mismatch is NOT the bottleneck.** exp_047 — HuberNMADLoss (δ=0.15) with all exp_035 hyperparams frozen → test NMAD 0.265 (WORSE than exp_035 NMADLoss at 0.249). The coordinate-space mismatch in NMADLoss (raw z vs normalized (z-ẑ)/(1+z)) is a theoretical concern but not what's holding back real-data performance. The bottleneck remains in the encoder, not the loss. <sup>exp_047</sup>
12. **Encoder unfreezing for NMADLoss regression destroys real utility regardless of data volume.** exp_048 (22:1 sim:real) and exp_048b (4.7:1 sim:real) both produce identical failure: best epoch=1, test NMAD ~0.27, recon MSE 3.4x drift. Reducing sim volume 5× changed nothing. The AE-pretrained encoder is fragile to gradient-driven modification from regression loss, at LR as low as 1e-5.
13. **RNC cannot bootstrap z-ordering from frozen features.** exp_050_RNC_frozen tested twice (T=2.0+L2-norm; T=0.5+raw features). Both flat: train loss 4.584→4.577 (-0.15%) over 53 epochs. RNC gradient requires z-discriminative features at projection init — which the frozen AE encoder does not produce. Test NMAD 0.242 = random init baseline.
14. **RNC unfrozen encoder drift is catastrophic (126×), not subtle.** exp_051_RNC_unfrozen: RNC loss flat (same 4.577 plateau) despite encoder LR 1e-5. Recon MSE drifted 0.008→1.039 (126× over 66 ep). The RNC gradient is noise — it flows back to the encoder but in random directions, shredding AE identity without organizing z-discrimination. Far worse than NMADLoss regression (3.4× drift).

### Current Work

**FUSE (Factorized Universal Encoder) program — complete.** Shared `h_universal` (reconstruction-anchored, reusable for future SFR/AGN heads) + task-specific early `h_z` branch supervised by redshift. **Stage 2 established the key result**: early conv-map features carry a small but real z-signal beyond the final 512-d latent (h_z linear probe **0.202 / R² +0.08** vs h_universal **0.215 / R² −0.04**). Direct z_head 0.184 is shrinkage (pred collapse std_ratio 0.33, R² −0.24) — never trust a z_head NMAD without range/R² checks (exp_045 lesson). **Stage 3** proved recon+distill anchors enable a *controlled* unfreeze (drift 0.92, cos 0.998 — no AE shredding, unlike exp_048/051) but the encoder stays too static to reorganize, and sim-dominated joint batches degrade real z (sim z doesn't transfer). See [`docs/factorized_universal_encoder.md`](docs/factorized_universal_encoder.md). All runs on **tigris** (GH200, ~20× faster than sporc).

**New sim data release — `simv4a`.** `data/training_format/grism_training_simv4a.parquet` (100k spectra, 5.5 GB): flat pre-selected HDF5, regridded 10800–17100 Å (matches real 3D-HST grid), z uniform over 0–4 (~26k/bin), catalog SNR 2.5–1579, plus new columns not used by the training pipeline yet (`selection_weight`, `snr_stratum`, line fluxes/errors, SED params). Prepared via `scripts/prep_sim_regridded.py --flat --snr-column catalog_snr --no-filter --passthrough-1d`. Grouped split by TARGETID (all 100k unique) prevents leakage.

**exp_052 — scratch end-to-end on simv4a (NEGATIVE, published).** Trained the full 1.1B SpecPT encoder + redshift head from random init directly on the redshift objective (NMADLoss, lr 1e-4, 400 ep, no pretrained AE). **Catastrophically overfit**: test NMAD **0.39042** (~50× worse than frozen-AE best 0.00785), train loss fell 1.18→0.187 while val loss *climbed* 0.77→0.85, best val NMAD 0.368 at ep 11, early stop ep 60. The 1.1B model has no reconstruction objective to shape the latent, so it memorizes the 80k training spectra. **This definitively confirms pretraining is essential** — the frozen autoencoder supplies the z-discriminative latent; end-to-end redshift-only training from scratch is ruled out.

**exp_053 — frozen regridded AE + exp_032 head on simv4a (NEGATIVE, published).** The data-lever test: identical config to the 0.00785 best (frozen AE, 12-block/1024 head, lr 1e-4, grouped split) but on the richer simv4a data. **Also catastrophically collapsed**: test NMAD **0.39077**, η 67.53%, early stop ep 75 — statistically indistinguishable from exp_052's scratch 0.39042. The pretrained AE latent is NOT z-discriminative on simv4a; the head overfits train (loss 1.19→0.22) while val loss climbs. **simv4a does not transfer to the redshift head in either setup** — this points at the v4a data pipeline (label alignment, spectrum construction, normalization), not the model.

**autoencoder_simv4a — continued AE pretraining (COMPLETED).** Recon-only adaptation of the regridded AE on simv4a (lr 1e-5, 200 ep): test recon NMAD 0.0403, cos 0.99797, 0 NaN. Passes the reconstruction gate but that is NOT the acceptance criterion.

**exp_054 — frozen simv4a-adapted AE + exp_032 head on simv4a (NEGATIVE, published).** The third consecutive simv4a failure (with exp_052 scratch and exp_053 regridded): test NMAD **0.3939**, η 67.45%, best val NMAD 0.4367 @ ep71. AE adaptation on simv4a does NOT recover the redshift signal. **Verdict: stop simv4a redshift-head work; investigate the v4a pipeline** (`scripts/prep_sim_regridded.py --flat --snr-column catalog_snr`) — label alignment / spectrum construction / normalization.

**exp_056 — active.** The missing control: **exact exp_032 config** (DESI combined AE, 12×1024 head, lr 1e-4, bs 128, 400 ep, pat 50, wd 5e-5, **zscore** normalization, **random** split) run on **simv4a** — only `data.path` differs. exp_053/054 differed from exp_032 in AE (regridded/adapted vs DESI) + split (grouped vs random) — note the `normalization` config key was never read by train.py, so all runs used z-score at load time. If exp_056 also collapses to ~0.39, simv4a data itself is definitively the cause. Tigris job 83811.

**exp_055 — active.** First controlled test of the new **binned redshift head** (classification over uniform log(1+z) bins + per-bin sigmoid refinement, 24 bins over z≤4.0, λ_refine 0.3 / λ_nmad 0.7 / label_smoothing 0.05) on the known-good **v2_Q1** pipeline (exp_032 config: frozen DESI AE, 12×1024 head, lr 1e-4). Purpose: kill catastrophic outliers (exp_032 η = 15.17%, target <1%) while holding NMAD ≤0.016. Tigris job 83810.

### Next Experiments Under Consideration

| Priority | Experiment | Goal |
|----------|-----------|------|
| **High** | **exp_056: exact exp_032 config on simv4a (running)** | **Isolate the data file: if the byte-identical best config collapses to ~0.39 on simv4a, simv4a data is the cause — not AE/normalization/split.** |
| **High** | **exp_055: binned redshift head on v2_Q1 (running)** | **Kill catastrophic outliers (15.17% → <1%) while holding NMAD ≤0.016. First controlled test of the new binned head on known-good data.** |
| High | Investigate simv4a pipeline (exp_054 verdict): label alignment, spectrum construction, NaN/pad distribution, normalization stats | Find why simv4a gives NMAD 0.39 while v2_Q1/v3 give 0.0079–0.011 |
| High | Harden Stage-2 claim: multi-seed (2–3×) + bootstrap CI95 on ALL probe pathways + same-capacity h_universal control | Establish whether the 0.202-vs-0.215 bottleneck is real before more architecture work |
| High | Real-only controlled unfreeze (sim contributes recon only, never z) | Does encoder adaptation help real z at all, absent sim contamination? |
| High | Anti-shrinkage/calibration head on h_z (variance-preserving term, report probe NMAD as primary) | Convert the 0.184 shrinkage into genuine prediction spread (R² > 0) |
| Medium | Use v4a `selection_weight` as per-sample loss weighting | Fix low-SNR / high-z tail, which drives catastrophic outliers |
| Medium | Bigger multi-scale conv-map extractor (tap all 3 conv blocks + attention over 487 wavelength positions) | Recover more of the z signal the 512-d flatten discards |
| Medium | USE-Stage-C student as frozen backbone (swap for raw AE) | Cleaner low-SNR features via noise robustness (clean-vs-noisy cos 0.9995) |

---

## 🏆 Leaderboard (Synthetic Data)

| Rank | Experiment | NMAD | Outliers | Epochs | Notes |
|------|-----------|------|----------|--------|-------|
| 1 | `exp_032` | **0.00785** | **15.17%** | 325 | Q1 quality data + exp_013 config. Current best NMAD. |
| 2 | `exp_034` | **0.00909** | **12.73%** | 400 | **Best outlier rate ever (12.73% — 2.4 pp improvement).** Regridded data + unfrozen backbone. |
| 3 | `exp_033` | **0.01111** | **15.53%** | 257 | Regridded data (10800–17100 Å), frozen backbone. |
| 4 | `exp_013` | **0.01382** | **24.85%** | 354 | mlp_dim=1024, 12 blocks, DESI AE, residual fix. Previous best before exp_032. |
| 5 | `exp_031` | **0.01489** | **24.39%** | 344 | exp_013 + patience=150, epochs=600. |
| 6 | `exp_021` | **0.01506** | **24.07%** | 400 | patience 50→100. |
| 7 | `exp_014` | **0.01640** | **23.78%** | 295 | dropout_rate=0.1. |
| 8 | `exp_019` | **0.01670** | **24.70%** | 355 | batch_size 128→256. |
| 9 | `exp_023` | **0.016995** | **23.60%** | 502 | lr 1e-4→5e-5. |
| 10 | `exp_022` | **0.01712** | **23.99%** | 342 | epochs 400→600. |
| 11 | `exp_030` | **0.01816** | **23.84%** | 267 | Curriculum (50%→100%). |
| 12 | `exp_025` | **0.01848** | **23.85%** | 254 | TTA (N=10). |
| 13 | `exp_020` | **0.01909** | **23.55%** | 318 | warmup_epochs 500→50. |
| 14 | `exp_027` | **0.01934** | **24.25%** | 241 | Two-Stage (200+200, 4x outlier weight). |
| 15 | `exp_024` | **0.01950** | **23.30%** | 462 | weight_decay 5e-5→1e-4. |
| 16 | `exp_028` | **0.01987** | **23.79%** | 229 | Per-Sample Weights (inverse error by z-bin). |
| 17 | `exp_018` | **0.02062** | **23.74%** | 238 | mlp_dim 1024→768. |
| 18 | `exp_016` | **0.02100** | **24.04%** | 210 | weight_decay=1e-5. |
| 19 | `exp_017` | **0.02132** | **23.84%** | 220 | weight_decay=5e-4. |
| 20 | `exp_015` | **0.02156** | **23.37%** | 200 | dropout_rate=0.2. |
| 21 | `exp_008_v2` | **0.02295** | **23.31%** | 242 | Previous best before exp_013. |
| 22 | `exp_029` | **0.02539** | **24.26%** | 86 | MDN Head (K=5). Val loss diverged. |
| 23 | `exp_007` | **0.02565** | 23.61% | 257 | num_mlp_blocks 7→10. |
| 24 | `exp_008` | **0.02568** | 23.25% | 219 | Capacity saturating. |
| 25 | `exp_009` | **0.02611** | 23.07% | 154 | Higher LR worsened NMAD. |
| 26 | `exp_005` | **0.0279** | 23.18% | 249 | num_mlp_blocks 5→7. |
| 27 | `exp_000_baseline` | 0.0303 | 23.24% | 244 | Default config. |
| 28 | `exp_006` | 0.0332 | 23.98% | 193 | Regularization backfired. |
| 29 | `exp_004` | 0.0335 | 23.42% | — | lr 5e-5 — worse than baseline. |
| 30 | `exp_026` | **0.07718** | **30.36%** | 73 | HuberNMADLoss. 5.6x worse. Loss scale mismatch. |
| 31 | `exp_052` | 0.39042 | 68.09% | 60 | SCRATCH end-to-end on simv4a (no pretrained AE). Catastrophic overfit — confirms pretraining is essential. |
| 32 | `exp_053` | 0.39077 | 68.87% | 75 | Frozen regridded AE + exp_032 head on simv4a. Also catastrophic — simv4a does not transfer. |
| 33 | `exp_054` | 0.3939 | 67.45% | 71 | Frozen simv4a-adapted AE + exp_032 head on simv4a. Third simv4a failure — AE adaptation doesn't recover z-signal. |

> 📝 `exp_034` uses the regridded sim data (10800–17100 Å) with an **unfrozen** DESI autoencoder (end-to-end training). It set the all-time outlier record (12.73%) at the cost of a moderate NMAD increase (0.00785 → 0.00909). Despite grid alignment, real-data transfer remains challenging — see [Real 3D-HST Evaluation](#-real-3d-hst-evaluation) for transfer learning results.

*Last updated: 2026-08-12 UTC*

---

## ⚙️ How It Works

```
W&B run finishes
    ↓
watcher.py (polls every 60s)
    ↓
trigger.py → hermes chat -q --profile specpt-hst -s specpt-orchestrator
    │                 (skill loaded via -s, AGENTS.md embedded in skill)
    ├── terminal → Analyst      (deepseek-v4-pro)   — W&B analysis, returns JSON
    ├── terminal → Experimenter (deepseek-v4-pro)   — next config, returns JSON
    ├── 3.5 Verify experimenter side-effects (config file, EXPERIMENTS.md row)
    ├── delegate_task → Runner  (deepseek-v4-flash) — git + SSH + sbatch
    ├── 4.5 Verify runner side-effects (squeue, git log, jobs.csv)
    ├── delegate_task → Memory  (deepseek-v4-flash) — update state files
    └── 5.5 Verify memory side-effects (re-read all 4 state files)
    ↓
SLURM trains on cluster
    ↓
New W&B run appears → watcher detects → next cycle
```

### Retry Behavior
- **Runner failure** (SSH timeout, Duo MFA stall, sbatch error): retried every 5 min for up to 1 hour
- **Orchestrator crash** (OOM, process killed): logged to dead-letter queue
- **Analyst/Experimenter failure**: retried once, then skipped

---

## 📂 Project Structure

```
specpt-hst-sim/
├── .opencode/            OpenCode config + agent definitions
├── configs/              Experiment YAML configs (one per experiment)
├── daemon/               Watcher, trigger, webhook, W&B helpers
│   ├── logs/             Orchestrator cycle logs
│   └── *.py              Daemon scripts
├── docs/                 Technical documentation
├── notebooks/            Jupyter notebooks for analysis
│   └── 02_outlier_analysis.ipynb   Outlier spectra deep-dive
├── reports/              Analysis reports (HTML)
│   └── figures/          Generated plots from analysis
├── scripts/              SLURM training scripts
├── src/                  Source code (model, training, losses, dataloader)
├── EXPERIMENTS.md        Full experiment log
├── SOUL.md               Project identity + goals
└── README.md             You are here
```

---

## 🚦 System Status

| Component | Model | Status |
|-----------|-------|--------|
| Watcher | — | ⏸️ Idle (manual submission) |
| Orchestrator | deepseek-v4-flash | ⏸️ Idle |
| Analyst | deepseek-v4-pro | ⏸️ Idle |
| Experimenter | deepseek-v4-pro | ⏸️ Idle |
| Runner | deepseek-v4-flash | ⏸️ Idle |
| Memory | deepseek-v4-flash | ⏸️ Idle |
| SLURM Cluster | — | sporcsubmit.rc.rit.edu |
| W&B | — | ckb2084-rochester…/specpt-hst-sim |
| Dead-letter queue | — | 0 entries |

---

## 🔗 Links

- [W&B Project](https://wandb.ai/ckb2084-rochester-institute-of-technology/specpt-hst-sim)
- [Experiment Log](EXPERIMENTS.md)
- [Project Charter](SOUL.md)

---

*Built with [Hermes Agent](https://hermes.ai) · Driven by W&B · Trained on RC RIT SLURM*
