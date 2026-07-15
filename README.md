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
| NMAD | < 0.020 | **0.00785** (exp_032) | **✅ ACHIEVED** |
| NMAD Stretch | < 0.010 | **0.00785** (exp_032) | **✅ ACHIEVED** |
| Catastrophic outliers | < 1% | **15.17%** (exp_032) | **14.17%** |
| ECE | < 0.1 | — | — |

---

## 🔬 Active Experiments

| Experiment | Stage | Status | Notes |
|------------|-------|--------|-------|
| Autoencoder retrain | Regridded AE | ⏳ Running (job 21399914) | lr=1e-5, batch=64, NaN detection added |
| exp_034 | Unfrozen redshift, regridded | ⏳ Running (job 21399915) | lr=1e-5, batch=64, freeze_backbone=false |
| exp_033 | Frozen redshift, regridded | ✅ Complete (NMAD 0.01111) | Grid aligned to 10800–17100 Å, z-score norm |
| Real 3D-HST Stage 1 | Linear probe | ✅ Complete (NMAD 0.2105) | exp_032 head trained on 7.8k real spectra |
| Real 3D-HST Stage 2 | Partial freeze | ✅ Complete (NMAD 0.2073) | init from Stage 1 best, 40 epochs, early stopped at 10 |

> ℹ️ **Grid alignment.** The simulation data has been regridded from 10311–17465 Å to the real 3D-HST grid (10800–17100 Å, 0.81 Å/pix). This eliminates the 187-pixel feature shift that caused catastrophic real-data eval failure (NMAD ~0.50 → 0.2105 after grid-aware resampling + NaN masking). See [`scripts/prep_sim_regridded.py`](scripts/prep_sim_regridded.py) and track [`EXPERIMENTS.md`](EXPERIMENTS.md) for details.

> ℹ️ **Outlier Analysis.** See [`notebooks/02_outlier_analysis.ipynb`](notebooks/02_outlier_analysis.ipynb) for deep-dive analysis of what makes outlier spectra different. All checkpoints now saved with experiment-specific names (`exp_NNN_best_model.pth`) to prevent cross-experiment contamination.

> ⚠️ **Autoencoder frozen (frozen runs).** Experiments exp_001 and exp_002 failed because they modified the autoencoder architecture. The autoencoder is a pretrained, frozen model by default. New `freeze_backbone: false` option allows end-to-end training (exp_034).

---

## 🧪 Real 3D-HST Evaluation

Fine-tuning exp_032 on real 3D-HST grism data. Two-stage approach: linear probe → partial freeze. The new exp_033/exp_034 regridded checkpoints should improve transfer since the sim training grid now matches the real-data grid.

| Stage | Approach | Test NMAD | Eta | Status |
|-------|----------|-----------|-----|--------|
| Pre-fix | Frozen exp_032 (no input fix) | ~0.50 | — | Comparison baseline |
| **1** | **Linear probe (head only)** | **0.2105** | **47.6%** | ✅ Complete |
| 2 | Partial freeze (encoder + MLP + head) | 0.2073 | 47.3% | ✅ Complete |
| 3 | Retrain on regridded sim (exp_033) | 0.01111 (sim) | — | ⏳ Real-data eval pending |
| 4 | End-to-end unfrozen (exp_034) | — | — | ⏳ Running (job 21399915) |

---

## 🏆 Leaderboard (Synthetic Data)

| Rank | Experiment | NMAD | Outliers | Epochs | Notes |
|------|-----------|------|----------|--------|-------|
| 1 | `exp_032` | **0.00785** | **15.17%** | 325 | Q1 quality data + exp_013 config. Current best. |
| 2 | `exp_033` | **0.01111** | **15.53%** | 257 | Regridded data (10800–17100 Å), frozen backbone. 42% NMAD regression expected from grid shift. |
| 3 | `exp_013` | **0.01382** | **24.85%** | 354 | mlp_dim=1024, 12 blocks, DESI AE, residual fix. |
| 4 | `exp_031` | **0.01489** | **24.39%** | 344 | exp_013 + patience=150, epochs=600. |
| 5 | `exp_021` | **0.01506** | **24.07%** | 400 | patience 50→100. |
| 6 | `exp_014` | **0.01640** | **23.78%** | 295 | dropout_rate=0.1. |
| 7 | `exp_019` | **0.01670** | **24.70%** | 355 | batch_size 128→256. |
| 8 | `exp_023` | **0.016995** | **23.60%** | 502 | lr 1e-4→5e-5. |
| 9 | `exp_022` | **0.01712** | **23.99%** | 342 | epochs 400→600. |
| 10 | `exp_030` | **0.01816** | **23.84%** | 267 | Curriculum (50%→100%). |
| 11 | `exp_025` | **0.01848** | **23.85%** | 254 | TTA (N=10). |
| 12 | `exp_020` | **0.01909** | **23.55%** | 318 | warmup_epochs 500→50. |
| 13 | `exp_027` | **0.01934** | **24.25%** | 241 | Two-Stage (200+200, 4x outlier weight). |
| 14 | `exp_024` | **0.01950** | **23.30%** | 462 | weight_decay 5e-5→1e-4. |
| 15 | `exp_028` | **0.01987** | **23.79%** | 229 | Per-Sample Weights (inverse error by z-bin). |
| 16 | `exp_018` | **0.02062** | **23.74%** | 238 | mlp_dim 1024→768. |
| 17 | `exp_016` | **0.02100** | **24.04%** | 210 | weight_decay=1e-5. |
| 18 | `exp_017` | **0.02132** | **23.84%** | 220 | weight_decay=5e-4. |
| 19 | `exp_015` | **0.02156** | **23.37%** | 200 | dropout_rate=0.2. |
| 20 | `exp_008_v2` | **0.02295** | **23.31%** | 242 | Previous best before exp_013. |
| 21 | `exp_029` | **0.02539** | **24.26%** | 86 | MDN Head (K=5). Val loss diverged. |
| 22 | `exp_007` | **0.02565** | 23.61% | 257 | num_mlp_blocks 7→10. |
| 23 | `exp_008` | **0.02568** | 23.25% | 219 | Capacity saturating. |
| 24 | `exp_009` | **0.02611** | 23.07% | 154 | Higher LR worsened NMAD. |
| 25 | `exp_005` | **0.0279** | 23.18% | 249 | num_mlp_blocks 5→7. |
| 26 | `exp_000_baseline` | 0.0303 | 23.24% | 244 | Default config. |
| 27 | `exp_006` | 0.0332 | 23.98% | 193 | Regularization backfired. |
| 28 | `exp_004` | 0.0335 | 23.42% | — | lr 5e-5 — worse than baseline. |
| 29 | `exp_026` | **0.07718** | **30.36%** | 73 | HuberNMADLoss. 5.6x worse. Loss scale mismatch. |

> 📝 `exp_033` uses the regridded sim data (10800–17100 Å) with the original frozen DESI autoencoder. The 42% NMAD regression vs `exp_032` is expected — the frozen convs were trained on 0.92 Å/pix resolution and receive 0.81 Å/pix input after regridding. `exp_034` tests whether unfreezing the backbone recovers this gap.

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
├── .hermes/              Agent config + orchestrator workflow
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
