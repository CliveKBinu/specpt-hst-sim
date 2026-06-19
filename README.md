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
| NMAD | < 0.020 | **0.01382** (exp_013) | **✅ ACHIEVED** |
| Catastrophic outliers | < 1% | **22.86%** (exp_008_v2 ep240; final 23.31%) | **21.86%** |
| ECE | < 0.1 | — | — |

---

## 🔬 Active Experiments

| # | Name | Change | Status | NMAD | Outliers | Last Update |
|---|------|--------|--------|------|----------|-------------|
| exp_020 | warmup_epochs 500→50, batch=256, mlp_dim=1024, 12 blocks | Fix LR warmup starvation (warmup_epochs 500→50) | 📤 **Submitting…** | — | — | 2026-06-19 |

> ℹ️ **exp_019** (batch_size 128→256) finished: NMAD 0.0167 — 19% better than exp_018, but 500-epoch warmup starved the model. LR only 7.1e-5 at ep355, never reached full 1e-4. exp_020 shortens warmup to 50 epochs.

> ⚠️ **Autoencoder frozen.** Experiments exp_001 and exp_002 failed because they modified the autoencoder architecture (d_model / num_layers). The autoencoder is a pretrained, frozen model. Only the redshift estimator head (num_mlp_blocks, mlp_dim, dropout_rate, training params) can be changed. See diagnostics in [`EXPERIMENTS.md`](EXPERIMENTS.md) for details.

---

## 🏆 Leaderboard

| Rank | Experiment | NMAD | Outliers | Epochs | Notes |
|------|-----------|------|----------|--------|-------|
| 1 | `exp_013` | **0.01382** | **24.85%** | 354 | mlp_dim=1024, 12 blocks, DESI AE, residual fix. **NEW BEST NMAD.** (+39.8% over exp_008_v2) |
| 2 | `exp_014` | **0.01640** | **23.78%** | 295 | dropout_rate=0.1. NMAD worsened 0.01382→0.01640. Overfitting gap persists at 9.6x. Dropout 0.1 too weak. |
| 3 | `exp_018` | **0.02062** | **23.74%** | 238 | mlp_dim 1024→768. NMAD degraded 49% (0.01382→0.02062). 5th consecutive post-exp_013 degradation. |
| 4 | `exp_016` | **0.02100** | **24.04%** | 210 | weight_decay=1e-5 worsened NMAD 52% (0.01382→0.02100). Overfitting gap 9.1x unchanged. |
| 5 | `exp_017` | **0.02132** | **23.84%** | 220 | weight_decay=5e-4. NMAD worsened 54% (0.01382→0.02132). Both wd directions degraded. wd=5e-5 sweet spot. |
| 6 | `exp_015` | **0.02156** | **23.37%** | 200 | dropout_rate=0.2. NMAD worsened to 0.02156 (56% worse than 0.01382 best). Monotonic degradation. |
| 7 | `exp_008_v2` | **0.02295** | **23.31%** | 242 | num_mlp_blocks=12 + DESI autoencoder. Previous best. |
| 8 | `exp_007` | **0.02565** | 23.61% | 257 | num_mlp_blocks 7→10, weight_decay 5e-5 |
| 9 | `exp_008` | **0.02568** | 23.25% | 219 | num_mlp_blocks 10→12. Capacity saturating. |
| 10 | `exp_009` | **0.02611** | 23.07% | 154 | lr 2e-4, DESI autoencoder. Higher LR worsened NMAD. |
| 11 | `exp_005` | **0.0279** | 23.18% | 249 | num_mlp_blocks 5→7, lr 1e-4 |
| 12 | `exp_000_baseline` | 0.0303 | 23.24% | 244 | Default config |
| 13 | `exp_006` | 0.0332 | 23.98%† | 193 | weight_decay 1e-4 — regularization backfired |
| 14 | `exp_004` | 0.0335 | 23.42% | — | lr 5e-5 — worse than baseline |

*\*exp_008_v2 best catastrophic outliers was 22.86% at epoch 240 (final: 23.31%)*
*†exp_006 final catastrophic outliers from W&B: 24.50% (best: 23.98%)*

*Last updated by orchestrator (hermes) at 2026-06-19 08:10 UTC*

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
├── scripts/              SLURM training scripts
├── EXPERIMENTS.md        Full experiment log
├── SOUL.md               Project identity + goals
└── README.md             You are here
```

---

## 🚦 System Status

| Component | Model | Status |
|-----------|-------|--------|
| Watcher | — | Active (60s polling) |
| Orchestrator | deepseek-v4-flash | Active |
| Analyst | deepseek-v4-pro | Active |
| Experimenter | deepseek-v4-pro | Active |
| Runner | deepseek-v4-flash | Active |
| Memory | deepseek-v4-flash | Active |
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
