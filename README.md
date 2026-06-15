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
| NMAD | < 0.020 | 0.0303 | 0.0103 |
| Catastrophic outliers | < 1% | 23.24% | 22.24% |
| ECE | < 0.1 | (not logged) | — |

---

## 🔬 Active Experiments

| # | Name | Change | Status | NMAD | Outliers | Last Update |
|---|---|---|---|---|---|---|
| — | (pending) | — | awaiting next cycle | — | — | — |

> ⚠️ **Autoencoder frozen.** Experiments exp_001 and exp_002 failed because they modified the autoencoder architecture (d_model / num_layers). The autoencoder is a pretrained, frozen model. Only the redshift estimator head (num_mlp_blocks, mlp_dim, dropout_rate, training params) can be changed. See diagnostics in [`EXPERIMENTS.md`](EXPERIMENTS.md) for details.

---

## 🏆 Leaderboard

| Rank | Experiment | NMAD | Outliers | Epochs | Notes |
|------|-----------|------|----------|--------|-------|
| 1 | `exp_000_baseline` | 0.0303 | 23.24% | 244 | Default config |

*Last updated by orchestrator (hermes) at 2026-06-13 01:01 UTC*

---

## ⚙️ How It Works

```
W&B run finishes
    ↓
watcher.py (polls every 60s)
    ↓
trigger.py → hermes orchestrator
    ├── Analyst   (deepseek-v4-pro)  — reads W&B metrics, diagnoses issues
    ├── Experimenter (deepseek-v4-pro) — generates next experiment config
    ├── Runner    (deepseek-v4-flash) — commits, SSH to SLURM, submits job
    └── Memory    (deepseek-v4-flash) — updates state files, EXPERIMENTS.md, README
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
