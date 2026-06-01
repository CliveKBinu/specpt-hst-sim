# SpecPT-HST-Sim — Implementation Plan

## Overview

Autonomous SpecPT training on simulated HST grism data. A continuous optimization loop that:
1. Trains SpecPT on RIT SLURM cluster
2. Automatically analyzes results via W&B
3. Generates hypotheses and submits next experiment
4. Loops indefinitely until goal is reached

## System Architecture

```
watcher.py (your PC, polls every 30 min)
    │
    ▼ Checks W&B API for finished runs
    │
    ▼ If found: opencode --agent specpt-orchestrator
    │
    ├──► specpt-analyst          (read W&B metrics)
    ├──► specpt-experimenter     (write next config)
    ├──► specpt-runner           (SSH → sbatch)
    └──► specpt-memory           (update state)
    │
    ▼
New job submitted → W&B → watcher detects → loop (continuous)
```

## Configuration

| Item | Value |
|---|---|
| **GitHub repo** | `CliveKBinu/specpt-hst-sim` |
| **W&B entity** | `ckb2084-rochester-institute-of-technology` |
| **W&B project** | `specpt-hst-sim` |
| **SSH host** | `sporcsubmit.rc.rit.edu` (user: ckb2084) |
| **Training data** | Simulated HST grism data (port from HST_GRISM_Sim) |

## Model Assignments (OpenCode Go)

All agents use the **OpenCode Go** provider: `opencode-go/<model-id>` ($5 first month, $10/month, includes 3 models: DeepSeek V4 Pro, DeepSeek V4 Flash, MiniMax M2.5).

| Agent | Model | Reasoning Demand | Cost (requests/5h) | Rationale |
|---|---|---|---|---|
| **Orchestrator** | `opencode-go/deepseek-v4-flash` | Low | 31,650 | Coordination only — calls subagents, no heavy reasoning |
| **Analyst** | `opencode-go/deepseek-v4-pro` | High | 3,450 | Pattern analysis on W&B metrics needs strongest reasoning |
| **Experimenter** | `opencode-go/deepseek-v4-pro` | High | 3,450 | Hypothesis generation is highest-reasoning task |
| **Runner** | `opencode-go/minimax-m2.5` | Medium | 6,300 | SSH + git tool calling, reliable execution |
| **Memory** | `opencode-go/deepseek-v4-flash` | Low | 31,650 | State writes, no heavy reasoning needed |

**Cost per cycle** (1 full optimization loop):
- 2 × V4 Pro calls (analyst + experimenter) — most expensive
- 2 × V4 Flash calls (orchestrator + memory) — cheapest
- 1 × M2.5 call (runner) — middle
- V4 Flash is ~9x cheaper per request than V4 Pro
- MiMo-V2.5 was considered for memory but V4 Flash gives unified cheap tier with orchestrator

**Go Limits:**
- 5-hour: $12, Weekly: $30, Monthly: $60
- V4 Pro: ~3,450 requests/5h — ample for 1-2 calls per cycle
- V4 Flash: ~31,650 requests/5h — ample for both orchestration + memory

## Directory Structure

```
F:\personal_projects\specpt-hst-sim\
│
├── .opencode/
│   ├── agents/
│   │   ├── specpt-orchestrator.md   # Main loop coordinator
│   │   ├── specpt-experimenter.md   # Proposes hypotheses, writes configs
│   │   ├── specpt-analyst.md         # Reads W&B, logs patterns
│   │   ├── specpt-runner.md          # SSH → sbatch
│   │   └── specpt-memory.md          # Updates state + SOUL.md
│   └── opencode.json
│
├── daemon/
│   ├── webhook_server.py           # Flask server (optional, for Pro plan)
│   ├── trigger.py                  # Manual trigger entry
│   ├── watcher.py                  # W&B poller (primary trigger)
│   └── requirements.txt
│
├── src/specpt/
│   ├── __init__.py
│   ├── model.py                    # SpecPT + EnhancedSpecPTForRedshift
│   ├── losses.py                   # NMADLoss
│   ├── dataloader.py               # HST grism loader
│   └── training/
│       ├── train.py                # Main training loop (wandb.init)
│       └── eval.py                 # Evaluation utilities
│
├── configs/
│   ├── defaults.yaml               # Base config (merged by all experiments)
│   └── exp_000.yaml                # First experiment (baseline)
│
├── scripts/
│   ├── slurm_train.sh              # SLURM batch script
│   └── ssh_submit.sh               # SSH helper wrapper
│
├── outputs/
│   ├── logs/
│   └── err/
│
├── SOUL.md                         # Project identity (agent updates)
├── EXPERIMENTS.md                  # Auto-updated experiment log
├── jobs.csv                        # Job tracking
├── implementation.md               # This file — build plan
└── README.md
```

---

## Implementation Phases

### P0 — Repository Setup

1. Create directory structure
2. Create `.gitignore` (Python, venv, __pycache__, .env, outputs/)
3. Initialize git: `git init`, `git remote add origin https://github.com/ckb2084/specpt-hst-sim.git`
4. Create initial commit with directory structure
5. Push to GitHub

### P1 — Training Code Port

**Source files:**
- `F:\personal_projects\HST_GRISM_Sim\SpecPT_z.ipynb` (5652 lines)
- `F:\personal_projects\HST_GRISM_Sim\scripts\dataloader.py`

**Files to create:**

**`src/specpt/__init__.py`**
- Exports: SpecPT, EnhancedSpecPTForRedshift, NMADLoss, ImprovedResidualMLPBlock, Swish

**`src/specpt/model.py`**
- `Swish` activation function
- `ImprovedResidualMLPBlock` — MLP block with Swish + layernorm + residual
- `SpecPT` class (CNN + Transformer autoencoder) — encoder processes 7781-pixel spectra through conv layers + transformer encoder, decoder reconstructs via transformer decoder + conv transpose
- `EnhancedSpecPTForRedshift` class — MLP redshift head with attention over encoder output
- `SpectrumNormalizer` utilities:
  - median, minmax, zscore, robust, continuum, snr_weighted, log_transform, mad_normalize, wavelength_dependent

**`src/specpt/losses.py`**
- `NMADLoss` — Normalized Median Absolute Deviation as differentiable loss:
  ```python
  class NMADLoss(nn.Module):
      def __init__(self, eps=1e-8):
          super().__init__()
          self.eps = eps
      def forward(self, z_pred, z_true):
          mad = torch.mean(torch.abs(z_pred - z_true))
          normalization = torch.clamp(torch.std(z_true), min=self.eps)
          return mad / normalization
  ```
- Combined loss (NMAD + MSE, tunable ratio)

**`src/specpt/dataloader.py`**
- `HSTGrismDataset` class
- Load pkl file (grism_specPT_v5.pkl)
- Normalization: median normalize + handle NaN/inf
- Train/val/test split: 80/10/10
- Batch loading for 7781-pixel spectra
- Configurable normalization method

**`src/specpt/training/train.py`**
- CLI args: `--config`, `--wandb_entity`, `--wandb_project`, `--resume`
- `wandb.init(project=... , entity=..., config=...)`
- Model initialization: SpecPT → EnhancedSpecPTForRedshift
- Load pretrained weights (autoencoder + redshift head)
- Optimizer: AdamW
- Scheduler: ReduceLROnPlateau
- Training loop: epoch → forward → NMAD loss → backward → step
- Validation: compute NMAD, bias, catastrophic outlier rate
- Checkpoint: save best model to W&B artifact
- Early stopping with configurable patience
- Metrics logged: train_loss, val_nmad, val_z_bias, catastrophic_outliers, lr

**`src/specpt/training/eval.py`**
- Compute: NMAD, bias, catastrophic outliers (>5% error), ECE (expected calibration error)
- Generate: predictions vs truth scatter, histogram of residuals
- Export: metrics as JSON for agent consumption

### P2 — SLURM Script + SSH Helper

**`scripts/slurm_train.sh`**

```bash
#!/bin/bash -l
#SBATCH --job-name=SpecPT_Exp
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/err/%x_%j.err
#SBATCH --mail-user=slack:@ckb2084
#SBATCH --mail-type=ALL
#SBATCH --time=1-23:0:00
#SBATCH --account=redshift
#SBATCH --nodes=1
#SBATCH --partition=sporc
#SBATCH --mem=128g
#SBATCH --gres=gpu:h100:1

EXP_NAME=${1:-exp_000}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
spack load cuda@12.4.0 /obxqih4

cd ~/specpt-hst-sim
git pull origin main

python -m src.specpt.training.train \
    --config "configs/${EXP_NAME}.yaml" \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT"
```

**`scripts/ssh_submit.sh`**

```bash
#!/bin/bash
EXP_NAME=$1
ssh ckb2084@sporcsubmit.rc.rit.edu "cd ~/specpt-hst-sim && sbatch scripts/slurm_train.sh $EXP_NAME"
```

### P3 — Daemon (watcher.py)

**`daemon/watcher.py`** — Primary trigger, polls W&B every 30 minutes:
- Polls W&B for any finished runs since last check
- Triggers orchestrator if found
- State file tracks last check timestamp

**`daemon/webhook_server.py`** — Optional, for W&B Pro/Enterprise plan:
- Flask server on port 8001
- `POST /wandb` — receives run data, triggers opencode orchestrator
- `GET /health` — health check
- Requires Cloudflare tunnel + W&B webhook automation

**`daemon/trigger.py`** — Manual trigger for testing:
- Usage: `python trigger.py run_id state`
- Calls opencode orchestrator directly

### P4 — GitHub Webhook Relay (Optional, Pro plan only)

> **Note:** W&B webhook automations require Pro or Enterprise plan. Student plan uses watcher.py polling instead (P3).

**`.github/workflows/wandb-relay.yml`** — Not included by default. Create if upgrading to Pro.

**Setup steps (student plan — watcher.py):**
1. Run `python daemon/watcher.py` on your PC (keep running)
2. Watcher polls W&B every 30 minutes for finished runs
3. When found, triggers orchestrator automatically

**Setup steps (Pro plan — webhook, optional):**
1. GitHub repo → Settings → Secrets → Actions → New secret
   - Name: `LOCAL_AGENT_URL`
   - Value: `https://specpt-training.clivekbinu.me`
   - Name: `WEBHOOK_SECRET`
   - Value: `<generated secret>`
2. W&B dashboard → Team Settings → Secrets → New secret
   - Name: `GITHUB_PAT`
   - Value: `<your GitHub PAT>`
   - Name: `WEBHOOK_SECRET`
   - Value: `<same secret as above>`
3. W&B dashboard → Settings → Webhooks → Add webhook
   - Payload URL: `https://api.github.com/repos/CliveKBinu/specpt-hst-sim/dispatches`
   - Content type: `application/json`
   - Event: repository_dispatch, type: `wandb-webhook`
   - Payload: `{"runId": "{{run.id}}", "runName": "{{run.name}}", "state": "{{run.state}}"}`
4. Cloudflare tunnel: add Published application route for `specpt-training.clivekbinu.me` → `localhost:8001`

### P5 — The 5 Opencode Agents

Each agent is a `.md` file in `.opencode/agents/` with YAML frontmatter.

**1. `specpt-orchestrator.md`** — Main loop coordinator
- Triggered by daemon when W&B run finishes
- Reads SOUL.md, EXPERIMENTS.md, jobs.csv
- Calls analyst → experimenter → runner → memory in sequence
- Handles crashes, retries (max 3 per config)
- Environment variables: SPECPT_RUN_ID, SPECPT_RUN_NAME, SPECPT_RUN_STATE

**2. `specpt-analyst.md`** — W&B result analysis
- Reads SPECPT_RUN_ID from env
- Queries W&B API for run metrics (val_nmad, val_z_bias, catastrophic_outliers, train_loss)
- Compares to best run in EXPERIMENTS.md
- Identifies what improved/degraded and why
- Logs analysis to EXPERIMENTS.md

**3. `specpt-experimenter.md`** — Hypothesis generation
- Reads EXPERIMENTS.md (all history) + configs/defaults.yaml
- Identifies patterns: what improves NMAD, what causes outliers, where plateau occurs
- Generates next hypothesis (one change only):
  - Hyperparameter: lr, batch_size, dropout, weight_decay
  - Architecture: num_layers, d_model, num_mlp_blocks
  - Training: epochs, patience, lr_schedule
- Writes configs/exp_N.yaml with merged config + changes
- Strategy: push same direction if improving, reverse if degrading, pivot if plateaued

**4. `specpt-runner.md`** — SLURM job submission
- Reads the new config
- Git: branch, commit, push
- SSH `ckb2084@sporcsubmit.rc.rit.edu` → `cd ~/specpt-hst-sim && git pull && sbatch scripts/slurm_train.sh exp_N`
- Captures job_id, updates jobs.csv
- Reports job_id back to orchestrator

**5. `specpt-memory.md`** — State maintenance
- Updates SOUL.md with current experiment number, best NMAD, direction
- Writes to ctx_memory (category: SPECPT_HST_SIM_STATE)
- Updates jobs.csv: marks previous job as completed/failed
- Finalizes EXPERIMENTS.md entry

### P6 — SOUL.md + Templates

**`SOUL.md`** — Project identity document (updated by agents):

```markdown
# SpecPT-HST-Sim — SOUL

## Identity
Autonomous optimization engine for SpecPT redshift estimation on HST grism
simulated data. You exist to continuously improve model performance through
systematic experimentation.

## Mission
- Primary metric: NMAD — target < 0.020
- Secondary: Catastrophic outliers (>5% z error) — target < 1%
- Tertiary: Confidence calibration ECE — target < 0.1

## Current State (updated by agents)
- Active experiment: exp_000
- Best NMAD: (from EXPERIMENTS.md)
- Total experiments completed: 0

## Operating Rules
1. One experiment at a time (one SLURM job on cluster)
2. One change per experiment (controlled variables)
3. Log EVERY experiment to EXPERIMENTS.md with reasoning
4. If job crashes → diagnose → fix → retry (max 3x per config)
5. If metric improves → push in same direction
6. If metric degrades → reverse or pivot to alternative
7. Never stop the loop without explicit user permission

## Agent Chain
watcher → orchestrator → analyst → experimenter → runner → memory → loop

## Forbidden
- Never submit without writing EXPERIMENTS.md entry first
- Never ignore a crashed job — always diagnose
- Never change more than one thing per experiment
- Never stop without user permission
```

**`EXPERIMENTS.md`** — Auto-updated experiment log (template):
```markdown
# SpecPT-HST-Sim Experiment Log

## Running
| exp | job_id | run_id | status |
|-----|--------|--------|--------|

## Completed

---
```

**`jobs.csv`** — Job tracking (template):
```csv
exp_name,job_id,run_id,config_hash,status,val_nmad,val_z_bias,catastrophic_outliers,start_time,end_time
```

### P7 — Config Files

**`configs/defaults.yaml`**

```yaml
model:
  input_size: 7781
  d_model: 512
  nhead: 8
  num_encoder_layers: 3
  num_decoder_layers: 3
  dim_feedforward: 2048
  dropout: 0.1
  num_mlp_blocks: 5
  mlp_dim: 512

training:
  lr: 1.0e-4
  batch_size: 128
  epochs: 400
  patience: 50
  weight_decay: 5e-5
  lr_scheduler: ReduceLROnPlateau
  lr_scheduler_factor: 0.1
  lr_scheduler_patience: 20

data:
  path: /home/ckb2084/research/SpecPT/data/grism_specPT_v5.pkl
  pretrained_autoencoder: /home/ckb2084/research/galax_spec/saved_models/SpecPT_training_HST_augmented_autoencoder.pth
  pretrained_redshift: /home/ckb2084/research/galax_spec/pretrained_weights/SpecPT_DESI_Combined_EnhancedSpecPTForRedshift_500_new.pth
  normalization: median
  val_split: 0.1
  test_split: 0.1

wandb:
  entity: ckb2084-rochester-institute-of-technology
  project: specpt-hst-sim

expected_duration_hours: 4
```

**`configs/exp_000.yaml`**

```yaml
name: exp_000_baseline
description: >
  Baseline SpecPT training with default hyperparameters on HST grism simulated data.
  This establishes the starting point for all optimization experiments.
parent: defaults
changes: []
stage: baseline
```

### P8 — opencode.json + .gitignore + README

**`.opencode/opencode.json`**
```json
{
  "$schema": "https://opencode.ai/config.json",
  "agent": {
    "specpt-orchestrator": {
      "description": "Main SpecPT-HST-Sim training orchestrator",
      "mode": "primary",
      "model": "opencode-go/deepseek-v4-flash"
    },
    "specpt-analyst": {
      "description": "W&B run analysis",
      "mode": "subagent",
      "model": "opencode-go/deepseek-v4-pro"
    },
    "specpt-experimenter": {
      "description": "Hypothesis generation, config writing",
      "mode": "subagent",
      "model": "opencode-go/deepseek-v4-pro"
    },
    "specpt-runner": {
      "description": "SSH → SLURM submission",
      "mode": "subagent",
      "model": "opencode-go/minimax-m2.5"
    },
    "specpt-memory": {
      "description": "State maintenance, SOUL.md updates",
      "mode": "subagent",
      "model": "opencode-go/deepseek-v4-flash"
    }
  },
  "instructions": ["SOUL.md"]
}
```

**`.gitignore`**
```
__pycache__/
*.py[cod]
*$py.class
*.so
.env
venv/
*.egg-info/
dist/
build/
outputs/logs/*
outputs/err/*
!outputs/logs/.gitkeep
!outputs/err/.gitkeep
daemon/webhook.log
.DS_Store
thumbs.db
```

---

## Implementation Order

| Phase | Task | Files Created | Effort |
|---|---|---|---|
| P0 | Repo setup — git, .gitignore, structure | All dirs + .gitignore | 1h |
| P1 | Port training code from HST_GRISM_Sim | src/specpt/*.py | 2h |
| P2 | SLURM + SSH scripts | scripts/slurm_train.sh, ssh_submit.sh | 1h |
| P3 | Daemon (watcher.py) | daemon/*.py | 1h |
| P4 | GitHub webhook relay (optional) | .github/workflows/wandb-relay.yml | User action |
| P5 | 5 opencode agents | .opencode/agents/*.md | 3h |
| P6 | SOUL.md + experiment templates | SOUL.md, EXPERIMENTS.md, jobs.csv | 1h |
| P7 | Config files | configs/defaults.yaml, exp_000.yaml | 0.5h |
| P8 | opencode.json + README | .opencode/opencode.json, README.md | 0.5h |
| P9 | W&B integration | Student: watcher.py polling / Pro: webhook setup | User action |
| P10 | End-to-end test | Submit exp_000, verify full loop | 1h |

**Total: ~12 hours across 11 phases**

---

## Agent Chain (Per Optimization Loop)

1. `watcher.py` polls W&B API every 30 minutes
2. If finished run found → `subprocess.Popen(['opencode', '--agent', 'specpt-orchestrator', ...])`
3. Orchestrator loads state (SOUL.md, EXPERIMENTS.md, jobs.csv)
4. Orchestrator calls `specpt-analyst`:
   - Queries W&B for run metrics
   - Compares to best run
   - Appends analysis to EXPERIMENTS.md
5. Orchestrator calls `specpt-experimenter`:
   - Reads all experiments + defaults
   - Identifies patterns
   - Writes configs/exp_N.yaml
6. Orchestrator calls `specpt-runner`:
   - Git commit + push config
   - SSH `ckb2084@sporcsubmit.rc.rit.edu` → sbatch
   - Records job_id in jobs.csv
7. Orchestrator calls `specpt-memory`:
   - Updates SOUL.md (best NMAD, direction, exp count)
   - Writes ctx_memory
   - Finalizes EXPERIMENTS.md
8. New job on cluster → W&B → watcher detects → loop repeats

## Error Handling

- **Job crash**: analyst diagnoses → experimenter fixes → runner retries (max 3x)
- **SSH failure**: runner retries after 5 min → memory logs failure
- **Git conflict**: runner rebases → retry push
- **Docker/conda error**: analyst logs → user notified via slack
- **Daemon crash**: watcher.py can be restarted manually
- **Webhook miss**: watcher.py polls W&B every 30 min as primary trigger

## Success Criteria

- NMAD < 0.020 (primary goal)
- Catastrophic outlier rate < 1%
- Agent runs 10+ loops without human intervention
- EXPERIMENTS.md shows coherent optimization trajectory
- Every config change is a single controlled variable

---
