# SpecPT-HST-Sim — SOUL

## Identity
Autonomous optimization engine for SpecPT redshift estimation. Runs two parallel tracks: (1) synthetic HST grism simulation (target NMAD < 0.001, current best 0.00785 from exp_032), (2) real 3D-HST grism data transfer learning (target NMAD < 0.100, current best 0.24883 from exp_035).

## Mission
**Synthetic track (achieved):**
- Primary: NMAD < 0.001 (current: 0.00785, exp_032 — on hold)
- Secondary: Catastrophic outliers η < 1% (current: 12.73%, exp_034 — outlier improvement needed)
- Tertiary: Confidence calibration ECE < 0.1

**Real-data track (active):**
- Primary: NMAD < 0.100 (current: 0.24883, exp_035 frozen linear probe)
- Secondary: Catastrophic outliers η < 25% (current: 54.6%)
- Tertiary: Not tracked yet

## Current State (updated by agents)
- Last updated: 2026-07-20
- Active experiment: exp_045_RF_fixed (completed — see results below)
- Best NMAD (synthetic): 0.00785 (exp_032, Q1 quality data)
- Best NMAD (real-data): **0.20767 (exp_045_RF_fixed, Random Forest on frozen 512-d latents)**
- Total experiments completed: ~39 synthetic + 11 real-data = ~50 numeric exp_NNN launched
- Direction:
  - **RF beats linear probe by 16.5% NMAD (0.20767 vs 0.24883) and η drops from 54.6% to 49.8%.**
    - But improvement is from **implicit shrinkage**: RF predicts narrow [0.73, 1.80] vs true [0.01, 3.47]. Low-z and high-z spectra all predicted near the training mean.
    - Test R² = 0.094 → predictions don't track z-variation at all.
    - This is better than linear probe on NMAD but NOT because RF found non-linear z-discriminative structure. RF's bagging + deep trees acts as a smoother, avoiding extreme errors.
  - **Real-data head architecture axis: partially revived.** Non-parametric regression on the frozen 512-d latent does help — trees can still exploit residual structure linear probes miss. But the help is from variance reduction, not improved latent representation.
  - **Next experiments under consideration:**
    - B1 (pre-attention RF): is the MHA adding value or just decorative?
    - B3 (SNR-bucketed NMAD): where is the catastrophic η coming from? (49.8% is still terrible.)
    - D1 (sim+real joint training): domain gap remains the dominant lever.
    - ExtraTrees/GBDT: do other tree methods preserve or improve the RF's shrinkage benefit?
  - **exp_044 (RF) fixed** — shape bug diagnosed and fixed in exp_045_RF_fixed. Cached features reused.

## Frozen Architecture Constraints
The SpecPT autoencoder (conv layers + transformers) is pretrained and frozen by default. The `freeze_backbone: false` option exists but has caused overfitting on real data (exp_037-040).
**Never change these params** — they break checkpoint loading:
- input_size = 7781, d_model = 512, nhead = 8
- num_encoder_layers = 3, num_decoder_layers = 3
- dim_feedforward = 2048, dropout = 0.1

Only the redshift estimator head can be modified: num_mlp_blocks, mlp_dim, dropout_rate, and training hyperparams (lr, batch_size, epochs, patience, weight_decay). For the real-data track, the regridded backbone is expected (10800–17100 Å grid).

## Operating Rules
1. One experiment at a time (one SLURM job in sbatch queue)
2. One change per experiment (controlled variables)
3. Log EVERY experiment to EXPERIMENTS.md with reasoning
4. If job crashes → diagnose → fix → retry (max 3x per config)
5. If metric improves → push in same direction
6. If metric degrades → reverse or pivot to alternative
7. Never stop the loop without explicit user permission
8. **State files:** On every experiment update (submission, completion, failure), update all four canonical state files in order: EXPERIMENTS.md → jobs.csv → SOUL.md → README.md.
9. **Real-data specific:** Never unfreeze the encoder on real data without a stronger regularizer than 8,892 samples alone (joint sim loss or domain-adaptation loss required — lesson from exp_037-040).

## Agent Chain
watcher → trigger → hermes chat -q --profile specpt-hst -s specpt-orchestrator
    ├── terminal → Analyst (Pro)
    ├── terminal → Experimenter (Pro)
    ├── verify: config + EXPERIMENTS.md
    ├── delegate_task → Runner (flash)
    ├── verify: squeue + git log + jobs.csv
    ├── delegate_task → Memory (flash)
    └── verify: re-read all 4 state files

## Forbidden
- Never submit without writing EXPERIMENTS.md entry first
- Never ignore a crashed job — always diagnose
- Never change more than one thing per experiment
- Never unfreeze encoder without a regularizer (real-data track lesson)
- Never Stop without user permission
- Never use sklearn on 2-d target arrays without ravel() (exp_044 lesson — shape mismatches produce silent invalid metrics)
