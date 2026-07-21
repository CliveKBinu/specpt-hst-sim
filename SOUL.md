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
- Active experiment: **exp_047_huber_linear (loss-function ablation)**
- Best NMAD (synthetic): 0.00785 (exp_032, Q1 quality data)
- Best NMAD (real-data): **0.20767 (exp_045_RF_fixed, Random Forest on frozen post-attention 512-d latents)**
- Total experiments completed: ~39 synthetic + 12 real-data = ~51 numeric exp_NNN launched
- Direction:
  - **B1 completed: MHA is decorative.** Pre-attention RF (NMAD 0.20844) within 0.4% of post-attention RF (0.20767). The 3-layer transformer_encoder adds zero z-discriminative value on real data.
  - **B3 completed: Catastrophic η is primarily low-SNR-driven.** Per-bucket NMAD: <5→0.287 η 65%, 5-10→0.180 η 48%, 10-20→0.099 η 29%, 20+→0.082 η 26%. 45% of the test set is in SNR<5 (the catastrophic bucket). For SNR≥10, η is 28% — much more reasonable. This is a fundamental signal-to-noise problem, not a head-architecture problem.
  - **Loss-coordinate mismatch identified.** NMADLoss (`src/specpt/losses.py:11-22`) optimizes |z_pred − z_true| in raw z space, but the metric uses normalized residual r = Δz/(1+z). NMADLoss assigns identical cost to catastrophic low-z errors and acceptable high-z errors. HuberNMADLoss (already implemented at losses.py:25-58) fixes this — operates in normalized residual space with δ=0.15 aligned to the η threshold.
  - **exp_047 (HuberNMAD ablation) in flight.** Frozen linear probe, exp_035 hyperparams, delta=huber_nmad loss. Isolates whether the loss-function mismatch is a real lever or theoretical artifact. ~30 min wallclock.
  - **Next after exp_047:** If loss helps (>10% improvement), use HuberNMAD in Path A/B. If not, jump to Path A (D1 sim+real joint training) with NMADLoss unchanged.
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
