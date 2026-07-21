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
- Active experiment: exp_045_RF_fixed (Random Forest on frozen encoder 512-d latents, re-run after exp_044 shape bug fix)
- Best NMAD (synthetic): 0.00785 (exp_032, Q1 quality data)
- Best NMAD (real-data): 0.24883 (exp_035, frozen linear probe, regridded backbone)
- Total experiments completed: ~39 synthetic + 10 real-data = ~49 numeric exp_NNN was launched
- Direction:
  - **Real-data head architecture axis exhausted.** Exp_035-043 swept frozen linear/MLP/ResNet/metric and end-to-end unfrozen. All head variants land in 0.249–0.284 NMAD. Best result = frozen linear probe (exp_035, NMAD 0.249). Metric learning never converged. Augmentation degrades everything by ~35%.
  - **exp_044 (RF) failed** due to shape bug (y arrays 2-d; compute_metrics broadcast to pairwise matrix). Train R²=0.87 / Test R²=0.09 (valid — the overfit signal is damning). Fixed re-run = exp_045_RF_fixed using cached features.
  - **Domain gap is the bottleneck.** The encoder's 512-d latent preserves reconstruction info (97% cosine on real data) but does not organize z-discriminatively — entangled features, no pressure in pretraining.
  - **Next pivot under consideration:** sim+real joint training (D1) — unfreezes the encoder using 100k sim+8.9k real labels together. Self-supervised pretraining on unlabeled real data (D4). Redshift-contrastive autoencoder (F1/F2).
  - **Diagnostic experiments deferred:** B1 (pre-attention RF), B2 (per-layer transformer taps), B3 (SNR-bucketed NMAD). To run after exp_045_RF_fixed result.

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
