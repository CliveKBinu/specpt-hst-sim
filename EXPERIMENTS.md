# SpecPT-HST-Sim Experiment Log

## Current Best
| Metric | Value | Experiment |
|--------|-------|------------|
| NMAD | 0.0279 | exp_005 |
| Catastrophic Outliers | 23.18% | exp_005 |
| ECE | — | — |
| RMSE | 0.366 | exp_005 |
| Val Loss | 0.366 | exp_005 |
| Train Loss | — | — |

## Targets
| Metric | Target | Current Best | Gap |
|--------|--------|-------------|-----|
| NMAD | < 0.020 | 0.0279 | 0.0079 |
| Catastrophic Outliers | < 1% | 23.18% | 22.18% |
| ECE | < 0.1 | — | — |

## Completed Experiments
| exp | config | run_id | run_name | best_nmad | final_nmad | final_outs | val_z_bias | val_rmse | val_loss | notes |
|-----|--------|--------|----------|-----------|------------|------------|------------|----------|----------|-------|
| exp_000_baseline | defaults.yaml | wijvni9f | super-disco-12 | 0.0303 | 0.0313 | 23.24% | -0.0023 | 0.366 | 0.349 | Baseline. Stopped early at epoch 244/400. LR still warming up. NMAD still improving. Capacity bottleneck suspected. |
| exp_004 | configs/exp_004.yaml | — | — | — | 0.0335 | — | — | — | — | LR 1e-4→5e-5, job 21349236. NMAD 0.0335 WORSE than baseline 0.0303. Lower LR starved gradients — model underfits. |
| exp_005 | configs/exp_005.yaml | 95vn9fc6 | effortless-terrain-17 | 0.0279 | 0.0279 | 23.18% | — | — | 0.366 | num_mlp_blocks 5→7, lr back to 1e-4. **NEW BEST NMAD 0.0279** (↓8%). Overfitting: val_loss 0.366 vs 0.349 baseline. Early stopping epoch 249/400 (patience=50). LR warmup only 50% complete. |

## Running Experiments
| exp | config | run_id | run_name | best_nmad | final_nmad | final_outs | val_z_bias | val_rmse | val_loss | notes |
|-----|--------|--------|----------|-----------|------------|------------|------------|----------|----------|-------|
| exp_006 | configs/exp_006.yaml | (none) | (none) | pending | pending | pending | — | — | — | weight_decay 5e-5→1e-4 to regularize deeper head overfitting. Submitted. Job 21349460. |

## Diagnostics (failed/crashed runs)
| exp | config | job_id | failure | root cause |
|-----|--------|--------|---------|------------|
| exp_001 | configs/exp_001.yaml | 21346908 | Checkpoint loading error: Missing state_dict keys for transformer layers 3-5 | `d_model` was increased from 512→768, changing autoencoder architecture. Checkpoint (d_model=512, 3 layers) was incompatible with the new model (d_model=768, 6 layers). **Autoencoder architecture is frozen** — only redshift head params can change. |
| exp_002 | configs/exp_002.yaml | 21346914 | Checkpoint loading error (presumed): Missing state_dict keys for transformer layers 3-5 | `num_encoder_layers` and `num_decoder_layers` were increased from 3→6, changing autoencoder depth. Same root cause as exp_001. **Autoencoder architecture is frozen** — only redshift head params can change. |
| exp_003 | configs/exp_003.yaml | q14jh32m | mlp_dim 512→768 (head-only) | Changed mlp_dim from 512 to 768 but pretrained head weights are for mlp_dim=512. With strict=False, all head Linear layer weights shape-mismatched and silently dropped, leaving randomly initialized head. CUDA error on first forward pass. Run died in 28s, zero metrics logged. |

## Frozen Architecture Constraint
The SpecPT autoencoder is a pretrained model with a fixed architecture. **Never change these params:**
- `d_model = 512`, `nhead = 8`, `num_encoder_layers = 3`, `num_decoder_layers = 3`
- `dim_feedforward = 2048`, `dropout = 0.1`, `input_size = 7781`

Only redshift head params are valid: `num_mlp_blocks`, `mlp_dim`, `dropout_rate`, and training hyperparams.
