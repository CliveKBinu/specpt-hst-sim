# SpecPT-HST-Sim Experiment Log

## Current Best
| Metric | Value | Experiment |
|--------|-------|------------|
| NMAD | 0.0303 | exp_000_baseline |
| Catastrophic Outliers | 23.24% | exp_000_baseline |
| ECE | — | — |
| RMSE | 0.366 | exp_000_baseline |
| Val Loss | 0.349 | exp_000_baseline |
| Train Loss | 0.178 | exp_000_baseline |

## Targets
| Metric | Target | Current Best | Gap |
|--------|--------|-------------|-----|
| NMAD | < 0.020 | 0.0303 | 0.0103 |
| Catastrophic Outliers | < 1% | 23.24% | 22.24% |
| ECE | < 0.1 | — | — |

## Completed Experiments
| exp | config | run_id | run_name | best_nmad | final_nmad | final_outs | val_z_bias | val_rmse | val_loss | notes |
|-----|--------|--------|----------|-----------|------------|------------|------------|----------|----------|-------|
| exp_000_baseline | defaults.yaml | wijvni9f | super-disco-12 | 0.0303 | 0.0313 | 23.24% | -0.0023 | 0.366 | 0.349 | Baseline. Stopped early at epoch 244/400. LR still warming up. NMAD still improving. Capacity bottleneck suspected. |

## Running Experiments
| exp | config | run_id | run_name | best_nmad | final_nmad | final_outs | val_z_bias | val_rmse | val_loss | notes |
|-----|--------|--------|----------|-----------|------------|------------|------------|----------|----------|-------|
| exp_004 | configs/exp_004.yaml | — | — | — | — | — | — | — | — | LR 1e-4→5e-5 for stable convergence. Addresses early plateau and train-val gap from exp_000. Submitted. Job 21349236. |

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
