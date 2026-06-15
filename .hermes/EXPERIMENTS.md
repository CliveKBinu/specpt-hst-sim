# SpecPT-HST-Sim Experiment Log

## Current Best
| Metric | Value | Experiment |
|--------|-------|------------|
| NMAD | **0.0279** | **exp_005** |
| Catastrophic Outliers | 23.18% | exp_005 |
| ECE | — | — |
| RMSE | 0.425 | exp_005 |
| Val Loss | 0.366 | exp_005 |
| Train Loss | 0.178 | exp_000_baseline |

## Targets
| Metric | Target | Current Best | Gap |
|--------|--------|-------------|-----|
| NMAD | < 0.020 | **0.0279** | **0.0079** |
| Catastrophic Outliers | < 1% | 23.18% | 22.18% |
| ECE | < 0.1 | — | — |

## Completed Experiments
| exp | config | run_id | run_name | best_nmad | final_nmad | final_outs | val_z_bias | val_rmse | val_loss | notes |
|-----|--------|--------|----------|-----------|------------|------------|------------|----------|----------|-------|
| exp_000_baseline | defaults.yaml | wijvni9f | super-disco-12 | 0.0303 | 0.0313 | 23.24% | -0.0023 | 0.366 | 0.349 | Baseline. Stopped early at epoch 244/400. LR still warming up. NMAD still improving. Capacity bottleneck suspected. |
| exp_004 | configs/exp_004.yaml | gx70j2j8 | stoic-spaceship-16 | 0.0335 | 0.0346 | 23.42% | -0.0023 | 0.382 | 0.365 | lr 1e-4→5e-5. NMAD WORSENED (0.0335 vs 0.0303 baseline). Lower LR starved gradients — model underfits. Capacity bottleneck confirmed. |
| exp_005 | configs/exp_005.yaml | 95vn9fc6 | effortless-terrain-17 | 0.0279 | 0.0279 | 23.18% | 0.0011 | 0.425 | 0.366 | num_mlp_blocks 5→7, lr 1e-4. NMAD IMPROVED 8% to 0.0279 (new best). Overfitting: val_loss rose, early stopping at epoch 249. NMAD still decreasing at termination. |

## Running Experiments
| exp | config | run_id | run_name | best_nmad | final_nmad | final_outs | val_z_bias | val_rmse | val_loss | notes |
|-----|--------|--------|----------|-----------|------------|------------|------------|----------|----------|-------|
| exp_001 | configs/exp_001.yaml | 21346908 | — | — | — | — | — | — | — | d_model 512→768. Capacity increase. Submitted. Job 21346908. |
| exp_002 | configs/exp_002.yaml | 21346914 | — | — | — | — | — | — | — | encoder/decoder depth 3→6. Complementary to exp_001 width increase. Submitted. Job 21346914. |
| exp_006 | configs/exp_006.yaml | (none) | (none) | pending | weight_decay 5e-5→1e-4 to regularize deeper head overfitting. Keeping num_mlp_blocks=7 (exp_005's winning architecture). |

## Diagnostics (failed/crashed runs)
| exp | run_name | run_id | failure | diagnosis |
|-----|----------|--------|---------|-----------|
| exp_001 | (unknown) | (unknown) | d_model 512→768 | Modified model.d_model which is a FROZEN autoencoder param. Checkpoint key mismatch — autoencoder checkpoint has d_model=512, cannot load into model with d_model=768. Run died in ~30s during model init, zero metrics logged. |
| exp_002 | distinctive-cosmos-14 | ke9d4u5g | num_encoder_layers 3→6, num_decoder_layers 3→6 | Modified frozen autoencoder depth. Checkpoint key mismatch — autoencoder checkpoint has 3 layers, cannot load into model with 6 layers. Run died in 29s during model init, zero metrics logged. |
| exp_003 | quiet-shadow-15 | q14jh32m | mlp_dim 512→768 (head-only) | Changed mlp_dim from 512 to 768 but pretrained head weights are for mlp_dim=512. With strict=False, all head Linear layer weights shape-mismatched and silently dropped by PyTorch, leaving randomly initialized head. CUDA error on first forward pass. Run died in 28s, zero metrics logged. |
