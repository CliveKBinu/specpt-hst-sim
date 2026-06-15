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
| exp_001 | configs/exp_001.yaml | 21346908 | — | — | — | — | — | — | — | d_model 512→768. Capacity increase. Submitted. Job 21346908. |
| exp_002 | configs/exp_002.yaml | 21346914 | — | — | — | — | — | — | — | encoder/decoder depth 3→6. Complementary to exp_001 width increase. Submitted. Job 21346914. |
| exp_003 | configs/exp_003.yaml | — | — | — | — | — | — | — | — | mlp_dim 512→768. Head capacity increase. Pending. |

## Diagnostics (failed/crashed runs)
| exp | run_name | run_id | failure | diagnosis |
|-----|----------|--------|---------|-----------|
| exp_001 | (unknown) | (unknown) | d_model 512→768 | Modified model.d_model which is a FROZEN autoencoder param. Checkpoint key mismatch — autoencoder checkpoint has d_model=512, cannot load into model with d_model=768. Run died in ~30s during model init, zero metrics logged. |
| exp_002 | distinctive-cosmos-14 | ke9d4u5g | num_encoder_layers 3→6, num_decoder_layers 3→6 | Modified frozen autoencoder depth. Checkpoint key mismatch — autoencoder checkpoint has 3 layers, cannot load into model with 6 layers. Run died in 29s during model init, zero metrics logged. |
