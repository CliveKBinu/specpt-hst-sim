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
| exp_002 | configs/exp_002.yaml | — | — | — | — | — | — | — | — | pending. Encoder/decoder depth 3→6. Complementary to exp_001 width increase. |

## Diagnostics (failed/crashed runs)
*None yet*
