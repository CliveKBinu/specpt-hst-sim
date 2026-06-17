# SpecPT-HST-Sim Experiment Log

## Current Best
| Metric | Value | Experiment |
|--------|-------|------------|
| NMAD | **0.02295** | **exp_008_v2** |
| Catastrophic Outliers | 22.86% | exp_008_v2 (ep240) |
| ECE | — | — |
| RMSE | 0.393 | exp_008_v2 |
| Val Loss | 0.358 | exp_007 / exp_008_v2 |
| Train Loss | 0.178 | exp_000_baseline |

## Targets
| Metric | Target | Current Best | Gap |
|--------|--------|-------------|-----|
| NMAD | < 0.020 | **0.02295** | **0.00295** |
| Catastrophic Outliers | < 1% | 22.86% | 21.86% |
| ECE | < 0.1 | — | — |

*Last updated: 2026-06-17T14:00 UTC*

## Completed Experiments
| exp | config | run_id | run_name | best_nmad | final_nmad | final_outs | val_z_bias | val_rmse | val_loss | notes |
|-----|--------|--------|----------|-----------|------------|------------|------------|----------|----------|-------|
| exp_000_baseline | defaults.yaml | wijvni9f | super-disco-12 | 0.0303 | 0.0313 | 23.24% | -0.0023 | 0.366 | 0.349 | Baseline. Stopped early at epoch 244/400. LR still warming up. NMAD still improving. Capacity bottleneck suspected. |
| exp_004 | configs/exp_004.yaml | gx70j2j8 | stoic-spaceship-16 | 0.0335 | 0.0346 | 23.42% | -0.0023 | 0.382 | 0.365 | lr 1e-4→5e-5. NMAD WORSENED (0.0335 vs 0.0303 baseline). Lower LR starved gradients — model underfits. Capacity bottleneck confirmed. |
| exp_005 | configs/exp_005.yaml | 95vn9fc6 | effortless-terrain-17 | 0.0279 | 0.0279 | 23.18% | 0.0011 | 0.425 | 0.366 | num_mlp_blocks 5→7, lr 1e-4. NMAD IMPROVED 8% to 0.0279 (new best). Overfitting: val_loss rose, early stopping at epoch 249. NMAD still decreasing at termination. |
| exp_006 | configs/exp_006.yaml | oy9u11a1 | apricot-oath-18 | 0.0332 | 0.0342 | 23.98% | 0.0003 | 0.387 | 0.361 | weight_decay 5e-5→1e-4. NMAD WORSENED (0.0332 vs 0.0279 best). Regularization backfired — deeper head needs capacity, not constriction. Trained 193/400 epochs. NMAD still improving at termination. |
| exp_007 | configs/exp_007.yaml | n84weil0 | distinctive-bee-19 | 0.02565 | 0.02565 | 23.61% | -0.00231 | 0.389 | 0.358 | num_mlp_blocks 7→10, weight_decay 1e-4→5e-5 (revert reg). NEW BEST NMAD 0.02565 (8% improvement over exp_005). NMAD still improving at termination epoch 257. LR decayed 1e-4→5.16e-05. |
| exp_008 | configs/exp_008.yaml | ld28scut | drawn-sun-20 | 0.02568 | 0.02655 | 23.25% | -0.00045 | 0.375 | 0.348 | num_mlp_blocks 10→12, lr 1e-4, wd 5e-5. NMAD 0.02568 plateaued (statistically tied with exp_007 0.02565). Capacity saturating at ~10 blocks. Early stopped ep 219/400. |
| exp_008_v2 | configs/exp_008.yaml | 2cnzeyqt | rural-bush-21 | **0.02295** | 0.02385 | 23.31% | -0.00369 | 0.393 | 0.358 | num_mlp_blocks=12 + DESI combined autoencoder + lr=1e-4. NEW BEST NMAD 0.02295 (10.5% over exp_007). Test NMAD 0.02726. Early stopped ep242/400. |
| exp_009 | configs/exp_009.yaml | 2689fu6f | proud-sea-21 | 0.02611 | 0.02706 | 23.07% | — | — | 0.360 | lr 1e-4→2e-4, num_mlp_blocks=12, DESI autoencoder. Higher LR worsened NMAD (0.02611 vs 0.02295). Test NMAD 0.03283. Ep154/400 (early stopped). |

## Running Experiments
| exp | config | run_id | run_name | best_nmad | final_nmad | final_outs | val_z_bias | val_rmse | val_loss | notes |
|-----|--------|--------|----------|-----------|------------|------------|------------|----------|----------|-------|
| exp_011 | configs/exp_011.yaml | — | — | — | — | — | — | — | — | mlp_dim 512→1024 (wider head), pretrained_redshift="" (train head from scratch). Tests if increased head width improves NMAD now that shape-mismatch constraint is removed. Submitted. |

## Diagnostics (failed/crashed runs)
| exp | run_name | run_id | failure | diagnosis |
|-----|----------|--------|---------|-----------|
| exp_001 | kind-snowball-13 | 2enr6gyg | d_model 512→768 | Modified model.d_model which is a FROZEN autoencoder param. Checkpoint key mismatch — autoencoder checkpoint has d_model=512, cannot load into model with d_model=768. Run died in ~30s during model init, zero metrics logged. |
| exp_002 | distinctive-cosmos-14 | ke9d4u5g | num_encoder_layers 3→6, num_decoder_layers 3→6 | Modified frozen autoencoder depth. Checkpoint key mismatch — autoencoder checkpoint has 3 layers, cannot load into model with 6 layers. Run died in 29s during model init, zero metrics logged. |
| exp_003 | quiet-shadow-15 | q14jh32m | mlp_dim 512→768 (head-only) | Changed mlp_dim from 512 to 768 but pretrained head weights are for mlp_dim=512. With strict=False, all head Linear layer weights shape-mismatched and silently dropped by PyTorch, leaving randomly initialized head. CUDA error on first forward pass. Run died in 28s, zero metrics logged. |
| exp_010 | noble-frog-23 | nsomfkte | mlp_dim 512→1024 (head-width pivot) | Identical to exp_003 failure but with mlp_dim 512→1024 and 12 MLP blocks. All 12 mlp_blocks.* tensors size-mismatched (60 total). strict=False does NOT allow size mismatches — only missing/extra keys. RuntimeError at model init. Died in 21s, zero metrics logged. |

## Early Untracked Runs
These are early test baseline runs on the HST augmented autoencoder before the tracking system was operational. All failed during model init or data loading and are kept for historical reference.

| run_name | run_id | state | failure |
|----------|--------|-------|---------|
| swept-lake-1 | f7rovfr1 | failed | Early test baseline (pre-tracking) |
| royal-totem-2 | 3lgznraa | failed | Early test baseline (pre-tracking) |
| fragrant-plasma-3 | b4ot3x2b | failed | Early test baseline (pre-tracking) |
| wandering-firebrand-4 | cos4bv9c | failed | Early test baseline (pre-tracking) |
| elated-sun-5 | o1wa1qkr | failed | Early test baseline (pre-tracking) |
| resilient-glitter-6 | wbb06713 | failed | Early test baseline (pre-tracking) |
| dutiful-planet-7 | lkxytnm1 | failed | Early test baseline (pre-tracking) |
| vague-wind-8 | 4hqjpxd9 | failed | Early test baseline (pre-tracking) |
| twilight-dream-9 | hmjmutda | failed | Early test baseline (pre-tracking) |
| sandy-flower-10 | vvv6icm1 | failed | Early test baseline (pre-tracking) |
| amber-spaceship-11 | r832efo5 | failed | Early test baseline (pre-tracking) |

Also, SLURM job 21346405 (failed) and 21346407 (completed) correspond to early test runs before systematic experiment tracking began.
