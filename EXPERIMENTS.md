# SpecPT-HST-Sim Experiment Log

## Current Best
| Metric | Value | Experiment |
|--------|-------|------------|
| NMAD | **0.01382** | **exp_013** |
| Catastrophic Outliers | 22.86% | exp_008_v2 (ep240) |
| ECE | — | — |
| RMSE | 0.399 | exp_013 |
| Val Loss | 0.377 | exp_013 |
| Train Loss | 0.040 | exp_013 |

## Targets
| Metric | Target | Current Best | Gap |
|--------|--------|-------------|-----|
| NMAD | < 0.020 | **0.01382** | **— TARGET ACHIEVED —** |
| Catastrophic Outliers | < 1% | 22.86% | 21.86% |
| ECE | < 0.1 | — | — |

*Last updated: 2026-06-19 23:52 UTC*

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
| exp_013 | configs/exp_013.yaml | fpw7o9pz | ruby-surf-26 | 0.01382 | 0.01539 | 24.85% | — | 0.399 | 0.377 | mlp_dim 1024, blocks=12, DESI autoencoder, residual fix. NEW BEST NMAD 0.01382! Severe overfitting: train_loss=0.040 vs val_loss=0.377 (9.7x gap). |
| exp_014 | configs/exp_014.yaml | m03w5sgt | comic-sea-27 | 0.01640 | 0.01870 | 23.78% | — | 0.429 | 0.384 | dropout_rate=0.1 added to head. NMAD worsened 0.01382→0.01640. Overfitting gap unchanged at 9.6x. Dropout 0.1 too weak to regularize 12-block 1024-dim head. Epochs: 295. |
| exp_015 | configs/exp_015.yaml | bi2vfuwc | fragrant-blaze-28 | 0.02156 | 0.02248 | 23.37% | — | 0.3875 | 0.3669 | dropout_rate=0.2. NMAD worsened to 0.02156 (56% worse than 0.01382 best). Dropout 0.0→0.1→0.2 shows monotonic degradation — regularization is wrong lever. Early stopped epoch 200/400. |
| exp_016 | configs/exp_016.yaml | c1y7wiur | celestial-wildflower-29 | 0.02100 | 0.02100 | 24.04 | — | 0.3875 | 0.3728 | weight_decay=1e-5 worsened NMAD 52% (0.01382→0.02100). Overfitting gap 9.1x unchanged. Early stopped ep210/400. Recommendation: tune regularization opposite direction. |
| exp_017 | configs/exp_017.yaml | syo645wg | iconic-flower-30 | 0.02132 | 0.0245 | 23.84% | — | 0.4058 | 0.3688 | weight_decay=5e-4 worsened NMAD 54% (0.01382→0.02132). Both regularization directions degraded — wd is exhausted. Early stopped ep220/400. Recommendation: hold_direction, revert to wd=5e-5. |
| exp_018 | configs/exp_018.yaml | 7ygfz3c1 | sandy-frog-31 | 0.02062 | 0.02093 | 23.74% | — | 0.4113 | 0.3722 | mlp_dim 1024→768 reduced overfitting gap 9.7x→8.5x but NMAD degraded 49% (0.01382→0.02062). Fifth consecutive post-exp_013 degradation. Recommendation: hold_direction. Ep238/400 (early stopped). |
| exp_019 | configs/exp_019.yaml | lff5uyf0 | snowy-valley-32 | 0.0167 | 0.0167 | 24.70 | — | 0.399 | 0.370 | batch_size 128→256: larger batch for smoother gradients. NMAD 0.0167 (21% worse than 0.01382 best, but 19% better than exp_018 0.02062). Overfitting gap 10.2x. Critical: 500-epoch warmup starved model — LR only 7.1e-5 at ep355, never reached full 1e-4. ReduceLROnPlateau never triggered. Ep355/400 early stop. |
| exp_020 | configs/exp_020.yaml | mjt4144y | deep-dew-33 | 0.01909 | 0.01953 | 23.55% | — | 0.402 | 0.3739 | warmup_epochs 500→50: warmup fix worked, LR ascended to 6.36e-5, but NMAD WORSENED to 0.01909 (worse than exp_019 0.0167, 38% worse than best 0.01382). Severe overfitting: train_loss 0.038 vs val_loss 0.374 (gap 0.3308). Outliers rising 23.0%→23.6%. Early stop ep318/400. NMAD still decreasing at termination — longer patience may help. |
| exp_021 | configs/exp_021.yaml | x5j6t95u | amber-gorge-34 | 0.01506 | 0.01727 | 24.07% | — | 0.4448 | 0.3719 | patience 50→100: exp_020 early-stopped while NMAD still decreasing. Longer patience enabled full 400-epoch training. Best NMAD 0.01506 at epoch 373 but degraded in final 27 epochs. Train/val loss gap ~10x — severe overfitting persists. LR barely decayed (1e-4→8e-5). |
| exp_022 | configs/exp_022.yaml | h15nlcte | elated-bird-35 | 0.01712 | 0.01755 | 23.99% | — | 0.419 | 0.368 | epochs 400→600 backfired: early-stopped ep342/600. Best NMAD 0.01712 at ep341 (worse than exp_021 0.01506). LR barely moved (6.8e-5). 7th consecutive post-exp_013 degradation. |
| exp_023 | configs/exp_023.yaml | l070d2wj | balmy-snow-36 | 0.016995 | 0.020748 | 23.60% | — | 0.400 | 0.371 | lr 1e-4→5e-5: hold_direction — best NMAD 0.016995 (improved vs recent experiments but 8th consecutive degradation vs 0.01382 best). NMAD oscillated 0.017-0.023. Completed 502/600 epochs. |


## Running Experiments
| exp | config | run_id | run_name | best_nmad | final_nmad | final_outs | val_z_bias | val_rmse | val_loss | notes |
|-----|--------|--------|----------|-----------|------------|------------|------------|----------|----------|-------|
| exp_024 | configs/exp_024.yaml | 21359571 | — | — | — | — | — | — | — | weight_decay 5e-5→1e-4 at lr=5e-5 (restore effective regularization balance in high-capacity head). |
| exp_025 | configs/exp_025.yaml | 21362553 | — | — | — | — | — | — | — | Test-Time Augmentation (TTA): n_aug=10, noise_std=0.01, shift=3, scale=[0.95,1.05]. |
| exp_026 | configs/exp_026.yaml | 21362554 | — | — | — | — | — | — | — | HuberNMADLoss: quadratic for small errors, linear for large. Delta=0.15. |
| exp_027 | configs/exp_027.yaml | 21362555 | — | — | — | — | — | — | — | Two-Stage Training: Stage 1 (200 ep) + Stage 2 (200 ep with outliers weighted 4x). |
| exp_028 | configs/exp_028.yaml | 21362556 | — | — | — | — | — | — | — | Per-Sample Loss Weighting: inverse error weighting by redshift bin. |
| exp_029 | configs/exp_029.yaml | 21362557 | — | — | — | — | — | — | — | MDN Head: K=5 Gaussian mixture for uncertainty-aware prediction. |
| exp_030 | configs/exp_030.yaml | 21362566 | — | — | — | — | — | — | — | Curriculum Learning: start with 50% easiest, ramp to 100% over 100 epochs. |

## Diagnostics (failed/crashed runs)
| exp | run_name | run_id | failure | diagnosis |
|-----|----------|--------|---------|-----------|
| exp_001 | kind-snowball-13 | 2enr6gyg | d_model 512→768 | Modified model.d_model which is a FROZEN autoencoder param. Checkpoint key mismatch — autoencoder checkpoint has d_model=512, cannot load into model with d_model=768. Run died in ~30s during model init, zero metrics logged. |
| exp_002 | distinctive-cosmos-14 | ke9d4u5g | num_encoder_layers 3→6, num_decoder_layers 3→6 | Modified frozen autoencoder depth. Checkpoint key mismatch — autoencoder checkpoint has 3 layers, cannot load into model with 6 layers. Run died in 29s during model init, zero metrics logged. |
| exp_003 | quiet-shadow-15 | q14jh32m | mlp_dim 512→768 (head-only) | Changed mlp_dim from 512 to 768 but pretrained head weights are for mlp_dim=512. With strict=False, all head Linear layer weights shape-mismatched and silently dropped by PyTorch, leaving randomly initialized head. CUDA error on first forward pass. Run died in 28s, zero metrics logged. |
| exp_010 | noble-frog-23 | nsomfkte | mlp_dim 512→1024 (head-width pivot) | Identical to exp_003 failure but with mlp_dim 512→1024 and 12 MLP blocks. All 12 mlp_blocks.* tensors size-mismatched (60 total). strict=False does NOT allow size mismatches — only missing/extra keys. RuntimeError at model init. Died in 21s, zero metrics logged. |
| exp_011 | peach-pine-24 | ee7l4hgl | pretrained_redshift="" → torch.load("") crash | pretrained_redshift="" bypassed the guard, passed empty string path to torch.load(), raising FileNotFoundError. Not a shape-mismatch like exp_010 — different root cause. Code fixed with guard before torch.load(). Died during model init, zero metrics logged. |
| exp_012 | sparkling-wood-25 | 2e9b7ic2 | mlp_dim 1024 → residual dim mismatch | mlp_dim=1024, num_mlp_blocks=12, ImprovedResidualMLPBlock residual connection size mismatch — residual target dim (512) ≠ MLP output dim (1024). Added residual projection layer to fix. Died during model init, zero metrics logged. |

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
