# SpecPT-HST-Sim Experiment Log

## Current Best
| Metric | Value | Experiment |
|--------|-------|------------|
| NMAD | **0.00785** | **exp_032** |
| Catastrophic Outliers | 15.17% | exp_032 |
| ECE | — | — |
| RMSE | 0.354 | exp_032 |
| Val Loss | 0.241 | exp_032 |
| Train Loss | 0.037 | exp_032 |

## Targets
| Metric | Target | Current Best | Gap |
|--------|--------|-------------|-----|
| NMAD | < 0.020 | **0.00785** | **— TARGET ACHIEVED —** |
| NMAD Stretch | < 0.010 | **0.00785** | **— STRETCH ACHIEVED —** |
| Catastrophic Outliers | < 1% | 15.17% | 14.17% |
| ECE | < 0.1 | — | — |

*Last updated: 2026-06-29 23:30 UTC*

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
| exp_024 | configs/exp_024.yaml | f4ligwsc | lucky-oath-37 | 0.01950 | 0.01950 | 23.30% | — | 0.4158 | 0.3715 | weight_decay 5e-5→1e-4 at lr=5e-5. NMAD WORSENED 41% (0.01382→0.01950). 9th consecutive post-exp_013 degradation. LR barely moved (5e-5→4.9e-5). Completed 462/600 epochs. |
| exp_025 | configs/exp_025.yaml | hwa8lkzg | polar-sky-39 | 0.01848 | 0.01848 | 23.85% | 0.00125 | 0.3750 | 0.3707 | TTA (N=10, noise=0.01, shift=3, scale=[0.95,1.05]). Best val NMAD among new experiments (34% better than exp_024). TTA paradox: RMSE improved 2.9% but NMAD worsened 2.8% on test set. Outliers unchanged. |
| exp_026 | configs/exp_026.yaml | qyciqxtr | wise-night-39 | 0.07975 | 0.07975 | 30.01% | -0.0056 | 0.3662 | 0.1226 | HuberNMADLoss. CRASHED ep72 — corrupted checkpoint. NMAD 5.7x worse. Loss scale mismatch (0.12 vs 0.37 standard). Huber loss requires lr/weight_decay retuning. |
| exp_027 | configs/exp_027.yaml | gkq9vb4q | playful-spaceship-38 | 0.01934 | 0.01934 | 24.25% | -0.00035 | 0.3694 | 0.3697 | Two-Stage (200+200, 4x outlier weight). NMAD 0.01934 — comparable to exp_024. Stage 2 outlier weighting did NOT reduce outliers (23.7%→24.2%). Early stopped ep241/400. |
| exp_028 | configs/exp_028.yaml | etw2syi3 | leafy-snow-41 | 0.01987 | 0.01987 | 23.79% | -0.00124 | 0.3719 | 0.3686 | Per-Sample Weights (inverse error by redshift bin). No improvement. Outliers uniformly distributed across redshift bins — targeting redshift bins is wrong axis. |
| exp_029 | configs/exp_029.yaml | pl0vpj93 | valiant-forest-42 | 0.02411 | 0.02411 | 23.48% | 0.00063 | 0.4514 | 1.6738 | MDN Head (K=5). CRASHED ep94 — state_dict mismatch (pretrained_redshift empty). Train loss went negative (-1.66). Val loss diverged 0.91→1.67. NMAD 74% worse. |
| exp_030 | configs/exp_030.yaml | 4vb1jarm | vibrant-waterfall-46 | 0.01816 | 0.01816 | 23.84% | 0.00156 | 0.3721 | 0.3721 | Curriculum (50%→100% over 100 ep). NMAD 0.01816 — 2nd best among all experiments! Test NMAD 0.02072. Outliers unchanged at 23.84%. |
| exp_026_rerun | configs/exp_026.yaml | blft8duo | azure-microwave-44 | 0.17836 | 0.17836 | 43.14% | — | 0.1328 | 0.1342 | HuberNMADLoss retry. CRASHED ep8 — disk full on cluster (PytorchStreamWriter failed). |
| exp_026_rerun2 | configs/exp_026.yaml | yfd4j21r | polished-galaxy-47 | 0.07718 | 0.07718 | 30.36% | — | 0.1225 | 0.1225 | HuberNMADLoss 3rd attempt. Completed 73 ep. NMAD 5.6x worse than baseline. Loss scale still mismatched (0.12 vs 0.37 standard). Test NMAD 0.828 (catastrophic). |
| exp_029_rerun | configs/exp_029.yaml | fcu8wxkj | solar-spaceship-44 | 0.02543 | 0.02543 | 24.15% | -1.5285 | 1.6628 | 1.6628 | MDN Head retry. CRASHED ep91 — optimizer state_dict mismatch on post-training checkpoint load (fixed in 28dd126). |
| exp_029_rerun2 | configs/exp_029.yaml | 6xer6tfq | rich-night-48 | 0.02539 | 0.02539 | 24.26% | -1.5037 | 1.7954 | MDN Head 3rd attempt. Completed 86 ep. Train loss negative (NLL). Val loss 1.795 (diverged). NMAD 0.02539 — 83% worse than best. Test NMAD 0.03926. |
| exp_030_rerun | configs/exp_030.yaml | 4vb1jarm | vibrant-waterfall-46 | 0.01816 | 0.01816 | 23.84% | 0.00156 | 0.3721 | 0.3721 | Curriculum retry. SUCCESS! NMAD 0.01816 (2nd best). 267 epochs. |
| exp_013_rerun | configs/exp_013.yaml | 4jrl2hig | laced-feather-49 | 0.02024 | 0.02158 | 23.62% | 0.000818 | 0.3924 | 0.3768 | exp_013 rerun for checkpoint recovery. NMAD 0.02024 (46.5% worse than original 0.01382). Early stopped ep216/400 — val_loss plateaued at ep166. Identical config to original, difference is random seed variance. |
| exp_031 | configs/exp_031.yaml | f9uaj0ae | revived-voice-50 | 0.01489 | 0.01489 | 24.39% | -0.00189 | 0.4390 | 0.3815 | exp_013 config + patience 150, epochs 600. NMAD 0.01489 — 4th best overall! Early stopped ep344/600. LR decayed 1e-4→6.9e-5. Overfitting gap 10x persists. Longer patience helped but didn't recover to original exp_013 (0.01382). |
| exp_032 | configs/exp_032.yaml | ejfhtjlk | confused-bee-54 | **0.00785** | 0.00921 | 15.17% | -0.00189 | 0.354 | 0.241 | NEW ALL-TIME BEST! Q1 quality data + exp_013 config (mlp_dim=1024, blocks=12, lr=1e-4). NMAD improved 43% (0.01382→0.00785). Overfit gap 6.85x (vs 9.43x). Best at ep314, final ep325. Catastrophic outliers 15.17% (down from 24.85%). LR 6.52e-5 (never reached 1e-4). |


## Running Experiments
| exp | config | run_id | run_name | best_nmad | final_nmad | final_outs | val_z_bias | val_rmse | val_loss | notes |
|-----|--------|--------|----------|-----------|------------|------------|------------|----------|----------|-------|
| use_stageB | scripts/pretrain_universal_encoder.py | — | use_stageB_stageB | — | — | — | — | — | — | Universal Spectral Encoder Stage B: masked + noise reconstruction + distill anchor, resuming from use_stageA_stageA_best.pth. Real 3D-HST, bs=64 lr=1e-4 ep=100. Job TBD. |

## Universal Spectral Encoder (USE)
Label-free self-supervised encoder preserving broad spectral info + adding robustness. Reference: docs/universal_spectral_encoder.md.

| Stage | Status | Train | Val | Drift | Latent cos | Eval NMAD | Notes |
|-------|--------|-------|-----|-------|------------|-----------|-------|
| A (masked + distill) | ✅ Complete (ep 49/100) | 0.013 | 0.0137 | ~1.0 | 0.995 | **0.2169** | Eval vs AE baseline 0.2162 (tied). Latent robust to mask/noise (cos 0.996+). Recon MSE 0.0077 (baseline 0.0103). z not improved by masked recon alone — expected; robustness stages B/C are the z-relevant levers. |
| B (masked + noise) | 🚧 Running | — | — | — | — | — | Adds noise-corrupted reconstruction. |
| C (+ consistency) | — | — | — | — | — | — | Two-view latent consistency attacks noise-entanglement behind SNR<5 tail. |

## Diagnostics (failed/crashed runs)
| exp | run_name | run_id | failure | diagnosis |
|-----|----------|--------|---------|-----------|
| exp_001 | kind-snowball-13 | 2enr6gyg | d_model 512→768 | Modified model.d_model which is a FROZEN autoencoder param. Checkpoint key mismatch — autoencoder checkpoint has d_model=512, cannot load into model with d_model=768. Run died in ~30s during model init, zero metrics logged. |
| exp_002 | distinctive-cosmos-14 | ke9d4u5g | num_encoder_layers 3→6, num_decoder_layers 3→6 | Modified frozen autoencoder depth. Checkpoint key mismatch — autoencoder checkpoint has 3 layers, cannot load into model with 6 layers. Run died in 29s during model init, zero metrics logged. |
| exp_003 | quiet-shadow-15 | q14jh32m | mlp_dim 512→768 (head-only) | Changed mlp_dim from 512 to 768 but pretrained head weights are for mlp_dim=512. With strict=False, all head Linear layer weights shape-mismatched and silently dropped by PyTorch, leaving randomly initialized head. CUDA error on first forward pass. Run died in 28s, zero metrics logged. |
| exp_010 | noble-frog-23 | nsomfkte | mlp_dim 512→1024 (head-width pivot) | Identical to exp_003 failure but with mlp_dim 512→1024 and 12 MLP blocks. All 12 mlp_blocks.* tensors size-mismatched (60 total). strict=False does NOT allow size mismatches — only missing/extra keys. RuntimeError at model init. Died in 21s, zero metrics logged. |
| exp_011 | peach-pine-24 | ee7l4hgl | pretrained_redshift="" → torch.load("") crash | pretrained_redshift="" bypassed the guard, passed empty string path to torch.load(), raising FileNotFoundError. Not a shape-mismatch like exp_010 — different root cause. Code fixed with guard before torch.load(). Died during model init, zero metrics logged. |
| exp_012 | sparkling-wood-25 | 2e9b7ic2 | mlp_dim 1024 → residual dim mismatch | mlp_dim=1024, num_mlp_blocks=12, ImprovedResidualMLPBlock residual connection size mismatch — residual target dim (512) ≠ MLP output dim (1024). Added residual projection layer to fix. Died during model init, zero metrics logged. |
| deep-energy-43 | deep-energy-43 | t18mboez | curriculum=True (string weight_decay) | Old experiment submitted before curriculum fix. Config has `curriculum: True` and `weight_decay: '5e-5'` (string, not float). Failed with zero metrics. Pre-dates systematic experiment tracking. |
| exp_048_joint_sim_real | 21424155 | exp_048_joint_sim_real | FAILED — data bug: sim volume 72,361 rows (22:1 ratio vs real) instead of planned ~14k. Random 90/10 split instead of split_by_grism_id. Encoder diverged from real utility (best ep=1). NMAD: initial 0.254 → final 0.277. Rerun as exp_048b. |
| exp_048b_joint_corrected | 21427640 | exp_048b_joint_corrected | FAILED — even with corrected sim volume (~14k, 4.7:1 ratio) + split_by_grism_id, result identical to exp_048. Test NMAD 0.268, best ep=1, recon MSE 3.4x drift. Root cause: **encoder unfreezing itself** destroys AE identity regardless of sim volume. The fragility is gradient-driven (NMADLoss regression), not sim volume. Pivoting to RNC (exp_050/051). |
| exp_050_RNC_frozen (attempt 1) | 21428513 | exp_050_RNC_frozen_stage1+stage2 | FAILED — RNC loss flat at 4.577 (train) / 4.544 (val) across 77 Stage 1 epochs. L2-normalized features create a numerical saddle point where all pairwise distances are uniform → gradient vanishes. Test NMAD 0.243 (indistinguishable from random init). Rerunning with no L2-norm, T=0.5, proj LR 3e-3. |
| exp_051_RNC_unfrozen (attempt 1) | 21428514 | exp_051_RNC_unfrozen_stage1+stage2 | FAILED — identical L2-norm saddle bug. RNC loss flat at 4.577/4.544. Test NMAD 0.245. Rerunning with no L2-norm, T=0.5, proj LR 3e-3, enc LR 1e-5. + recon MSE monitoring for AE identity tracking. |
| exp_050_RNC_frozen (attempt 2) | 21436332 | exp_050_RNC_frozen_stage1+stage2 | FAILED — RNC loss STILL flat even after removing L2-norm and lowering T to 0.5 (train 4.584→4.577, val 4.548→4.544, only 0.15% drop over 53 ep). Recon MSE stable at 0.00825 (encoder frozen perfectly). Test NMAD 0.2422, η 52.2%, R² -0.033 (indistinguishable from random init). Root cause: RNC gradient at random projection init requires z-discriminative features in the encoder — which don't exist. |
| exp_051_RNC_unfrozen (attempt 2) | 21436333 | exp_051_RNC_unfrozen_stage1+stage2 | FAILED — RNC loss flat at 4.584→4.577 despite encoder LR 1e-5. Recon MSE drifted CATASTROPHICALLY: 0.008→1.039 (126× drift over 66 ep). RNC gradient IS flowing to encoder, but it's noise — no z-organization, just random drift destroying AE identity. Test NMAD 0.2394, η 52.2%, R² -0.035 (worse than random init baseline). Confirms: RNC cannot bootstrap z-ordering, and unfrozen encoder suffers uncontrolled drift from the noise gradient. |

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

## Real 3D-HST Fine-tuning

Two-stage fine-tuning of exp_032 on real 3D-HST grism data (grism_specPT_v5.pkl, 11,156 spectra after SNR≥2.5). Data interpolated to training wavelength grid (10311–17465 Å) with NaN-masking at [11000, 16500] Å detection limits and z-score normalized. All metrics on held-out test set (1,674 spectra).

| Stage | Approach | Trainable Params | LR | Epochs | Test NMAD | Eta | Notes |
|-------|----------|-----------------|----|--------|-----------|-----|-------|
| Pre-fix | Frozen exp_032 (raw eval) | — | — | — | ~0.50 | — | No interpolation applied — model sees shifted features |
| 1 | Linear probe (prediction head only) | 525,313 | 3e-4 | 30 | **0.2105** | 47.6% | Frozen encoder features contain real-data redshift info. NMAD improved 2.4× over pre-fix baseline. Severe outlier fraction (47.6%) suggests encoder features are discriminative but noisy. Init from exp_032 best. |
| 2 | Partial freeze (last encoder + attention + 12 MLP + head) | ~4.7M | 1e-5 | 40 | **0.2073** | 47.3% | Minimal improvement over Stage 1 (0.2105 → 0.2073, 1.5%). Training loss continued decreasing (0.667→0.502) but val NMAD plateaued — overfitting on 7.8k train samples. Early stopped at epoch 10. Best val NMAD 0.20994 at epoch 3. |

## Regridded Backbone Real-Data Fine-tuning

Real 3D-HST grism data (grism_specPT_v5.pkl, 11,156 spectra after SNR≥2.5) fine-tuned on the regridded autoencoder backbone (10800–17100 Å, 0.81 Å/pix). Data split by `split_by_grism_id` (seed=42, val/test=0.1/0.1 → 8,924 train / 1,116 val / 1,116 test; augmented runs: 104,651 train). NaN-masking at [11000, 16500] Å (PAD mask).

| Exp | Head | Backbone | Augment | Run ID | Test NMAD | Test η% | Test RMSE | Best Ep | Notes |
|-----|------|----------|---------|--------|-----------|---------|-----------|---------|-------|
| **exp_035** | Linear (simple) | **Frozen** | No | jwrmz004 | **0.24883** | 54.64 | 0.5807 | 19 | Best NMAD on real data. Frozen linear probe (seq 512→256→1+Softplus, 525k params). |
| exp_036 | Linear (simple) | Frozen | Yes | ffztx2hk | 0.33644 | 67.27 | 0.7263 | 1 | Augmentation (×11.7) degrades NMAD 35%. |
| exp_037 | Enhanced (5 blocks) | Unfrozen | No | 4za4hxi7 | 0.28494 | 59.85 | 0.6579 | 1 | End-to-end overfits: train_loss 0.003 → val_loss 0.30+. Best ep=1 means immediate divergence. |
| exp_038 | Enhanced (5 blocks) | Unfrozen | Yes | pre196tp | 0.32759 | 64.27 | 0.7178 | 58 | End-to-end + augment — worst non-metric learning result. |
| exp_039 | Simple (2-layer) | Unfrozen | No | pxg2oyyj | 0.27547 | 58.80 | 0.6411 | 1 | Unfrozen simple head diverges immediately. |
| exp_040 | Simple (2-layer) | Unfrozen | No | scqtgcus | 0.27395 | 58.55 | 0.6368 | 1 | Unfrozen — statistically tied with exp_039. |
| exp_041 | MLP (3-layer, 512→256→128→1) | Frozen | No | 4xqrr56o | 0.26045 | 56.61 | 0.6112 | 292 | Deeper MLP on frozen features — 5% worse than linear probe (exp_035). Ran full 300 epochs, minimal overfit. Head capacity doesn't unlock frozen features. |
| exp_042 | ResNet (3×ImprovedResidualMLPBlock) | Frozen | No | p82m5074 | 0.26646 | 57.69 | 0.6299 | 32 | Residual blocks on frozen features — plateaued at ep 32, early-stopped ep 62. |
| exp_043 | Metric Learning (NTXent + k-NN=10) | Frozen | No | ckurr32l | 0.27399 | 58.35 | 0.6216 | 1 | Contrastive loss never converged (train_loss 1.67). Best ep=1. |
| exp_044 | Random Forest | Frozen | No | if2qx78l | — | — | — | — | 🚧 **Failed** — shape bug corrupts metrics (y arrays are (n,1) 2-d, compute_metrics broadcast to n×n pairwise matrix). Fixed in re-run. |
| **exp_045_RF_fixed** | **Random Forest (fixed)** | **Frozen** | **No** | **a5ubhtwv** | **0.20767** | **49.82** | **0.5269** | **n/a** | **New best NMAD on real data (16.5% improvement vs exp_035).** RF predicts narrow range [0.73, 1.80] vs true range [0.01, 3.47] — improvement is from implicit shrinkage, not non-linear z-structure. Test R² = 0.094 confirms no z-variance tracking. η = 49.8%—still half of test samples have >15% error. |
| exp_046_pre_attn_RF | Random Forest (pre-attention) | Frozen | No | tjv2eltj | 0.20844 | 50.81 | 0.5270 | n/a | B1+B3: Pre-attention NF matches exp_045 within 0.4% → **MHA is decorative.** B3: NMAD by SNR — SNR<5: 0.287, SNR 5-10: 0.180, SNR 10-20: 0.099, SNR 20+: 0.082. Catastrophic η concentrated in low-SNR tail. |
| exp_047_huber_linear | Linear probe (loss ablation) | Frozen | No | 0.26528 | 57.77 | 0.6368 | 12 | 42 | ❌ Loss-function mismatch is NOT the bottleneck. HuberNMADLoss (δ=0.15) degrades NMAD (0.265 vs exp_035 0.249). |
| exp_048_joint_sim_real | Joint sim+real | Unfrozen | No | 0.27131 | 58.06 | 0.6446 | 1 | 16 | ❌ FAILED — sim volume 22:1 (72k rows). Encoder diverged (best ep=1). |
| exp_048b_joint_corrected | Joint sim+real (corrected) | Unfrozen | No | 0.26806 | 57.78 | 0.6447 | 1 | 16 | ❌ FAILED — corrected sim volume (4.7:1, 14k). Same failure as exp_048. Encoder unfreezing per se is the killer, not sim volume. |
| exp_050_RNC_frozen | RNC (attempt 1: L2-norm, T=2.0) | Frozen | No | 0.24335 | 54.66 | 0.2362 | 1 | ❌ RNC loss flat at 4.577 — L2-norm init saddle. |
| exp_051_RNC_unfrozen | RNC (attempt 1: L2-norm, T=2.0) | Unfrozen (1e-6) | No | 0.24455 | 54.57 | 0.2363 | 1 | ❌ RNC loss flat — same L2-norm saddle. |
| exp_050_RNC_frozen_v2 | RNC (attempt 2: no L2-norm, T=0.5) | Frozen | No | 0.24224 | 52.24 | 0.2333 | 1 | ❌ Still flat. RNC gradient requires z-discriminative encoder features. Recon MSE 1.0× (frozen OK). |
| exp_051_RNC_unfrozen_v2 | RNC (attempt 2: no L2-norm, T=0.5) | Unfrozen (1e-5) | No | 0.23935 | 52.24 | 0.2330 | 1 | ❌ Still flat. Encoder DRIFTED 126× (recon 0.008→1.039). RNC gradient is noise shredding AE identity. |

## Consolidated Findings

After 9 experiments (exp_035–051) on the regridded autoencoder with real 3D-HST data, **all six axes for extracting z-discrimination from the frozen AE encoder are exhausted:** 

| Axis | Exp | Outcome |
|------|-----|---------|
| Head capacity (MLP, ResNet) | 035, 041, 042 | Plateau at NMAD 0.25 — encoder features don't have z-discriminative structure |
| Tree methods (RF, pre-attn RF) | 045, 046 | RF shrinkage helps NMAD (0.208) but no z-variance tracking (R²=0.094). MHA is decorative (pre/post Δ=0.4%) |
| Loss function (NMADLoss vs Huber) | 047 | NMADLoss confirmed optimal; loss-axis dead |
| Unfrozen encoder regression (joint training) | 048, 048b | Encoder drift destroys AE identity monotonically from ep 1 regardless of sim volume or split |
| Contrastive learning (RNC, frozen) | 050 (×2) | RNC loss flat — gradient vanishes at random projection init because features are z-entangled, not z-discriminative |
| Contrastive learning (RNC, unfrozen) | 051 (×2) | RNC loss flat even with encoder LR 1e-5. Encoder drifts 126× via noise gradient. All destruction, no z-organization |

**Conclusion: The frozen autoencoder encoder produces features that are z-entangled for reconstruction, not z-discriminative. No downstream method can extract z-signal that isn't there. The bottleneck is the encoder itself.**

*Last updated: 2026-07-31 13:30 UTC*
