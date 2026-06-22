# Outlier Reduction Plan

## Current State
- **Best NMAD:** 0.01382 (exp_013) — target <0.020 **ACHIEVED**
- **Outliers:** 23-24% — target <1% — **22x gap**
- **Root cause:** Outlier count is data-invariant (same ~23% across ALL 20 experiments). The loss function treats all samples equally, so the model can't distinguish easy from hard examples.

## Analysis Summary

### Key Findings from 20 Experiments
1. **Outlier floor is data-invariant:** All runs plateau at 23-25% outliers regardless of model capacity, regularization, LR, batch size, or training duration
2. **Overfitting is universal:** Train/val loss gap of 8.5-11.8x across all runs
3. **Gradient saturation:** Deep MLP blocks (layers 5-11) show near-zero gradients by mid-training; only the final prediction layer shows significant gradient activity
4. **NMAD vs outliers are decoupled:** Improving NMAD does not reduce outliers — the model learns to predict the median well but fails on the tails

### What Has Been Tried (EXCLUDED)
- Capacity: blocks 5→7→10→12, dim 512→768→1024
- Regularization: dropout 0→0.1→0.2, weight_decay 1e-5→5e-4
- LR: 5e-5→2e-4
- Batch size: 128→256
- Training duration: 200→600 epochs, patience 50→100
- Warmup: 50→500 epochs
- Data: DESI combined autoencoder

## Six Approaches (Parallel Execution)

### Approach 1: Test-Time Augmentation (TTA) — exp_025
**Priority:** 1st — zero training cost, immediate results
**Expected:** 15-18% outliers

At inference, create N augmented copies of each test spectrum (noise, shifts, scaling), run model on each, average predictions. Reduces variance on high-uncertainty (outlier) predictions.

**Config:**
```yaml
tta:
  enabled: true
  n_augmentations: 10
  noise_std: 0.01
  max_shift: 3
  flux_scale_range: [0.95, 1.05]
```

**Files to modify:**
- `src/specpt/training/eval.py` — add `predict_with_tta()`
- `src/specpt/training/train.py` — add TTA evaluation block

---

### Approach 2: Huber-Adaptive NMAD Loss — exp_026
**Priority:** 2nd — addresses root cause of uniform error weighting
**Expected:** 15-18% outliers

Replace NMADLoss with Huber-style loss: quadratic for small errors (|Δz/(1+z)| < δ), linear for large errors. Prevents extreme outliers from dominating gradient updates.

**Config:**
```yaml
training:
  loss: huber_nmad
  loss_delta: 0.15
```

**Files to modify:**
- `src/specpt/losses.py` — add `HuberNMADLoss` class
- `src/specpt/training/train.py` — wire up loss selection

---

### Approach 3: Two-Stage Training — exp_027
**Priority:** 3rd — targeted re-training on identified problem samples
**Expected:** 12-15% outliers

Stage 1: Train normally for 200 epochs. Stage 2: Identify outlier samples (|Δz/(1+z)| > 0.15), resume training with those samples weighted 4x in loss.

**Config:**
```yaml
training:
  two_stage: true
  stage1_epochs: 200
  stage2_epochs: 200
  outlier_weight: 4.0
  outlier_threshold: 0.15
```

**Files to modify:**
- `src/specpt/training/train.py` — add two-stage loop and weighted sampling

---

### Approach 4: Per-Sample Loss Weighting — exp_028
**Priority:** 4th — addresses redshift-dependent error patterns
**Expected:** 16-19% outliers

Compute error distribution by redshift bin from baseline model, assign inverse-error weights, retrain with weighted loss.

**Config:**
```yaml
training:
  sample_weighting: redshift_inverse_error
  weight_bins: [0, 0.5, 1.0, 1.5, 2.0, 3.0]
```

**Files to modify:**
- `src/specpt/training/train.py` — add weight computation and weighted sampler

---

### Approach 5: Mixture Density Network (MDN) Head — exp_029
**Priority:** 5th — most complex but highest potential
**Expected:** 10-15% outliers

Replace point prediction with K=5 Gaussian mixture. Model outputs means, log-variances, and mixture weights. At inference, use highest-weight component mean.

**Config:**
```yaml
model:
  prediction_type: mdn
  num_mixtures: 5
```

**Files to modify:**
- `src/specpt/model.py` — add `MDNHead` module
- `src/specpt/losses.py` — add `MDNMADLoss`

---

### Approach 6: Hard Example Mining with Curriculum — exp_030
**Priority:** 6th — addresses training dynamics
**Expected:** 18-20% outliers

Start training with easiest 50% of samples, gradually increase to 100% over ramp period. Prevents outlier-dominated gradients early in training.

**Config:**
```yaml
training:
  curriculum: true
  curriculum_start_pct: 0.5
  curriculum_ramp_epochs: 100
```

**Files to modify:**
- `src/specpt/training/train.py` — add curriculum scheduler

---

## Execution Plan

| Experiment | Approach | Config | Est. Time |
|------------|----------|--------|-----------|
| exp_025 | TTA | configs/exp_025.yaml | 1 hour |
| exp_026 | HuberNMADLoss | configs/exp_026.yaml | 4 hours |
| exp_027 | Two-Stage | configs/exp_027.yaml | 4 hours |
| exp_028 | Per-Sample Weights | configs/exp_028.yaml | 4 hours |
| exp_029 | MDN Head | configs/exp_029.yaml | 6 hours |
| exp_030 | Curriculum | configs/exp_030.yaml | 4 hours |

**Total:** ~23 hours SLURM time across 6 parallel experiments

## Success Criteria
- **Primary:** Outliers < 15% (from 23%)
- **Secondary:** NMAD stays < 0.020 (no regression)
- **Stretch:** Outliers < 10%
