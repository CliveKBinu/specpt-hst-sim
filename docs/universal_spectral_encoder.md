# Universal Spectral Encoder (USE)

A reusable, self-supervised spectral representation for SpecPT. The encoder is
pretrained **without any science labels** so it can later serve arbitrary
downstream heads — redshift, SFR, AGN classification, stellar mass, emission-line
measurements — by attaching a task-specific head to a frozen latent.

## Motivation

The original SpecPT autoencoder is trained purely for reconstruction. It captures
broad spectral information (continuum, emission lines, morphology), but that
information is **entangled, not organized for any prediction task**. Nine
experiments (exp_035–exp_051) demonstrated that no downstream method — linear
probe, MLP, ResNet, Random Forest, metric learning, or Rank-N-Contrast — can
extract z-discriminative structure that the reconstruction objective never
explicitly organized. Unfreezing the encoder for regression (exp_048/048b) or
contrastive learning (exp_051) destroys the autoencoder identity via drift.

The Universal Spectral Encoder addresses this with **label-free pretraining**:

1. Reconstruction is preserved through a **latent-distillation anchor** to the
   frozen pretrained autoencoder, preventing catastrophic drift.
2. Robustness to noise and missing data is learned via **masked / noise-corrupted
   views** — no clean "denoised" target is required; the observed spectrum is the
   target.
3. **Two-view consistency** teaches the latent to ignore perturbation-specific
   noise and retain stable spectral structure.
4. Redshift (or any future task) is added only **afterwards**, as a frozen-latent
   head, so the representation is never organized around a single science label.

## Design

```
observed spectrum (z-score normalized, NaN-masked at [11000, 16500] Å)
        │
        ├── clean view ──────────────► student encoder ──► h_clean
        │                                                     │
        ├── masked view ─────────────► student encoder ──► h_masked ──► decoder ──► recon(x)
        │                                                     │
        └── noisy view(s) ───────────► student encoder ──► h_noisy  ──► decoder ──► recon(x)
                                                            │  └────────┐
                                        consistency loss ◄──┘           ▼
                                                        distill loss ◄── teacher latent
                                                                       (frozen AE encoder(x))
```

### Training objective

```
L = λ_mask · L_recon(masked → x)
  + λ_noise · L_recon(noisy → x)
  + λ_consistency · ||N(h_noisy1) − N(h_noisy2)||²
  + λ_distill · ||N(h_student) − N(h_teacher)||²
```

where:

- `L_recon` is **valid-pixel MSE**: reconstruction error averaged only over the
  in-band wavelengths (excludes the padding region [<11000, >16500] Å).
- `N(·)` is L2 normalization.
- `h_teacher = frozenAE.encoder(x_clean)` is the anchor latent from the original
  pretrained autoencoder, computed once per batch (teacher never trained).

### Why this avoids the previous failure modes

| Prior failure | How USE avoids it |
|---------------|-------------------|
| RNC flat loss on frozen features (exp_050) | No reliance on pre-existing z-ordering; learns robust spectral structure, not a z-rank geometry |
| Encoder drift under regression gradient (exp_048) | Reconstruction + distillation anchor keeps the student close to the AE; drift is gated and monitored |
| Encoder drift under contrastive gradient (exp_051, 126×) | Same anchor; contrastive term (consistency) is secondary and only between views of the same spectrum |
| Single-task bias | No labels are used in pretraining at all |

## Architecture

- **Student encoder + decoder**: identical `SpecPT` architecture
  (`input_size=7781, d_model=512, nhead=8, 3 enc layers, 3 dec layers,
  dim_feedforward=2048, dropout=0.1`). Frozen architecture params are unchanged.
- **Initialization**: student weights init from the regridded autoencoder
  checkpoint (`checkpoints/autoencoder_regrid_autoencoder_best.pth`) so it starts
  from a strong reconstruction point.
- **Teacher**: a frozen copy of the same checkpoint used only for the
  distillation anchor.
- **New checkpoint**: the trained student is saved under its own name
  (e.g. `checkpoints/use_stageA_best.pth`). The original autoencoder checkpoint
  is never modified.

## Training curriculum

The objective is enabled in stages so failures are diagnosable:

| Stage | Losses | Purpose |
|-------|--------|---------|
| A | mask recon + distill | Verify masked reconstruction pipeline, confirm no drift |
| B | + noise recon | Learn robustness to realistic noise corruption |
| C | + two-view consistency | Stabilize latent under perturbations |

Each stage is a separate experiment and checkpoint.

## Data

- **Source**: real 3D-HST grism spectra (`grism_specPT_v5.pkl`, `SNR ≥ 2.5`).
  Sim data is not required — no labels are needed.
- **Split**: `split_by_grism_id` (seed 42, val/test 0.1/0.1). Only train + val are
  used for pretraining; **test is held out** so downstream head evaluation is clean.
- **Normalization**: `SpectrumNormalizer.zscore_normalize` (same as the standard
  real-data pipeline).
- **Masking**: NaNs at padding wavelengths [<11000 Å, >16500 Å] are masked in the
  reconstruction loss (valid-pixel MSE).

## How to use

### 1. Pretrain

```bash
# Stage A: masked reconstruction + distillation anchor
python scripts/pretrain_universal_encoder.py \
    --exp_name use_stageA \
    --stage A \
    --epochs 100 --batch_size 64 --lr 1e-4

# Stage B: add noise reconstruction
python scripts/pretrain_universal_encoder.py \
    --exp_name use_stageB \
    --stage B --resume checkpoints/use_stageA_best.pth

# Stage C: add two-view consistency
python scripts/pretrain_universal_encoder.py \
    --exp_name use_stageC \
    --stage C --resume checkpoints/use_stageB_best.pth
```

### 2. Evaluate the frozen latent

```bash
python scripts/eval_universal_latent.py --exp_name use_stageC
```

Reports:
- Reconstruction MSE / cosine similarity on held-out test spectra
- Latent stability under masking / noise (mean latent cosine to clean view)
- Drift vs. original AE latent
- Linear-probe redshift head: NMAD, η, RMSE, R²
- SNR-bucketed NMAD (2.5–5, 5–10, 10–20, 20+)

### 3. Attach a downstream head

Future task heads (redshift, SFR, AGN, ...) freeze the USE encoder and train a
task head on `h = USE.encoder(x)`. For redshift:

```bash
python scripts/finetune_universal_redshift.py --ckpt checkpoints/use_stageC_best.pth
```

## Quality gates

The pretrained encoder is only accepted if:

- Clean-input reconstruction MSE ≤ 2× the frozen-AE baseline
- Reconstruction cosine similarity stays high
- Latent drift (vs. teacher) is bounded
- Masked / noisy reconstruction improves over a frozen-AE baseline on corrupted input
- Latent is stable: two views of the same spectrum give close latents
- A frozen-latent linear probe does not rely on prediction-range collapse
  (validate NMAD against R²; negative R² = shrinkage artifact, not learning)

## Files

| File | Purpose |
|------|---------|
| `src/specpt/self_supervised.py` | USE model wrapper, view transforms, losses |
| `scripts/pretrain_universal_encoder.py` | Self-supervised pretraining |
| `scripts/slurm_pretrain_universal.sh` | SLURM submission wrapper |
| `scripts/eval_universal_latent.py` | Frozen-latent evaluation + linear probe |
| `docs/universal_spectral_encoder.md` | This document |

## Future work

- SFR / AGN heads once labels exist (freeze encoder, train head, evaluate latent
  sufficiency per task).
- Adapter on the earlier conv feature map if a task needs wavelength-localized
  line information beyond the 512-d latent.
- Sim-data pretraining to enlarge the unlabeled corpus (with `split_by_grism_id`).
