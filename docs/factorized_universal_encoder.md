# Factorized Universal Spectral Encoder

A trainable encoder with two explicit outputs:

```text
spectrum
   -> shared spectral encoder
        ├── h_universal -> reconstruction decoder
        │                 future SFR/AGN/task heads
        │
        └── h_z         -> redshift head
```

**Central rule:** `h_universal` stays broad and reconstruction-driven (anchored to
the frozen AE); `h_z` is the only path that receives direct redshift supervision.
This prevents the whole latent from collapsing onto the z axis (the failure mode
of exp_048/051) while still letting z supervision shape a task-specific pathway.

## Motivation

All downstream axes on the *frozen* AE latent are exhausted:

| Axis | Experiments | Result |
|------|-------------|--------|
| Head capacity | exp_035-043 | plateau ~0.21-0.27 NMAD |
| Trees (shrinkage) | exp_045/046 | 0.2077 but R² 0.094 (prediction collapse) |
| Loss function | exp_047 | NMADLoss confirmed optimal |
| Unfrozen regression | exp_048/048b | encoder drift, best epoch 1 |
| Contrastive (RNC) | exp_050/051 | flat loss / 126x AE drift |
| Self-supervised robustness | USE A/B/C | preserves latent, z tied at ~0.217 |

USE proved the **anchor mechanism** (frozen-teacher distillation + reconstruction
+ drift gate) can train an encoder without destroying AE identity (cos ~0.995,
drift ~1.0). The factorized encoder reuses that mechanism and adds z supervision
to a **separate** task branch.

## Architecture

### Shared encoder (unchanged SpecPT dims)

```text
input: [B, 7781]
    -> Conv1/BN/ReLU -> Conv2/BN/ReLU -> Conv3/BN/ReLU -> MaxPool
    -> flatten -> Linear(..., 512) -> TransformerEncoder
    -> h_universal: [B, 512]
```

Frozen dims (checkpoint-compatible with the pretrained AE):

```text
input_size=7781  d_model=512  nhead=8
num_encoder_layers=3  num_decoder_layers=3  dim_feedforward=2048
```

Initialization: student from the regridded AE (or a USE checkpoint via
`--init_ckpt`); teacher = frozen AE copy (anchor only).

### Universal reconstruction path

```text
h_universal -> TransformerDecoder -> Linear layers -> reconstructed spectrum
```

### Redshift path (early z branch)

A small trainable branch reads the **post-pool, pre-projection** conv map
`[B, 256, 487]` where wavelength-localized line/continuum structure still lives:

```text
conv map [B, 256, 487]
    -> 1x1 conv 256->64 -> BN -> ReLU
    -> depthwise conv k=15, groups=64 -> BN -> ReLU
    -> global avg pool (in-band mask applied)
    -> MLP 64->128->128
    -> h_z [B, 128]
    -> Linear(128, 1) + Softplus -> z
```

A static downsampled in-band mask (from `downsample_valid_mask`) zeroes out
padding bands before pooling. The branch is intentionally small (~40k params).

## Loss

```text
L_universal = lambda_mask  * recon(masked -> clean)
            + lambda_noise * recon(noisy  -> clean)
            + lambda_cons  * consistency(h_noisy1, h_noisy2)
            + lambda_dist  * distill(h_shared, h_teacher)

L_task = lambda_z * NMADLoss(z_pred, z_true)     # ramped over z_ramp_epochs

L_total = L_universal + L_task
```

- Reconstruction: valid-pixel-only MSE (excludes padding wavelengths).
- Distill: L2 on L2-normalized `h_shared` vs frozen AE teacher latent.
- z loss: `NMADLoss` (the exp_047-validated choice). HuberNMAD is off the table.
- No orthogonality/separating loss in early stages: with only z labels we cannot
  prove separation is SFR/AGN-useful, and it adds a failure mode. Monitor
  shared-vs-z correlation and shared-latent probe instead.

## Stages

| Stage | Data | Shared encoder | z branch | Purpose |
|-------|------|----------------|----------|---------|
| 1 | sim | frozen | trainable | implementation / signal test (clean z) |
| 2 | real | frozen | trainable | does early conv info carry real z? |
| 3 | joint sim+real | trainable @ 1e-5 | trainable @ 3e-4 | the actual factorization |

Stage 3 keeps the recon + distill anchor **on**, and ramps `lambda_z` 0 -> target
over the first 10 epochs to avoid the exp_048/051 immediate-drift failure.

## Evaluation

`scripts/eval_factorized_encoder.py` reports on the held-out real test set:

- Shared-latent recon MSE/cosine vs AE baseline
- Latent stability (masked/noisy vs clean) + teacher drift
- Redshift from three pathways: frozen z_head direct, linear probe on h_z,
  linear probe on h_universal
- Anti-shrinkage: R², prediction range, pred/target std ratio
- Bootstrap 95% CI on test NMAD (z_head pathway)
- SNR-bucketed NMAD/η/R² (2.5-5, 5-10, 10-20, 20+)

## Gates

- **Reject** any NMAD improvement with R² ~ 0 or collapsed prediction range
  (exp_045 RF was shrinkage: 0.2077 but R² 0.094).
- Stage 3: recon drift <= 2.0x, shared->teacher cos >= 0.95, no best-epoch-1.
- Real test NMAD must beat 0.2162 (AE frozen-latent baseline) with positive R².

## Checkpoint format

```python
{
    "args": ...,
    "epoch": ...,
    "model_state_dict": model.student.state_dict(),
    "z_branch_state_dict": model.z_branch.state_dict(),
    "z_head_state_dict": model.z_head.state_dict(),
    "best_val_nmad": ...,
    "recon_mse_drift": ...,
    "latent_cos_to_teacher": ...,
}
```

The original AE and USE checkpoints are never modified.

## Files

```text
src/specpt/factorized.py                 model + z branch + build/load helpers
scripts/pretrain_factorized_encoder.py   staged pretraining (sim/real/joint)
scripts/eval_factorized_encoder.py       full evaluation harness
scripts/slurm_pretrain_factorized.sh     SLURM wrapper
scripts/slurm_eval_factorized.sh         SLURM wrapper
```

## Status

- [ ] Stage 1 (sim, frozen shared) — implementation/signal test
- [ ] Stage 2 (real, frozen shared) — early-feature diagnostic
- [ ] Stage 3 (joint, unfrozen) — factorization
