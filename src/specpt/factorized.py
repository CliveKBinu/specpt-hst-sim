"""Factorized Universal Spectral Encoder.

A trainable encoder with two explicit outputs:

    spectrum
       -> shared spectral encoder
            ├── h_universal -> reconstruction decoder
            │                 future SFR/AGN/task heads
            │
            └── h_z         -> redshift head

Design rules (docs/factorized_universal_encoder.md):
- The shared encoder reuses the frozen SpecPT architecture dimensions so the
  pretrained autoencoder checkpoint loads unchanged.
- h_universal is anchored to the frozen AE teacher latent and reconstruction,
  keeping the broad spectral objective.
- h_z comes from a small trainable branch on the early convolutional feature
  map, so z supervision shapes a task-specific pathway without overwriting the
  universal latent.
"""
import os
import numpy as np
import torch
import torch.nn as nn


class ZBranch(nn.Module):
    """Early spectral branch: conv map [B, C, B] -> h_z [B, z_dim].

    Reads the post-pool, pre-projection convolutional feature map where
    wavelength-localized line/continuum structure still lives. A static
    downsampled in-band mask zeroes out padding bands before global pooling.
    """

    def __init__(self, in_channels=256, n_bands=487, z_dim=128, band_mask=None):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=15, padding=7, groups=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, z_dim),
        )
        if band_mask is not None:
            m = np.asarray(band_mask, dtype=np.float32)
            assert m.shape == (n_bands,), f"band_mask shape {m.shape} != {(n_bands,)}"
            self.register_buffer("band_mask", torch.from_numpy(m)[None, None, :])
        else:
            self.register_buffer("band_mask", torch.ones(1, 1, n_bands))

    def forward(self, conv_map):
        """[B, in_channels, n_bands] -> [B, z_dim]."""
        x = self.features(conv_map) * self.band_mask
        x = self.pool(x).squeeze(-1)
        return self.mlp(x)


def downsample_valid_mask(valid_input_mask, n_bands=487, thresh=0.3):
    """Bin a length-N valid-pixel mask into n_bands coverage fractions.

    Each output band receives the fraction of valid input pixels it covers.
    Bands with coverage below `thresh` are treated as padding and masked out.
    """
    n_in = len(valid_input_mask)
    coverage = np.zeros(n_bands)
    for k in range(n_bands):
        lo = int(k * n_in / n_bands)
        hi = int((k + 1) * n_in / n_bands)
        coverage[k] = np.asarray(valid_input_mask[lo:hi]).mean()
    return (coverage > thresh).astype(np.float32)


class FactorizedEncoderModel(nn.Module):
    """Student SpecPT + frozen teacher anchor + early z branch + z head.

    Attributes
    ----------
    student : SpecPT
        Trainable shared encoder/decoder backbone (init from AE / USE ckpt).
    teacher : SpecPT
        Frozen copy used only for the distillation anchor latent.
    z_branch : ZBranch
        Early spectral branch producing h_z.
    z_head : nn.Sequential
        Linear(z_dim, 1) + Softplus redshift head.
    """

    def __init__(self, student, teacher=None, z_dim=128, band_mask=None):
        super().__init__()
        self.student = student
        self.teacher = teacher
        if self.teacher is not None:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()
        self.z_branch = ZBranch(z_dim=z_dim, band_mask=band_mask)
        self.z_head = nn.Sequential(nn.Linear(z_dim, 1), nn.Softplus())

    # ---- encoding paths ----
    def _encode(self, model, x):
        x = x.unsqueeze(1)
        x = model.forward_conv(x)
        x = x.flatten(start_dim=1)
        x = model.proj_to_d_model(x)
        x = x.unsqueeze(0)
        x = model.transformer_encoder(x)
        return x.squeeze(0)

    def shared_latent(self, x):
        """[B, 7781] -> [B, 512] h_universal (post-attention)."""
        return self._encode(self.student, x)

    def teacher_latent(self, x):
        return self._encode(self.teacher, x)

    def z_latent(self, x):
        """[B, 7781] -> [B, z_dim] h_z from the early conv map."""
        conv = self.student.forward_conv(x.unsqueeze(1))
        return self.z_branch(conv)

    def z_pred(self, x):
        """[B, 7781] -> [B, 1] redshift prediction (Softplus)."""
        return self.z_head(self.z_latent(x))

    def reconstruct(self, x):
        """[B, 7781] -> [B, 7781] full student autoencoder reconstruction."""
        return self.student(x)

    # ---- parameter management ----
    def set_shared_trainable(self, trainable):
        for p in self.student.parameters():
            p.requires_grad = bool(trainable)

    def trainable_param_groups(self, encoder_lr, z_lr, weight_decay):
        encoder_params = [p for p in self.student.parameters() if p.requires_grad]
        z_params = [p for p in self.z_branch.parameters() if p.requires_grad] \
            + [p for p in self.z_head.parameters() if p.requires_grad]
        groups = []
        if encoder_params:
            groups.append({"params": encoder_params, "lr": encoder_lr})
        if z_params:
            groups.append({"params": z_params, "lr": z_lr})
        return groups, weight_decay

    @property
    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def n_total(self):
        return sum(p.numel() for p in self.parameters())


def _load_specpt(ckpt_path, device="cpu"):
    from .model import SpecPT
    sp = SpecPT(input_size=7781, d_model=512, nhead=8,
                num_encoder_layers=3, num_decoder_layers=3,
                dim_feedforward=2048, dropout=0.1)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    sp.load_state_dict(state, strict=True)
    return sp.to(device)


def build_factorized_model(ae_ckpt_path, init_ckpt=None, z_dim=128,
                           band_mask=None, device="cpu"):
    """Build a FactorizedEncoderModel.

    Student initializes from the pretrained AE checkpoint (or `init_ckpt`,
    e.g. a USE checkpoint), teacher is a frozen copy of the AE. Any z_branch /
    z_head weights present in `init_ckpt` are loaded too.
    """
    student = _load_specpt(ae_ckpt_path, device)
    teacher = _load_specpt(ae_ckpt_path, device)
    model = FactorizedEncoderModel(student, teacher, z_dim=z_dim,
                                   band_mask=band_mask).to(device)
    if init_ckpt and os.path.exists(init_ckpt):
        ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        st = ck.get("model_state_dict", ck)
        missing, unexpected = model.student.load_state_dict(st, strict=False)
        if missing:
            print(f"  [init] student missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  [init] student unexpected keys: {unexpected[:5]}...")
        if "z_branch_state_dict" in ck:
            model.z_branch.load_state_dict(ck["z_branch_state_dict"], strict=True)
            print("  [init] loaded z_branch from checkpoint")
        if "z_head_state_dict" in ck:
            model.z_head.load_state_dict(ck["z_head_state_dict"], strict=True)
            print("  [init] loaded z_head from checkpoint")
    return model
