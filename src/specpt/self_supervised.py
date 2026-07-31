"""Universal Spectral Encoder (USE): self-supervised spectral representation.

Label-free pretraining that preserves broad spectral information via a
latent-distillation anchor to a frozen autoencoder, plus masked / noise-view
robustness and two-view consistency.

Components
----------
- UniversalEncoderModel : trainable student + frozen teacher anchor
- SpectralViews          : masked / noise-corrupted view generation
- valid_pixel_mse        : reconstruction loss over in-band pixels only
- latent_consistency     : two-view latent agreement
- latent_distill         : student-to-teacher latent anchor

Reference: docs/universal_spectral_encoder.md
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UniversalEncoderModel(nn.Module):
    """Student SpecPT encoder/decoder with a frozen teacher encoder anchor.

    The student shares the frozen-architecture SpecPT dimensions and is
    initialized from the pretrained autoencoder checkpoint. The teacher is a
    frozen copy used only to compute the distillation anchor latent.
    """

    def __init__(self, student, teacher=None):
        super().__init__()
        self.student = student
        self.teacher = teacher
        if self.teacher is not None:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

    def encode(self, model, x):
        """[B, 7781] -> [B, 512] post-attention latent."""
        x = x.unsqueeze(1)
        x = model.forward_conv(x)
        x = x.flatten(start_dim=1)
        x = model.proj_to_d_model(x)
        x = x.unsqueeze(0)
        x = model.transformer_encoder(x)
        return x.squeeze(0)

    def student_latent(self, x):
        return self.encode(self.student, x)

    def teacher_latent(self, x):
        return self.encode(self.teacher, x)

    def reconstruct(self, x):
        """[B, 7781] -> [B, 7781] full student autoencoder reconstruction."""
        return self.student(x)


class SpectralViews(nn.Module):
    """Generates masked and noise-corrupted views of clean spectra.

    The observed spectrum is the target for every view; no clean/denoised
    target is required.
    """

    def __init__(self, mask_chunks=(2, 5), mask_size=(30, 100),
                 noise_sigma=(0.01, 0.05), seed=None):
        super().__init__()
        self.mask_chunks = mask_chunks
        self.mask_size = mask_size
        self.noise_sigma = noise_sigma
        if seed is not None:
            torch.manual_seed(seed)

    def _mask_view(self, x):
        """Zero-out random contiguous chunks along the wavelength axis."""
        out = x.clone()
        b, n = x.shape
        n_chunks = torch.randint(self.mask_chunks[0], self.mask_chunks[1] + 1, (1,)).item()
        for _ in range(n_chunks):
            size = torch.randint(self.mask_size[0], self.mask_size[1] + 1, (1,)).item()
            start = torch.randint(0, n - size, (1,)).item()
            out[:, start:start + size] = 0.0
        return out

    def _noise_view(self, x):
        sigma = torch.empty(1).uniform_(*self.noise_sigma).item()
        return x + torch.randn_like(x) * sigma

    def forward(self, x):
        """Returns a dict of views: clean, masked, noisy1, noisy2."""
        return {
            "clean": x,
            "masked": self._mask_view(x),
            "noisy1": self._noise_view(x),
            "noisy2": self._noise_view(x),
        }


def valid_pixel_mse(pred, target, valid_mask):
    """MSE over in-band (non-padding) pixels only.

    Args:
        pred, target: [B, N]
        valid_mask: [N] boolean tensor (True = in-band pixel)
    """
    return F.mse_loss(pred[:, valid_mask], target[:, valid_mask])


def latent_consistency(h1, h2):
    """L2 distance between two L2-normalized latents."""
    return F.mse_loss(F.normalize(h1, p=2, dim=-1),
                      F.normalize(h2, p=2, dim=-1))


def latent_distill(h_student, h_teacher):
    """L2 distance between normalized student and teacher latents."""
    return F.mse_loss(F.normalize(h_student, p=2, dim=-1),
                      F.normalize(h_teacher, p=2, dim=-1))


def cosine_similarity(a, b):
    """Mean per-sample cosine similarity between two [B, D] tensors."""
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    return (a * b).sum(dim=-1).mean().item()


def build_use_model(ae_ckpt_path, teacher=True, device="cpu"):
    """Build a UniversalEncoderModel from the regridded autoencoder checkpoint.

    The student is initialized from the same weights as the (optional) frozen
    teacher, so it starts at a strong reconstruction point.
    """
    from .model import SpecPT

    def _build():
        sp = SpecPT(input_size=7781, d_model=512, nhead=8,
                    num_encoder_layers=3, num_decoder_layers=3,
                    dim_feedforward=2048, dropout=0.1)
        ckpt = torch.load(ae_ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        sp.load_state_dict(state, strict=True)
        return sp

    student = _build().to(device)
    t = _build().to(device) if teacher else None
    return UniversalEncoderModel(student, teacher=t)
