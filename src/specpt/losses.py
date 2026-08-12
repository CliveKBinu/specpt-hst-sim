import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NMADLoss(nn.Module):
    def __init__(self, normalization_factor="std", eps=1e-8):
        super().__init__()
        self.normalization_factor = normalization_factor
        self.eps = eps

    def forward(self, z_pred, z_true):
        z_pred = z_pred.flatten()
        z_true = z_true.flatten()
        mad = torch.mean(torch.abs(z_pred - z_true))
        if self.normalization_factor == "std":
            normalization = torch.clamp(torch.std(z_true), min=self.eps)
        else:
            median = torch.median(z_true)
            normalization = torch.clamp(
                torch.median(torch.abs(z_true - median)), min=self.eps
            )
        return mad / normalization


class HuberNMADLoss(nn.Module):
    """Huber-variant of NMADLoss that downweights extreme outlier errors.

    Uses quadratic weighting for small errors (|Δz/(1+z)| < delta)
    and linear weighting for large errors (|Δz/(1+z)| >= delta).
    This prevents extreme outliers from dominating gradient updates.
    """

    def __init__(self, delta=0.15, normalization_factor="std", eps=1e-8):
        super().__init__()
        self.delta = delta
        self.normalization_factor = normalization_factor
        self.eps = eps

    def forward(self, z_pred, z_true):
        z_pred = z_pred.flatten()
        z_true = z_true.flatten()
        delz = (z_pred - z_true) / (1 + z_true)
        abs_delz = torch.abs(delz)
        # Huber: quadratic for small errors, linear for large
        huber = torch.where(
            abs_delz < self.delta,
            0.5 * abs_delz ** 2 / self.delta,
            abs_delz - 0.5 * self.delta,
        )
        mad = torch.mean(huber)
        if self.normalization_factor == "std":
            normalization = torch.clamp(torch.std(z_true), min=self.eps)
        else:
            median = torch.median(z_true)
            normalization = torch.clamp(
                torch.median(torch.abs(z_true - median)), min=self.eps
            )
        return mad / normalization


class MDNMADLoss(nn.Module):
    """Negative log-likelihood loss for Mixture Density Network.

    Combines Gaussian NLL with NMAD-style normalization.
    """

    def __init__(self, normalization_factor="std", eps=1e-8):
        super().__init__()
        self.normalization_factor = normalization_factor
        self.eps = eps

    def forward(self, means, log_vars, mix_weights, z_true):
        """
        Args:
            means: (batch_size, num_mixtures * output_features)
            log_vars: (batch_size, num_mixtures * output_features)
            mix_weights: (batch_size, num_mixtures)
            z_true: (batch_size, output_features)

        Returns:
            loss: scalar
        """
        batch_size = means.shape[0]
        num_mixtures = mix_weights.shape[1]
        output_features = means.shape[1] // num_mixtures

        # Reshape for easier computation
        means = means.view(batch_size, num_mixtures, output_features)
        log_vars = log_vars.view(batch_size, num_mixtures, output_features)
        z_true = z_true.view(batch_size, 1, output_features)

        # Compute Gaussian log-likelihood
        var = torch.exp(log_vars)
        log_prob = -0.5 * (
            torch.log(2 * torch.pi * var) + (z_true - means) ** 2 / var
        )
        # Sum over output features
        log_prob = log_prob.sum(dim=-1)  # (batch_size, num_mixtures)

        # Weight by mixture weights
        log_mix = torch.log(mix_weights + 1e-8)
        log_likelihood = log_prob + log_mix  # (batch_size, num_mixtures)

        # Log-sum-exp for numerical stability
        log_likelihood = torch.logsumexp(log_likelihood, dim=-1)  # (batch_size,)

        # Negative log-likelihood
        nll = -log_likelihood.mean()

        # Add NMAD-style normalization
        with torch.no_grad():
            # Use highest-weight component mean as prediction
            best_comp = torch.argmax(mix_weights, dim=-1)  # (batch_size,)
            pred = means[torch.arange(batch_size), best_comp, 0]
            delz = (pred - z_true.squeeze(1)) / (1 + z_true.squeeze(1))
            if self.normalization_factor == "std":
                normalization = torch.clamp(torch.std(z_true.squeeze()), min=self.eps)
            else:
                median = torch.median(z_true.squeeze())
                normalization = torch.clamp(
                    torch.median(torch.abs(z_true.squeeze() - median)), min=self.eps
                )

        return nll + torch.mean(torch.abs(delz)) / normalization


class BinnedRedshiftLoss(nn.Module):
    """Hybrid classification and within-bin refinement loss."""

    def __init__(
        self,
        num_bins=24,
        z_bin_max=3.0,
        lambda_refine=0.3,
        lambda_nmad=0.7,
        label_smoothing=0.05,
        huber_delta=None,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.z_bin_max = z_bin_max
        bin_edges_log1p = np.linspace(0.0, np.log1p(z_bin_max), num_bins + 1)
        self.register_buffer(
            "bin_left_log1p", torch.from_numpy(bin_edges_log1p[:-1]).float()
        )
        self.bin_width = float(bin_edges_log1p[1] - bin_edges_log1p[0])
        self.lambda_refine = lambda_refine
        self.lambda_nmad = lambda_nmad
        self.label_smoothing = label_smoothing
        self.huber_delta = huber_delta if huber_delta is not None else self.bin_width / 2.0

    def forward(self, outputs, target):
        conf_logits = outputs["conf_logits"]
        within_z = outputs["within_z"]
        z_true = target.squeeze(-1)

        target_log1p = torch.log1p(z_true)
        target_bin = torch.clamp(
            (target_log1p / self.bin_width).long(), 0, self.num_bins - 1
        )
        bin_indices = torch.arange(self.num_bins, device=conf_logits.device)
        dist = torch.abs(bin_indices.unsqueeze(0) - target_bin.unsqueeze(1)).float()
        gauss = torch.exp(-0.5 * (dist / 0.85) ** 2)
        gauss = gauss / gauss.sum(dim=-1, keepdim=True)
        soft_target = (
            (1.0 - self.label_smoothing) * gauss
            + self.label_smoothing / self.num_bins
        )
        log_probs = F.log_softmax(conf_logits, dim=-1)
        loss_cls = -(soft_target * log_probs).sum(dim=-1).mean()

        probabilities = F.softmax(conf_logits, dim=-1)
        residual_k = (within_z - z_true.unsqueeze(1)) / (1.0 + z_true.unsqueeze(1))
        huber_k = F.smooth_l1_loss(
            residual_k, torch.zeros_like(residual_k), reduction="none", beta=self.huber_delta
        )
        loss_refine = (probabilities * huber_k).sum(dim=-1).mean()

        z_pred_soft = (probabilities * within_z).sum(dim=-1)
        residual = (z_pred_soft - z_true) / (1.0 + z_true)
        loss_nmad = F.smooth_l1_loss(
            residual, torch.zeros_like(residual), reduction="mean", beta=self.huber_delta
        )
        total = loss_cls + self.lambda_refine * loss_refine + self.lambda_nmad * loss_nmad
        return {
            "total": total,
            "loss_cls": loss_cls.detach(),
            "loss_refine": loss_refine.detach(),
            "loss_nmad": loss_nmad.detach(),
        }
