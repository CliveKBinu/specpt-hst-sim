import torch
import torch.nn as nn


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
