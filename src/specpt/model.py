import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
import numpy as np
from scipy import signal


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ImprovedResidualMLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.swish = Swish()
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.residual_proj = (
            nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        )

    def forward(self, x):
        residual = x
        x = self.swish(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        x = x + residual
        x = self.layer_norm(x)
        return self.swish(x)


class SpecPT(nn.Module):
    def __init__(
        self,
        input_size=7781,
        d_model=512,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=41, stride=2, padding=20)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=21, stride=2, padding=10)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=11, stride=2, padding=5)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        dummy_input = torch.zeros(1, 1, input_size)
        dummy_output = self.forward_conv(dummy_input)
        output_size = dummy_output.numel() // dummy_input.shape[0]

        self.proj_to_d_model = nn.Linear(output_size, d_model)
        self.linear1 = nn.Linear(d_model, output_size)
        self.linear2 = nn.Linear(output_size, input_size)

    def forward_conv(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.forward_conv(x)
        x = x.flatten(start_dim=1)
        x = self.proj_to_d_model(x)
        x = x.unsqueeze(0)
        encoded_features = self.transformer_encoder(x)
        decoded_features = self.transformer_decoder(
            encoded_features, encoded_features
        )
        x = decoded_features.squeeze(0)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class EnhancedSpecPTForRedshift(nn.Module):
    def __init__(
        self,
        pretrained_model,
        output_features=1,
        num_mlp_blocks=5,
        mlp_dim=512,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.encoder = pretrained_model.transformer_encoder
        self.proj_to_d_model = pretrained_model.proj_to_d_model
        self.pretrained_model = pretrained_model

        for param in list(self.encoder.parameters())[-4:]:
            param.requires_grad = True

        self.mlp_blocks = nn.Sequential(
            *[
                ImprovedResidualMLPBlock(
                    mlp_dim if i > 0 else 512, mlp_dim, dropout_rate
                )
                for i in range(num_mlp_blocks)
            ]
        )

        self.prediction = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim // 2),
            Swish(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim // 2, output_features),
            nn.Softplus(),
        )

        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pretrained_model.forward_conv(x)
        x = x.flatten(start_dim=1)
        x = self.proj_to_d_model(x)
        x = x.unsqueeze(0)
        encoded_features = self.encoder(x)
        encoded_features = encoded_features.squeeze(0)
        attn_output, _ = self.attention(
            encoded_features, encoded_features, encoded_features
        )
        x = attn_output + encoded_features
        x = self.mlp_blocks(x)
        redshift = self.prediction(x)
        return redshift


class MDNHead(nn.Module):
    """Mixture Density Network head for uncertainty-aware redshift prediction.

    Instead of predicting a single z value, outputs K Gaussian mixture components
    with means, log-variances, and mixture weights.
    """

    def __init__(self, input_dim, output_features=1, num_mixtures=5):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.output_features = output_features

        # Means: K means for each output feature
        self.means = nn.Linear(input_dim, num_mixtures * output_features)
        # Log-variances: K log-variances for each output feature
        self.log_vars = nn.Linear(input_dim, num_mixtures * output_features)
        # Mixture weights: K weights (softmax normalized)
        self.mix_weights = nn.Linear(input_dim, num_mixtures)

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim) features from MLP blocks

        Returns:
            means: (batch_size, num_mixtures * output_features)
            log_vars: (batch_size, num_mixtures * output_features)
            mix_weights: (batch_size, num_mixtures) - softmax normalized
        """
        means = self.means(x)
        log_vars = self.log_vars(x)
        mix_weights = F.softmax(self.mix_weights(x), dim=-1)
        return means, log_vars, mix_weights


class EnhancedSpecPTForRedshiftMDN(nn.Module):
    """SpecPT with MDN head for uncertainty-aware redshift prediction."""

    def __init__(
        self,
        pretrained_model,
        output_features=1,
        num_mlp_blocks=5,
        mlp_dim=512,
        dropout_rate=0.2,
        num_mixtures=5,
    ):
        super().__init__()
        self.encoder = pretrained_model.transformer_encoder
        self.proj_to_d_model = pretrained_model.proj_to_d_model
        self.pretrained_model = pretrained_model

        for param in list(self.encoder.parameters())[-4:]:
            param.requires_grad = True

        self.mlp_blocks = nn.Sequential(
            *[
                ImprovedResidualMLPBlock(
                    mlp_dim if i > 0 else 512, mlp_dim, dropout_rate
                )
                for i in range(num_mlp_blocks)
            ]
        )

        self.mdn_head = MDNHead(mlp_dim, output_features, num_mixtures)
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

    def forward(self, x):
        """
        Returns:
            means: (batch_size, num_mixtures * output_features)
            log_vars: (batch_size, num_mixtures * output_features)
            mix_weights: (batch_size, num_mixtures)
        """
        x = x.unsqueeze(1)
        x = self.pretrained_model.forward_conv(x)
        x = x.flatten(start_dim=1)
        x = self.proj_to_d_model(x)
        x = x.unsqueeze(0)
        encoded_features = self.encoder(x)
        encoded_features = encoded_features.squeeze(0)
        attn_output, _ = self.attention(
            encoded_features, encoded_features, encoded_features
        )
        x = attn_output + encoded_features
        x = self.mlp_blocks(x)
        means, log_vars, mix_weights = self.mdn_head(x)
        return means, log_vars, mix_weights


class SpectrumNormalizer:
    @staticmethod
    def median_normalize(spectrum):
        median_val = np.nanmedian(spectrum)
        if median_val > 0 and np.isfinite(median_val):
            return spectrum / median_val
        return np.array(spectrum, dtype=float)

    @staticmethod
    def minmax_normalize(spectrum, feature_range=(0, 1)):
        min_val = np.min(spectrum)
        max_val = np.max(spectrum)
        if max_val > min_val:
            normalized = (spectrum - min_val) / (max_val - min_val)
            return normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
        return spectrum

    @staticmethod
    def zscore_normalize(spectrum):
        clean = spectrum[~np.isnan(spectrum)]
        if len(clean) == 0:
            return np.zeros_like(spectrum, dtype=np.float32)
        std = np.std(clean)
        if std > 0:
            mean = np.mean(clean)
            normalized = (spectrum - mean) / std
            return np.nan_to_num(normalized, nan=0.0)
        return np.nan_to_num(spectrum, nan=0.0).astype(np.float32)

    @staticmethod
    def robust_normalize(spectrum):
        q25, q75 = np.percentile(spectrum, [25, 75])
        if q75 > q25:
            iqr = q75 - q25
            return (spectrum - q25) / iqr
        return spectrum

    @staticmethod
    def continuum_normalize(spectrum, polyorder=5):
        x = np.arange(len(spectrum))
        mask = ~np.isnan(spectrum)
        if np.sum(mask) > polyorder + 1:
            coeffs = np.polyfit(x[mask], spectrum[mask], polyorder)
            continuum = np.polyval(coeffs, x)
            if not np.any(continuum == 0):
                return spectrum / continuum
        return spectrum

    @staticmethod
    def snr_weighted_normalize(spectrum, window_length=51):
        smoothed = signal.savgol_filter(spectrum, window_length, 3)
        noise = spectrum - smoothed
        snr = np.abs(smoothed) / (np.std(noise) + 1e-10)
        weighted_spectrum = spectrum * (snr / (np.max(snr) + 1e-10))
        return weighted_spectrum

    @staticmethod
    def log_transform(spectrum, offset=1.0):
        if np.min(spectrum) < 0:
            offset = abs(np.min(spectrum)) + 1.0
        return np.log1p(spectrum + offset)

    @staticmethod
    def mad_normalize(spectrum):
        median = np.median(spectrum)
        mad = np.median(np.abs(spectrum - median))
        if mad > 0:
            return (spectrum - median) / mad
        return spectrum

    @staticmethod
    def wavelength_dependent_normalize(spectrum, wavelengths):
        segments = 5
        segment_length = len(spectrum) // segments
        normalized = np.zeros_like(spectrum)
        for i in range(segments):
            start = i * segment_length
            end = start + segment_length if i < segments - 1 else len(spectrum)
            segment = spectrum[start:end]
            if np.std(segment) > 0:
                normalized[start:end] = (segment - np.mean(segment)) / np.std(segment)
            else:
                normalized[start:end] = segment
        return normalized
