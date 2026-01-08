"""
ATAT: Astronomical Transformer for time series And Tabular data

Implementation based on: https://arxiv.org/abs/2405.03078

Key components:
1. Time Modulation (TM): Fourier-based temporal encoding
2. Quantile Feature Tokenizer (QFT): Tabular feature embedding
3. Light-curve Transformer: Processes time series
4. Tabular Transformer: Processes features
5. Multi-modal fusion MLP

Simplified for MALLORN dataset (6 bands, binary classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import QuantileTransformer


class TimeModulation(nn.Module):
    """
    Time Modulation mechanism for encoding observation times.

    Uses learnable Fourier series to modulate flux embeddings based on time.
    This handles irregular sampling naturally.
    """

    def __init__(
        self,
        input_dim: int = 2,  # (flux, flux_err)
        embed_dim: int = 64,
        n_harmonics: int = 32,
        t_max: float = 1500.0,
        n_bands: int = 6
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_harmonics = n_harmonics
        self.t_max = t_max
        self.n_bands = n_bands

        # Linear projection for flux input
        self.flux_proj = nn.Linear(input_dim, embed_dim)

        # Learnable Fourier coefficients per band
        # gamma_1 coefficients (for multiplicative modulation)
        self.alpha1 = nn.Parameter(torch.randn(n_bands, n_harmonics, embed_dim) * 0.01)
        self.beta1 = nn.Parameter(torch.randn(n_bands, n_harmonics, embed_dim) * 0.01)

        # gamma_2 coefficients (for additive bias)
        self.alpha2 = nn.Parameter(torch.randn(n_bands, n_harmonics, embed_dim) * 0.01)
        self.beta2 = nn.Parameter(torch.randn(n_bands, n_harmonics, embed_dim) * 0.01)

    def forward(self, flux: torch.Tensor, time: torch.Tensor, band_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flux: (batch, seq_len, 2) - flux and flux_err
            time: (batch, seq_len) - observation times
            band_idx: (batch, seq_len) - band indices (0-5)

        Returns:
            (batch, seq_len, embed_dim) - time-modulated embeddings
        """
        batch_size, seq_len, _ = flux.shape

        # Project flux to embedding dimension
        x = self.flux_proj(flux)  # (batch, seq_len, embed_dim)

        # Compute Fourier features for each observation
        # Normalize time to [0, 1]
        t_norm = time / self.t_max  # (batch, seq_len)

        # Compute harmonic frequencies: 2*pi*h*t for h = 1..H
        h = torch.arange(1, self.n_harmonics + 1, device=flux.device).float()  # (H,)
        phases = 2 * np.pi * h.unsqueeze(0).unsqueeze(0) * t_norm.unsqueeze(-1)  # (batch, seq_len, H)

        sin_phases = torch.sin(phases)  # (batch, seq_len, H)
        cos_phases = torch.cos(phases)  # (batch, seq_len, H)

        # Get band-specific coefficients
        # band_idx: (batch, seq_len) -> need to gather coefficients
        band_idx_exp = band_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.n_harmonics, self.embed_dim)

        alpha1 = self.alpha1.unsqueeze(0).expand(batch_size, -1, -1, -1)
        beta1 = self.beta1.unsqueeze(0).expand(batch_size, -1, -1, -1)
        alpha2 = self.alpha2.unsqueeze(0).expand(batch_size, -1, -1, -1)
        beta2 = self.beta2.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Gather band-specific coefficients
        alpha1_b = torch.gather(alpha1, 1, band_idx_exp)  # (batch, seq_len, H, embed_dim)
        beta1_b = torch.gather(beta1, 1, band_idx_exp)
        alpha2_b = torch.gather(alpha2, 1, band_idx_exp)
        beta2_b = torch.gather(beta2, 1, band_idx_exp)

        # Compute gamma_1 and gamma_2 (Fourier series)
        sin_exp = sin_phases.unsqueeze(-1)  # (batch, seq_len, H, 1)
        cos_exp = cos_phases.unsqueeze(-1)  # (batch, seq_len, H, 1)

        gamma1 = (alpha1_b * sin_exp + beta1_b * cos_exp).sum(dim=2)  # (batch, seq_len, embed_dim)
        gamma2 = (alpha2_b * sin_exp + beta2_b * cos_exp).sum(dim=2)  # (batch, seq_len, embed_dim)

        # Apply time modulation: x * gamma1 + gamma2
        output = x * (1 + gamma1) + gamma2

        return output


class QuantileFeatureTokenizer(nn.Module):
    """
    Quantile Feature Tokenizer for tabular data.

    Maps each scalar feature to an embedding via:
    1. Quantile transformation (fitted on training data)
    2. Learnable affine projection
    """

    def __init__(self, n_features: int, embed_dim: int = 32):
        super().__init__()
        self.n_features = n_features
        self.embed_dim = embed_dim

        # Learnable projection for each feature
        self.weights = nn.Parameter(torch.randn(n_features, embed_dim) * 0.01)
        self.biases = nn.Parameter(torch.zeros(n_features, embed_dim))

        # Quantile transformer (fitted externally)
        self.qt = None

    def fit_quantile_transform(self, features: np.ndarray):
        """Fit quantile transformer on training features."""
        self.qt = QuantileTransformer(output_distribution='normal', random_state=42)
        self.qt.fit(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, n_features) - already quantile-transformed

        Returns:
            (batch, n_features, embed_dim) - feature embeddings
        """
        # Expand features for element-wise multiplication
        # features: (batch, n_features) -> (batch, n_features, 1)
        f_exp = features.unsqueeze(-1)

        # Apply affine transformation per feature
        # weights: (n_features, embed_dim)
        # output: (batch, n_features, embed_dim)
        output = f_exp * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)

        return output


class ATATLightCurveEncoder(nn.Module):
    """
    Transformer encoder for light curves with time modulation.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.2,
        max_seq_len: int = 400,
        n_bands: int = 6
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Time modulation
        self.time_mod = TimeModulation(
            input_dim=2,
            embed_dim=embed_dim,
            n_harmonics=32,
            t_max=1500.0,
            n_bands=n_bands
        )

        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer norm
        self.ln = nn.LayerNorm(embed_dim)

    def forward(
        self,
        flux: torch.Tensor,
        time: torch.Tensor,
        band_idx: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            flux: (batch, seq_len, 2) - flux and flux_err
            time: (batch, seq_len) - observation times
            band_idx: (batch, seq_len) - band indices
            mask: (batch, seq_len) - 1 for valid, 0 for padding

        Returns:
            (batch, embed_dim) - classification embedding
        """
        batch_size = flux.shape[0]

        # Apply time modulation
        x = self.time_mod(flux, time, band_idx)  # (batch, seq_len, embed_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, embed_dim)

        # Update mask for CLS token
        cls_mask = torch.ones(batch_size, 1, device=mask.device)
        mask = torch.cat([cls_mask, mask], dim=1)

        # Create attention mask (True = ignore)
        attn_mask = (1 - mask).bool()

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Extract CLS token output
        cls_output = x[:, 0, :]  # (batch, embed_dim)

        return self.ln(cls_output)


class ATATTabularEncoder(nn.Module):
    """
    Transformer encoder for tabular features with QFT.
    """

    def __init__(
        self,
        n_features: int,
        embed_dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Quantile Feature Tokenizer
        self.qft = QuantileFeatureTokenizer(n_features, embed_dim)

        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer norm
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, n_features) - quantile-transformed features

        Returns:
            (batch, embed_dim) - classification embedding
        """
        batch_size = features.shape[0]

        # Apply QFT
        x = self.qft(features)  # (batch, n_features, embed_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, n_features+1, embed_dim)

        # Apply transformer (no masking needed for tabular)
        x = self.transformer(x)

        # Extract CLS token output
        cls_output = x[:, 0, :]  # (batch, embed_dim)

        return self.ln(cls_output)


class ATAT(nn.Module):
    """
    Full ATAT model for astronomical classification.

    Combines light-curve and tabular transformers with fusion MLP.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        lc_embed_dim: int = 64,
        tab_embed_dim: int = 32,
        lc_layers: int = 3,
        tab_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.2,
        n_bands: int = 6
    ):
        super().__init__()

        self.n_classes = n_classes

        # Light-curve encoder
        self.lc_encoder = ATATLightCurveEncoder(
            embed_dim=lc_embed_dim,
            n_heads=n_heads,
            n_layers=lc_layers,
            dropout=dropout,
            n_bands=n_bands
        )

        # Tabular encoder
        self.tab_encoder = ATATTabularEncoder(
            n_features=n_features,
            embed_dim=tab_embed_dim,
            n_heads=n_heads,
            n_layers=tab_layers,
            dropout=dropout
        )

        # Fusion MLP
        fusion_dim = lc_embed_dim + tab_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, n_classes)
        )

        # For light-curve only mode
        self.lc_classifier = nn.Linear(lc_embed_dim, n_classes)

        # For tabular only mode
        self.tab_classifier = nn.Linear(tab_embed_dim, n_classes)

    def forward(
        self,
        flux: torch.Tensor,
        time: torch.Tensor,
        band_idx: torch.Tensor,
        lc_mask: torch.Tensor,
        features: torch.Tensor,
        mode: str = 'both'
    ) -> torch.Tensor:
        """
        Args:
            flux: (batch, seq_len, 2) - flux and flux_err
            time: (batch, seq_len) - observation times
            band_idx: (batch, seq_len) - band indices
            lc_mask: (batch, seq_len) - light-curve mask
            features: (batch, n_features) - tabular features
            mode: 'both', 'lc', or 'tab'

        Returns:
            (batch, n_classes) - class logits
        """
        if mode == 'lc':
            lc_emb = self.lc_encoder(flux, time, band_idx, lc_mask)
            return self.lc_classifier(lc_emb)

        elif mode == 'tab':
            tab_emb = self.tab_encoder(features)
            return self.tab_classifier(tab_emb)

        else:  # both
            lc_emb = self.lc_encoder(flux, time, band_idx, lc_mask)
            tab_emb = self.tab_encoder(features)

            # Concatenate and fuse
            combined = torch.cat([lc_emb, tab_emb], dim=-1)
            return self.fusion(combined)


def prepare_lightcurve_batch(
    lightcurves: pd.DataFrame,
    object_ids: List[str],
    max_seq_len: int = 400,
    band_map: Dict[str, int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare light curve data for ATAT.

    Args:
        lightcurves: DataFrame with object_id, Time (MJD), Flux, Flux_err, Filter
        object_ids: List of object IDs to process
        max_seq_len: Maximum sequence length (pad/truncate)
        band_map: Mapping from band names to indices

    Returns:
        Tuple of (flux, time, band_idx, mask) tensors
    """
    if band_map is None:
        band_map = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'y': 5}

    n_objects = len(object_ids)

    flux_arr = np.zeros((n_objects, max_seq_len, 2), dtype=np.float32)
    time_arr = np.zeros((n_objects, max_seq_len), dtype=np.float32)
    band_arr = np.zeros((n_objects, max_seq_len), dtype=np.int64)
    mask_arr = np.zeros((n_objects, max_seq_len), dtype=np.float32)

    # Group by object_id for efficiency
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    for i, obj_id in enumerate(object_ids):
        if obj_id not in grouped:
            continue

        obj_lc = grouped[obj_id].sort_values('Time (MJD)')

        # Get observations
        times = obj_lc['Time (MJD)'].values
        fluxes = obj_lc['Flux'].values
        flux_errs = obj_lc['Flux_err'].values
        bands = obj_lc['Filter'].values

        # Normalize time to start at 0
        times = times - times.min()

        # Limit to max_seq_len
        n_obs = min(len(times), max_seq_len)

        time_arr[i, :n_obs] = times[:n_obs]
        flux_arr[i, :n_obs, 0] = fluxes[:n_obs]
        flux_arr[i, :n_obs, 1] = flux_errs[:n_obs]
        mask_arr[i, :n_obs] = 1.0

        for j in range(n_obs):
            band = bands[j]
            if band in band_map:
                band_arr[i, j] = band_map[band]

    # Normalize flux
    flux_scale = np.nanmedian(np.abs(flux_arr[:, :, 0][mask_arr > 0]))
    if flux_scale > 0:
        flux_arr[:, :, 0] /= flux_scale
        flux_arr[:, :, 1] /= flux_scale

    # Handle NaN/inf
    flux_arr = np.nan_to_num(flux_arr, nan=0, posinf=0, neginf=0)
    time_arr = np.nan_to_num(time_arr, nan=0, posinf=0, neginf=0)

    return (
        torch.from_numpy(flux_arr),
        torch.from_numpy(time_arr),
        torch.from_numpy(band_arr),
        torch.from_numpy(mask_arr)
    )


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    seq_len = 100
    n_features = 50

    # Create dummy data
    flux = torch.randn(batch_size, seq_len, 2)
    time = torch.rand(batch_size, seq_len) * 500
    band_idx = torch.randint(0, 6, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)
    features = torch.randn(batch_size, n_features)

    # Create model
    model = ATAT(n_features=n_features, n_classes=2)

    # Forward pass
    logits = model(flux, time, band_idx, mask, features, mode='both')
    print(f"Output shape: {logits.shape}")  # Should be (batch_size, 2)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
