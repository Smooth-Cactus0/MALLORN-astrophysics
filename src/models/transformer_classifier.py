"""
Transformer Classifier for Astronomical Lightcurves

Architecture:
1. Input embedding: Combines flux features + band embeddings + positional encoding
2. Transformer encoder: Multi-head self-attention layers
3. Sequence aggregation: Learnable [CLS] token or attention pooling
4. Classification head: MLP with metadata fusion

Key design choices:
- Positional encoding based on actual time values (not indices)
- Band embeddings to distinguish different photometric bands
- Attention masking for variable-length sequences
- Weighted loss for class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional


class SinusoidalTimeEncoding(nn.Module):
    """
    Positional encoding based on actual time values (MJD).

    Unlike standard positional encoding that uses sequence indices,
    this uses the actual time values from the lightcurve, which
    better captures the irregular temporal sampling in astronomical data.
    """

    def __init__(self, d_model: int, max_time: float = 1000.0):
        super().__init__()
        self.d_model = d_model
        self.max_time = max_time

        # Create frequency bands
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """
        Args:
            times: (batch, seq_len) tensor of time values

        Returns:
            (batch, seq_len, d_model) positional encodings
        """
        # Normalize times to [0, 1] range
        times_norm = times / self.max_time

        # Expand dimensions: (batch, seq_len, 1)
        times_norm = times_norm.unsqueeze(-1)

        # Compute sin/cos embeddings
        pe = torch.zeros(*times.shape, self.d_model, device=times.device)
        pe[..., 0::2] = torch.sin(times_norm * self.div_term)
        pe[..., 1::2] = torch.cos(times_norm * self.div_term)

        return pe


class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for astronomical lightcurves.

    Architecture:
    1. Feature embedding layer
    2. Band embedding (learnable)
    3. Sinusoidal time encoding
    4. [CLS] token for classification
    5. Transformer encoder layers
    6. Metadata fusion
    7. Classification head
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.2,
        n_bands: int = 6,
        n_features: int = 4,  # time, flux, flux_err, delta_t
        n_metadata: int = 2,   # redshift, EBV
        max_time: float = 500.0
    ):
        super().__init__()

        self.d_model = d_model

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Band embedding
        self.band_embedding = nn.Embedding(n_bands, d_model)

        # Time encoding
        self.time_encoding = SinusoidalTimeEncoding(d_model, max_time)

        # Learnable [CLS] token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # Metadata fusion (after transformer)
        self.metadata_proj = nn.Sequential(
            nn.Linear(n_metadata, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        features: torch.Tensor,
        bands: torch.Tensor,
        mask: torch.Tensor,
        metadata: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: (batch, seq_len, 4) - time, flux, flux_err, delta_t
            bands: (batch, seq_len) - band indices
            mask: (batch, seq_len) - 1 for valid, 0 for padding
            metadata: (batch, 2) - redshift, EBV

        Returns:
            Dict with 'logits' and 'probs' keys
        """
        batch_size, seq_len, _ = features.shape

        # Handle NaN values
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if metadata is not None:
            metadata = torch.nan_to_num(metadata, nan=0.0, posinf=0.0, neginf=0.0)

        # Extract time for positional encoding
        times = features[:, :, 0]  # (batch, seq_len)

        # 1. Feature projection
        x = self.feature_proj(features)  # (batch, seq_len, d_model)

        # 2. Add band embeddings
        band_emb = self.band_embedding(bands)  # (batch, seq_len, d_model)
        x = x + band_emb

        # 3. Add time encoding
        time_enc = self.time_encoding(times)  # (batch, seq_len, d_model)
        x = x + time_enc

        # 4. Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, d_model)

        # Update mask for [CLS] token (always valid)
        cls_mask = torch.ones(batch_size, 1, device=mask.device)
        extended_mask = torch.cat([cls_mask, mask], dim=1)  # (batch, seq_len+1)

        # Create attention mask (True = ignore)
        attn_mask = (extended_mask == 0)  # (batch, seq_len+1)

        # 5. Transformer encoder
        x = self.transformer(x, src_key_padding_mask=attn_mask)  # (batch, seq_len+1, d_model)

        # 6. Extract [CLS] token representation
        cls_output = x[:, 0, :]  # (batch, d_model)

        # 7. Fuse metadata
        if metadata is not None:
            meta_emb = self.metadata_proj(metadata)  # (batch, d_model//2)
            cls_output = torch.cat([cls_output, meta_emb], dim=-1)  # (batch, d_model + d_model//2)
        else:
            # If no metadata, pad with zeros
            zeros = torch.zeros(batch_size, self.d_model // 2, device=cls_output.device)
            cls_output = torch.cat([cls_output, zeros], dim=-1)

        # 8. Classification
        logits = self.classifier(cls_output)  # (batch, 1)

        # Clamp logits for numerical stability
        logits_clamped = torch.clamp(logits, -20, 20)
        probs = torch.sigmoid(logits_clamped).squeeze(-1)

        return {
            'logits': logits,
            'probs': probs
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focuses learning on hard examples by down-weighting easy ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.squeeze(-1)
        logits = torch.clamp(logits, -20, 20)

        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Compute focal weights
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Compute alpha weights
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Apply positive class weight
        class_weight = self.pos_weight * targets + (1 - targets)

        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Combine
        loss = alpha_t * focal_weight * class_weight * bce_loss

        return loss.mean()


if __name__ == "__main__":
    # Test the model
    batch_size = 8
    seq_len = 100

    model = TransformerClassifier()

    # Create dummy inputs
    features = torch.randn(batch_size, seq_len, 4)
    bands = torch.randint(0, 6, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)
    mask[:, 80:] = 0  # Simulate padding
    metadata = torch.randn(batch_size, 2)

    # Forward pass
    output = model(features, bands, mask, metadata)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Probs shape: {output['probs'].shape}")
    print(f"Sample probs: {output['probs'][:5]}")
