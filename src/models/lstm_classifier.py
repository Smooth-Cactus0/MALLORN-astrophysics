"""
LSTM Classifier for TDE Detection

Architecture:
1. Band embedding (learnable vectors for u,g,r,i,z,y)
2. Input projection layer
3. Bidirectional LSTM (captures forward and backward temporal patterns)
4. Self-attention (focuses on important timesteps)
5. Metadata fusion (redshift, EBV)
6. Classification head

Designed to be lightweight for smaller GPUs while still capturing
the temporal dynamics that distinguish TDEs from other transients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class Attention(nn.Module):
    """
    Self-attention mechanism for sequence classification.

    Learns which timesteps are most important for classification.
    Interpretable: attention weights show what the model focuses on.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

    def forward(self, lstm_output: torch.Tensor, mask: torch.Tensor) -> tuple:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) - 1 for valid, 0 for padding

        Returns:
            context: (batch, hidden_dim) - weighted sum of hidden states
            attention_weights: (batch, seq_len) - attention distribution
        """
        # Compute attention scores
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)

        # Mask padding positions with large negative value
        scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum of hidden states
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output                       # (batch, seq_len, hidden_dim)
        ).squeeze(1)  # (batch, hidden_dim)

        return context, attention_weights


class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM with attention for lightcurve classification.

    Architecture designed for:
    - Variable-length sequences (padding + masking)
    - Multi-band data (band embeddings)
    - Temporal patterns (bidirectional LSTM)
    - Focus on key moments (attention)
    - Auxiliary features (metadata fusion)
    """

    def __init__(
        self,
        input_dim: int = 4,           # [time, flux, flux_err, delta_t]
        n_bands: int = 6,              # u, g, r, i, z, y
        band_embed_dim: int = 8,       # Band embedding size
        hidden_dim: int = 64,          # LSTM hidden dimension
        n_layers: int = 2,             # Number of LSTM layers
        metadata_dim: int = 2,         # [redshift, EBV]
        dropout: float = 0.3,          # Dropout rate
        bidirectional: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        # Band embedding
        self.band_embedding = nn.Embedding(n_bands, band_embed_dim)

        # Input projection
        total_input_dim = input_dim + band_embed_dim
        self.input_projection = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention
        lstm_output_dim = hidden_dim * self.n_directions
        self.attention = Attention(lstm_output_dim)

        # Metadata projection
        self.metadata_projection = nn.Sequential(
            nn.Linear(metadata_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Classification head
        classifier_input_dim = lstm_output_dim + 16  # attention output + metadata
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # Binary classification
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # Xavier init for LSTM
                    nn.init.xavier_uniform_(param)
                elif len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        features: torch.Tensor,
        bands: torch.Tensor,
        mask: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: (batch, seq_len, 4) - [time, flux, flux_err, delta_t]
            bands: (batch, seq_len) - band indices
            mask: (batch, seq_len) - 1 for valid, 0 for padding
            metadata: (batch, 2) - [redshift, EBV]
            return_attention: whether to return attention weights

        Returns:
            Dictionary with:
            - logits: (batch, 1) - raw classification scores
            - probs: (batch,) - classification probabilities
            - attention_weights: (batch, seq_len) - if return_attention=True
        """
        batch_size, seq_len, _ = features.shape

        # Get band embeddings
        band_embeds = self.band_embedding(bands)  # (batch, seq_len, band_embed_dim)

        # Concatenate features with band embeddings
        x = torch.cat([features, band_embeds], dim=-1)  # (batch, seq_len, input_dim + band_embed_dim)

        # Project to hidden dimension
        x = self.input_projection(x)  # (batch, seq_len, hidden_dim)

        # Pack sequences for efficient LSTM processing
        lengths = mask.sum(dim=1).cpu()  # (batch,)

        # Handle edge case where all sequences have same length
        if lengths.min() == 0:
            lengths = lengths.clamp(min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # LSTM forward pass
        lstm_out, _ = self.lstm(packed)

        # Unpack
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=seq_len
        )  # (batch, seq_len, hidden_dim * n_directions)

        # Attention
        context, attention_weights = self.attention(lstm_out, mask)
        # context: (batch, hidden_dim * n_directions)

        # Metadata fusion
        if metadata is not None:
            meta_features = self.metadata_projection(metadata)  # (batch, 16)
            combined = torch.cat([context, meta_features], dim=-1)
        else:
            # If no metadata, pad with zeros
            meta_features = torch.zeros(batch_size, 16, device=features.device)
            combined = torch.cat([context, meta_features], dim=-1)

        # Classification
        logits = self.classifier(combined)  # (batch, 1)
        probs = torch.sigmoid(logits).squeeze(-1)  # (batch,)

        result = {
            'logits': logits,
            'probs': probs
        }

        if return_attention:
            result['attention_weights'] = attention_weights

        return result


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focuses training on hard examples by down-weighting easy ones.
    Particularly useful for rare event detection like TDEs.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weight for positive class (TDEs are minority)
        gamma: Focusing parameter (higher = more focus on hard examples)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, 1) - raw model outputs
            targets: (batch,) - binary labels

        Returns:
            Scalar loss value
        """
        # Clamp logits to prevent numerical instability
        logits_clamped = torch.clamp(logits.squeeze(-1), -20, 20)

        bce_loss = F.binary_cross_entropy_with_logits(
            logits_clamped, targets, reduction='none'
        )

        probs = torch.sigmoid(logits_clamped)
        # Clamp probabilities to prevent log(0)
        probs = torch.clamp(probs, 1e-7, 1 - 1e-7)

        p_t = probs * targets + (1 - probs) * (1 - targets)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce_loss

        # Handle NaN
        if torch.isnan(loss).any():
            return bce_loss.mean()

        return loss.mean()


class WeightedBCELoss(nn.Module):
    """
    Simple weighted BCE loss as a more stable alternative to Focal Loss.
    """

    def __init__(self, pos_weight: float = 20.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_clamped = torch.clamp(logits.squeeze(-1), -20, 20)
        pos_weight = torch.tensor([self.pos_weight], device=logits.device)
        return F.binary_cross_entropy_with_logits(
            logits_clamped, targets,
            pos_weight=pos_weight
        )


if __name__ == "__main__":
    # Test the model
    print("Testing LSTM Classifier...")

    # Create dummy data
    batch_size = 8
    seq_len = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    features = torch.randn(batch_size, seq_len, 4).to(device)
    bands = torch.randint(0, 6, (batch_size, seq_len)).to(device)
    mask = torch.ones(batch_size, seq_len).to(device)
    mask[:, 80:] = 0  # Simulate padding
    metadata = torch.randn(batch_size, 2).to(device)
    labels = torch.randint(0, 2, (batch_size,)).float().to(device)

    # Create model
    model = LSTMClassifier(
        hidden_dim=64,
        n_layers=2,
        dropout=0.3
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    output = model(features, bands, mask, metadata, return_attention=True)

    print(f"\nOutput shapes:")
    print(f"  Logits: {output['logits'].shape}")
    print(f"  Probs: {output['probs'].shape}")
    print(f"  Attention: {output['attention_weights'].shape}")

    # Test loss
    focal_loss = FocalLoss()
    loss = focal_loss(output['logits'], labels)
    print(f"\nFocal loss: {loss.item():.4f}")

    # Test backward pass
    loss.backward()
    print("Backward pass successful!")
