"""Conv2d-based subsampler: reduces time by 4x before the Conformer."""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ConvSubsampler(nn.Module):
    """Two stride-2 Conv2d layers → 4x time + frequency subsampling.

    Operates on log-mel spectrograms viewed as 2D images (time × freq).

    Input:  (B, T, n_mels)
    Output: (B, T//4, d_model)

    After two (3×3, stride=2, pad=1) convolutions:
      time: T  → T//4
      freq: n_mels → n_mels//4

    The output is projected from (d_model * n_mels//4) → d_model.

    Args:
        n_mels:   Number of mel bins (freq dimension), typically 80.
        d_model:  Conformer model dimension (output channels).
        dropout:  Dropout after projection.
    """

    def __init__(self, n_mels: int = 80, d_model: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        proj_in = d_model * (n_mels // 4)
        self.proj = nn.Linear(proj_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, n_mels)
        Returns:
            out: (B, T//4, d_model)
        """
        B, T, F = x.shape
        x = x.unsqueeze(1)              # (B, 1, T, n_mels)
        x = self.conv(x)                # (B, d_model, T//4, n_mels//4)
        B, C, T2, F2 = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()   # (B, T//4, d_model, n_mels//4)
        x = x.view(B, T2, C * F2)      # (B, T//4, d_model * n_mels//4)
        x = self.proj(x)               # (B, T//4, d_model)
        return self.dropout(x)

    def output_lengths(self, input_lengths: Tensor) -> Tensor:
        """Compute output time lengths after subsampling (for CTC mask).

        Each stride-2 conv: L_out = (L_in + 2*pad - kernel) // stride + 1
                                  = (L_in + 2 - 3) // 2 + 1
                                  = (L_in - 1) // 2 + 1
        """
        lengths = (input_lengths - 1) // 2 + 1  # after first conv
        lengths = (lengths - 1) // 2 + 1         # after second conv
        return lengths
