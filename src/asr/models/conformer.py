"""Conformer encoder for streaming ASR.

Architecture (per block):
    x → FeedForward (½ step) → MHSA → Convolution → FeedForward (½ step) → LayerNorm

Reference: Gulati et al. 2020, "Conformer: Convolution-augmented Transformer for ASR"

Streaming support:
    - Causal + limited left context for self-attention
    - Causal depthwise conv (no right context leak)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FeedForward(nn.Module):
    """Conformer feed-forward module: LN → Linear → SiLU → Dropout → Linear → Dropout."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + 0.5 * self.ff(self.norm(x))


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional causal + left-context mask.

    Args:
        d_model:      Model dimension.
        n_heads:      Number of attention heads.
        dropout:      Attention dropout.
        left_context: Number of past frames to attend to per head (streaming).
                      -1 = full left context (non-streaming / offline).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        left_context: int = -1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.left_context = left_context

    def _make_causal_mask(self, T: int, device: torch.device) -> Tensor:
        """Upper-triangular mask (True = ignore) for causal attention."""
        if self.left_context == -1:
            # Standard causal mask
            mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        else:
            # Limited left context: allow attending to [t - left_context, t]
            mask = torch.ones(T, T, device=device, dtype=torch.bool)
            for i in range(T):
                lo = max(0, i - self.left_context)
                mask[i, lo : i + 1] = False
        return mask

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: (B, T, d_model)
            key_padding_mask: (B, T) — True where positions are padded
        Returns:
            (B, T, d_model)
        """
        normed = self.norm(x)
        T = x.shape[1]
        attn_mask = self._make_causal_mask(T, x.device)
        out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask)
        return x + out


class ConvModule(nn.Module):
    """Conformer convolution module.

    LN → Pointwise Conv (expand) → GLU → Depthwise Conv (causal) → BN → SiLU → Pointwise Conv → Dropout

    Args:
        d_model:     Model dimension.
        kernel_size: Depthwise conv kernel size (odd number).
        dropout:     Output dropout.
        causal:      If True, use causal (left-only) padding for streaming.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
        causal: bool = True,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_expand = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.causal = causal
        self.kernel_size = kernel_size
        # For causal: pad (kernel_size - 1) on the left only
        self.depthwise = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=0,  # manual padding below
            groups=d_model,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pointwise_contract = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)              # (B, d_model, T)
        x = self.pointwise_expand(x)       # (B, 2*d_model, T)
        x = F.glu(x, dim=1)               # (B, d_model, T)

        # Causal padding: (kernel_size - 1) zeros on the left
        pad = self.kernel_size - 1 if self.causal else (self.kernel_size - 1) // 2
        x = F.pad(x, (pad, 0))
        x = self.depthwise(x)              # (B, d_model, T)

        x = self.bn(x)
        x = F.silu(x)
        x = self.pointwise_contract(x)     # (B, d_model, T)
        x = x.transpose(1, 2)             # (B, T, d_model)
        return residual + self.dropout(x)


class ConformerBlock(nn.Module):
    """Single Conformer block: FF → MHSA → Conv → FF → LN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
        left_context: int = -1,
        causal_conv: bool = True,
    ) -> None:
        super().__init__()
        self.ff1 = FeedForward(d_model, d_ff, dropout)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, left_context)
        self.conv = ConvModule(d_model, kernel_size, dropout, causal=causal_conv)
        self.ff2 = FeedForward(d_model, d_ff, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        x = self.ff1(x)
        x = self.attn(x, key_padding_mask)
        x = self.conv(x)
        x = self.ff2(x)
        return self.norm(x)


class ConformerEncoder(nn.Module):
    """Stack of ConformerBlocks with positional encoding.

    Args:
        n_layers:     Number of Conformer blocks.
        d_model:      Model dimension.
        n_heads:      Attention heads per block.
        d_ff:         Feed-forward inner dimension (typically 4 * d_model).
        kernel_size:  Depthwise conv kernel size.
        dropout:      Dropout rate throughout.
        left_context: Frames of left context for streaming (-1 = full).
        max_len:      Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        n_layers: int = 6,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        kernel_size: int = 31,
        dropout: float = 0.1,
        left_context: int = -1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Fixed sinusoidal positional encoding
        self.register_buffer("pos_enc", _make_sinusoidal_pe(max_len, d_model))

        self.input_proj = nn.Linear(d_model, d_model)  # after subsampler
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                kernel_size=kernel_size,
                dropout=dropout,
                left_context=left_context,
                causal_conv=(left_context != -1),
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """
        Args:
            x:       (B, T, d_model) — output from ConvSubsampler
            lengths: (B,) int — actual sequence lengths (before padding)
        Returns:
            out:     (B, T, d_model)
            lengths: (B,) — unchanged input lengths (for CTC loss)
        """
        T = x.shape[1]
        x = self.dropout(x + self.pos_enc[:, :T, :])

        # Build key_padding_mask: True where padded
        key_padding_mask: Tensor | None = None
        if lengths is not None:
            B = x.shape[0]
            key_padding_mask = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]

        for block in self.blocks:
            x = block(x, key_padding_mask)

        return x, lengths


def _make_sinusoidal_pe(max_len: int, d_model: int) -> Tensor:
    """Precompute sinusoidal positional encoding. Shape: (1, max_len, d_model)."""
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, max_len, d_model)
