"""CTC loss wrapper with length masking."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
    blank: int = 0,
    zero_infinity: bool = True,
) -> Tensor:
    """Compute CTC loss.

    Args:
        log_probs:      (T, B, vocab_size) — log-softmax output from encoder
        targets:        (B, L_max) — padded label IDs
        input_lengths:  (B,) — actual frame counts (after subsampling)
        target_lengths: (B,) — actual label lengths
        blank:          Blank token ID
        zero_infinity:  Replace inf loss with 0 (handles mismatched lengths gracefully)

    Returns:
        Scalar CTC loss (mean over batch).
    """
    return nn.functional.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="mean",
        zero_infinity=zero_infinity,
    )
