"""Greedy CTC decoding."""
from __future__ import annotations

import torch
from torch import Tensor


def greedy_ctc_decode(
    log_probs: Tensor,
    vocab,
    blank_id: int = 0,
) -> str:
    """Greedy CTC decode for a single sequence.

    Args:
        log_probs: (T, vocab_size) — log-softmax output
        vocab:     CharVocab (or anything with .decode(ids))
        blank_id:  CTC blank token ID
    Returns:
        Decoded transcript string.
    """
    ids = log_probs.argmax(dim=-1).tolist()  # (T,)
    return vocab.decode(ids, collapse_repeated=True, remove_blank=True)


def greedy_ctc_decode_batch(
    log_probs: Tensor,
    vocab,
    lengths: Tensor | None = None,
    blank_id: int = 0,
) -> list[str]:
    """Greedy CTC decode for a batch.

    Args:
        log_probs: (T, B, vocab_size) — log-softmax, CTC format
        vocab:     CharVocab
        lengths:   (B,) actual frame lengths; if None, use full T
        blank_id:  CTC blank token ID
    Returns:
        List of B decoded strings.
    """
    T, B, _ = log_probs.shape
    results = []
    for b in range(B):
        length = int(lengths[b].item()) if lengths is not None else T
        seq = log_probs[:length, b, :]  # (T_b, V)
        results.append(greedy_ctc_decode(seq, vocab, blank_id))
    return results
