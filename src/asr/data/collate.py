"""Collate function for ASR batches: pad audio + label sequences."""
from __future__ import annotations

import torch
from torch import Tensor


def asr_collate(
    items: list[dict],
    pad_audio: float = 0.0,
    pad_label: int = 1,  # PAD_ID
) -> dict[str, Tensor]:
    """Collate a list of ASR examples into a padded batch.

    Each item must have:
        "waveform"  : Tensor (T_samples,)   — raw audio
        "labels"    : Tensor (L,)            — token IDs
        "text"      : str                    — reference transcript

    Returns a dict with:
        "waveforms"         : (B, T_max)    float32
        "waveform_lengths"  : (B,)          int32 — actual sample counts
        "labels"            : (B, L_max)    int64 — padded label IDs
        "label_lengths"     : (B,)          int32 — actual label lengths
        "texts"             : list[str]     — reference transcripts
    """
    waveforms = [item["waveform"] for item in items]
    labels = [item["labels"] for item in items]
    texts = [item["text"] for item in items]

    waveform_lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.int32)
    label_lengths = torch.tensor([l.shape[0] for l in labels], dtype=torch.int32)

    # Pad waveforms to max length in batch
    T_max = int(waveform_lengths.max().item())
    waveforms_padded = torch.full((len(items), T_max), pad_audio)
    for i, w in enumerate(waveforms):
        waveforms_padded[i, : w.shape[0]] = w

    # Pad labels to max label length
    L_max = int(label_lengths.max().item())
    labels_padded = torch.full((len(items), L_max), pad_label, dtype=torch.int64)
    for i, l in enumerate(labels):
        labels_padded[i, : l.shape[0]] = l

    return {
        "waveforms": waveforms_padded,
        "waveform_lengths": waveform_lengths,
        "labels": labels_padded,
        "label_lengths": label_lengths,
        "texts": texts,
    }
