"""WER/CER evaluation for ASR."""
from __future__ import annotations

import logging
from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

try:
    import jiwer
    _JIWER_AVAILABLE = True
except ImportError:
    _JIWER_AVAILABLE = False


def _check_jiwer() -> None:
    if not _JIWER_AVAILABLE:
        raise ImportError("jiwer is required for WER/CER: pip install jiwer")


def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    """Word Error Rate: (S + D + I) / N.

    Args:
        references:  List of reference transcripts.
        hypotheses:  List of hypothesis transcripts (same length).
    Returns:
        WER as a float in [0, ∞) (can exceed 1.0 with many insertions).
    """
    _check_jiwer()
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ExpandCommonEnglishContractions(),
    ])
    return jiwer.wer(
        references,
        hypotheses,
        reference_transform=transform,
        hypothesis_transform=transform,
    )


def compute_cer(references: list[str], hypotheses: list[str]) -> float:
    """Character Error Rate.

    Args:
        references:  List of reference transcripts.
        hypotheses:  List of hypothesis transcripts.
    Returns:
        CER as a float in [0, ∞).
    """
    _check_jiwer()
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
    ])
    return jiwer.cer(
        references,
        hypotheses,
        reference_transform=transform,
        hypothesis_transform=transform,
    )


@torch.no_grad()
def evaluate_dataset(
    model,
    dataloader: DataLoader,
    vocab,
    device: str | torch.device = "cpu",
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate a LitASR model on a dataloader, returning WER and CER.

    Args:
        model:       LitASR (or any model with a __call__ returning (log_probs, lengths)).
        dataloader:  ASR DataLoader yielding batches from asr_collate.
        vocab:       CharVocab for decoding.
        device:      Device to run on.
        max_batches: Limit evaluation to first N batches (for quick checks).
    Returns:
        dict with "wer", "cer", "n_utterances".
    """
    from ..decoding.greedy import greedy_ctc_decode_batch

    model.eval()
    model.to(device)

    all_refs: list[str] = []
    all_hyps: list[str] = []

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        waveforms = batch["waveforms"].to(device)
        lengths = batch["waveform_lengths"].to(device)
        log_probs, out_lengths = model(waveforms, lengths)

        hyps = greedy_ctc_decode_batch(log_probs, vocab, out_lengths)
        refs = batch["texts"]

        all_refs.extend(refs)
        all_hyps.extend(hyps)

    if not all_refs:
        return {"wer": float("nan"), "cer": float("nan"), "n_utterances": 0}

    wer = compute_wer(all_refs, all_hyps)
    cer = compute_cer(all_refs, all_hyps)

    log.info("WER=%.3f  CER=%.3f  over %d utterances", wer, cer, len(all_refs))
    return {"wer": wer, "cer": cer, "n_utterances": len(all_refs)}
