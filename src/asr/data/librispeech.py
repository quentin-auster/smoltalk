"""LibriSpeech dataset wrapper.

Supports:
  - dev-clean / dev-other (small, for local prototyping)
  - train-clean-100 / train-clean-360 / train-other-500 (for VM runs)
  - test-clean / test-other (for final evaluation)

Uses torchaudio.datasets.LIBRISPEECH under the hood.
Audio is loaded as float32, resampled to 16 kHz if needed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .vocab import CharVocab

log = logging.getLogger(__name__)

try:
    import torchaudio
    _TORCHAUDIO_AVAILABLE = True
except ImportError:
    _TORCHAUDIO_AVAILABLE = False

# Map short names → torchaudio split strings
SPLIT_MAP = {
    "dev-clean": "dev-clean",
    "dev-other": "dev-other",
    "train-100": "train-clean-100",
    "train-360": "train-clean-360",
    "train-500": "train-other-500",
    "test-clean": "test-clean",
    "test-other": "test-other",
}

TARGET_SAMPLE_RATE = 16_000


class LibriSpeechDataset(Dataset):
    """LibriSpeech dataset returning (waveform, labels, text) dicts.

    Args:
        root:    Directory where LibriSpeech data is stored or will be downloaded.
        split:   One of the keys in SPLIT_MAP (e.g. "dev-clean", "train-100").
        vocab:   CharVocab for encoding transcripts to token IDs.
        download: Whether to download the dataset if not present.
        transform: Optional callable applied to (waveform, sample_rate) → waveform.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        vocab: CharVocab,
        download: bool = False,
        transform: Callable[[Tensor, int], Tensor] | None = None,
    ) -> None:
        if not _TORCHAUDIO_AVAILABLE:
            raise ImportError("torchaudio is required: pip install torchaudio")

        split_str = SPLIT_MAP.get(split, split)
        self.vocab = vocab
        self.transform = transform

        log.info("Loading LibriSpeech split=%s from %s", split_str, root)
        self._ds = torchaudio.datasets.LIBRISPEECH(
            root=str(root),
            url=split_str,
            download=download,
        )

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict:
        """Returns dict with keys: waveform, labels, text, sample_rate."""
        waveform, sample_rate, transcript, *_ = self._ds[idx]
        # waveform: (1, T_samples)
        waveform = waveform.squeeze(0)  # → (T_samples,)

        if sample_rate != TARGET_SAMPLE_RATE:
            # TODO: use torchaudio.transforms.Resample for on-the-fly resampling
            raise ValueError(
                f"Expected {TARGET_SAMPLE_RATE} Hz, got {sample_rate} Hz. "
                "Add a Resample transform or resample data offline."
            )

        if self.transform is not None:
            waveform = self.transform(waveform, sample_rate)

        text = transcript.lower().strip()
        labels = torch.tensor(self.vocab.encode(text), dtype=torch.int64)

        return {"waveform": waveform, "labels": labels, "text": text, "sample_rate": sample_rate}
