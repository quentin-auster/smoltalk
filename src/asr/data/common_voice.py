"""Common Voice dataset wrapper (for VM-scale runs).

Usage:
  1. Download Common Voice TSV + clips from https://commonvoice.mozilla.org/
  2. Point `root` at the extracted directory containing `validated.tsv` and `clips/`.

For local prototyping, prefer LibriSpeech dev-clean (auto-download).
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

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

TARGET_SAMPLE_RATE = 16_000


class CommonVoiceDataset(Dataset):
    """Common Voice English dataset from a local TSV extract.

    Expects directory structure:
        root/
          validated.tsv   (or train.tsv / dev.tsv / test.tsv)
          clips/
            <file>.mp3

    Args:
        root:   Path to the Common Voice language directory.
        split:  TSV file to load, e.g. "validated", "train", "dev", "test".
        vocab:  CharVocab for encoding transcripts.
        max_duration_s: Skip utterances longer than this (seconds). None = no limit.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "validated",
        vocab: CharVocab | None = None,
        max_duration_s: float | None = 20.0,
    ) -> None:
        if not _TORCHAUDIO_AVAILABLE:
            raise ImportError("torchaudio is required: pip install torchaudio")

        self.root = Path(root)
        self.vocab = vocab
        self.max_duration_s = max_duration_s
        self.clips_dir = self.root / "clips"

        tsv_path = self.root / f"{split}.tsv"
        if not tsv_path.exists():
            raise FileNotFoundError(f"TSV not found: {tsv_path}")

        self._examples: list[dict] = []
        with open(tsv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self._examples.append({
                    "path": row["path"],
                    "sentence": row["sentence"],
                })

        log.info("Loaded %d examples from Common Voice %s", len(self._examples), split)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self._examples[idx]
        audio_path = self.clips_dir / ex["path"]

        waveform, sample_rate = torchaudio.load(str(audio_path))
        waveform = waveform.mean(dim=0)  # stereo → mono

        if sample_rate != TARGET_SAMPLE_RATE:
            resample = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
            waveform = resample(waveform)

        text = ex["sentence"].lower().strip()
        labels: Tensor | None = None
        if self.vocab is not None:
            labels = torch.tensor(self.vocab.encode(text), dtype=torch.int64)

        return {"waveform": waveform, "labels": labels, "text": text, "sample_rate": TARGET_SAMPLE_RATE}
