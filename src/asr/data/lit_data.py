"""Lightning DataModule for ASR (LibriSpeech / Common Voice)."""
from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Callable

import lightning as L
from torch.utils.data import DataLoader, Dataset

from .vocab import CharVocab, build_char_vocab
from .collate import asr_collate
from .librispeech import LibriSpeechDataset

log = logging.getLogger(__name__)


class ASRDataModule(L.LightningDataModule):
    """Lightning DataModule for CTC-based ASR.

    Supports LibriSpeech out of the box. Common Voice requires a local extract.

    Milestone 1 defaults:
        - train_split = "dev-clean"   (5.4 h, auto-download)
        - val_split   = "dev-other"   (5.1 h, auto-download)

    VM-scale upgrades:
        - train_split = "train-100"   (100 h)
        - val_split   = "dev-clean"

    Args:
        data_root:    Directory to download / find LibriSpeech data.
        train_split:  LibriSpeech split name for training.
        val_split:    LibriSpeech split name for validation.
        test_split:   LibriSpeech split name for testing (optional).
        batch_size:   Batch size (number of utterances).
        num_workers:  DataLoader workers.
        download:     Download LibriSpeech if not present.
        transform:    Optional waveform transform (e.g. speed perturbation).
    """

    def __init__(
        self,
        data_root: str = "data/librispeech",
        train_split: str = "dev-clean",
        val_split: str = "dev-other",
        test_split: str | None = "test-clean",
        batch_size: int = 16,
        num_workers: int = 4,
        download: bool = True,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["transform"])
        self.data_root = Path(data_root)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.transform = transform

        self.vocab: CharVocab = build_char_vocab()
        self._train_ds: Dataset | None = None
        self._val_ds: Dataset | None = None
        self._test_ds: Dataset | None = None

    @property
    def vocab_size(self) -> int:
        return self.vocab.size

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self._train_ds = LibriSpeechDataset(
                root=self.data_root,
                split=self.train_split,
                vocab=self.vocab,
                download=self.download,
                transform=self.transform,
            )
            self._val_ds = LibriSpeechDataset(
                root=self.data_root,
                split=self.val_split,
                vocab=self.vocab,
                download=self.download,
            )
            log.info("Train: %d utterances | Val: %d utterances",
                     len(self._train_ds), len(self._val_ds))

        if stage in ("test", None) and self.test_split is not None:
            self._test_ds = LibriSpeechDataset(
                root=self.data_root,
                split=self.test_split,
                vocab=self.vocab,
                download=self.download,
            )

    def _make_loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=partial(asr_collate, pad_label=self.vocab.pad_id),
            pin_memory=False,
            drop_last=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self._test_ds is None:
            raise RuntimeError("test_split is not configured")
        return self._make_loader(self._test_ds, shuffle=False)
