"""Lightning module for supervised CTC-based ASR."""
from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn
import lightning as L
from torch import Tensor
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from ..audio.features import LogMelFeatures, SpecAugment
from .conv_subsampler import ConvSubsampler
from .conformer import ConformerEncoder
from ..decoding.greedy import greedy_ctc_decode_batch
from ..evaluation.metrics import compute_wer

log = logging.getLogger(__name__)


class LitASR(L.LightningModule):
    """Streaming Conformer + CTC, Lightning wrapper.

    Pipeline:
        waveform → LogMelFeatures → SpecAugment (train only)
                 → ConvSubsampler (4x downsample)
                 → ConformerEncoder
                 → Linear CTC head
                 → CTCLoss

    Args:
        vocab_size:     Total vocabulary size (from CharVocab.size).
        blank_id:       CTC blank token ID (0 by default).
        n_mels:         Mel filterbanks (default 80).
        d_model:        Conformer model dimension.
        n_layers:       Number of Conformer blocks.
        n_heads:        Attention heads.
        d_ff:           Feed-forward dimension.
        kernel_size:    Depthwise conv kernel size.
        dropout:        Dropout rate.
        left_context:   Left context frames for streaming (-1 = offline).
        lr:             Peak learning rate.
        weight_decay:   AdamW weight decay.
        warmup_steps:   Linear warmup steps.
        log_wer_every_n_epochs: How often to decode and log WER (expensive).
    """

    def __init__(
        self,
        vocab_size: int,
        blank_id: int = 0,
        n_mels: int = 80,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        d_ff: int = 1024,
        kernel_size: int = 31,
        dropout: float = 0.1,
        left_context: int = -1,
        lr: float = 5e-4,
        weight_decay: float = 1e-6,
        betas: Sequence[float] = (0.9, 0.98),
        warmup_steps: int = 10_000,
        log_wer_every_n_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.features = LogMelFeatures(n_mels=n_mels)
        self.spec_augment = SpecAugment()

        self.subsampler = ConvSubsampler(n_mels=n_mels, d_model=d_model, dropout=dropout)
        self.encoder = ConformerEncoder(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            kernel_size=kernel_size,
            dropout=dropout,
            left_context=left_context,
        )
        self.ctc_head = nn.Linear(d_model, vocab_size)
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)

        self.blank_id = blank_id
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = (betas[0], betas[1])
        self.warmup_steps = warmup_steps
        self.log_wer_every_n_epochs = log_wer_every_n_epochs

        self._val_wer_sum = 0.0
        self._val_wer_count = 0
        self._val_loss_sum = 0.0
        self._val_loss_count = 0

    def forward(self, waveforms: Tensor, waveform_lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Full forward pass: audio → log-probs.

        Args:
            waveforms:        (B, T_samples)
            waveform_lengths: (B,) actual sample counts
        Returns:
            log_probs:  (T_frames, B, vocab_size)  — log-softmax output for CTC
            out_lengths: (B,)  — frame lengths after subsampling
        """
        # Feature extraction (on CPU or GPU depending on device)
        feats = self.features(waveforms)        # (B, T_frames, n_mels)
        feats = self.spec_augment(feats)        # no-op at eval

        # Compute subsampled lengths from waveform lengths
        frame_lengths = waveform_lengths // self.features.hop_length
        out_lengths = self.subsampler.output_lengths(frame_lengths)

        x = self.subsampler(feats)             # (B, T//4, d_model)
        x, out_lengths = self.encoder(x, out_lengths)  # (B, T//4, d_model)
        logits = self.ctc_head(x)              # (B, T//4, vocab_size)

        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, V)
        return log_probs, out_lengths

    def _compute_ctc_loss(self, batch: dict) -> tuple[Tensor, Tensor]:
        log_probs, out_lengths = self(batch["waveforms"], batch["waveform_lengths"])
        loss = self.ctc_loss(log_probs, batch["labels"], out_lengths, batch["label_lengths"])
        return loss, log_probs

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        loss, _ = self._compute_ctc_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, log_probs = self._compute_ctc_loss(batch)
        self._val_loss_sum += loss.item()
        self._val_loss_count += 1

        # Decode and compute WER on first few batches of qualifying epochs
        if (self.current_epoch + 1) % self.log_wer_every_n_epochs == 0 and batch_idx < 10:
            hyps = greedy_ctc_decode_batch(log_probs, self.trainer.datamodule.vocab)
            refs = batch["texts"]
            for hyp, ref in zip(hyps, refs):
                wer = compute_wer([ref], [hyp])
                self._val_wer_sum += wer
                self._val_wer_count += 1

    def on_validation_epoch_start(self) -> None:
        self._val_wer_sum = 0.0
        self._val_wer_count = 0
        self._val_loss_sum = 0.0
        self._val_loss_count = 0

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking or self._val_loss_count == 0:
            return

        loss = self._val_loss_sum / self._val_loss_count
        self.log("val_loss", loss, prog_bar=True)
        msg = f"Epoch {self.current_epoch} | val_loss={loss:.4f}"

        if self._val_wer_count > 0:
            wer = self._val_wer_sum / self._val_wer_count
            self.log("val_wer", wer, prog_bar=True)
            msg += f" | val_wer={wer:.3f}"

        self.print(msg)
        log.info(msg)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(self.warmup_steps, 1)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
