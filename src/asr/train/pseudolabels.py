"""Pseudo-label generation and filtering for semi-supervised training (Milestone 4).

Pipeline:
  1. Load a trained LitASR checkpoint.
  2. Decode unlabeled audio with greedy CTC.
  3. Filter predictions by confidence (average log-prob per frame).
  4. Write filtered (audio_path, transcript) pairs to a TSV for use in
     the semi-supervised DataModule.

Usage:
    uv run python -m asr.train.pseudolabels \\
        ckpt=outputs/.../last.ckpt \\
        audio_dir=data/unlabeled \\
        out=data/pseudolabels.tsv \\
        confidence_threshold=0.8

TODO (Milestone 4):
  - Implement the unlabeled audio scanner
  - Implement confidence scoring (avg log-prob / frame length)
  - Implement the TSV writer
  - Integrate with ASRDataModule for semi-supervised batching
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def generate_pseudolabels(
    ckpt_path: str,
    audio_dir: str,
    out_tsv: str,
    confidence_threshold: float = 0.8,
    device: str = "cpu",
    batch_size: int = 8,
) -> None:
    """Generate and filter pseudo-labels from a trained ASR model.

    Args:
        ckpt_path:            Path to LitASR checkpoint (.ckpt).
        audio_dir:            Directory of unlabeled .wav / .flac files.
        out_tsv:              Output TSV path (audio_path \\t transcript).
        confidence_threshold: Minimum average log-prob per frame to keep.
        device:               Device string.
        batch_size:           Batch size for inference.
    """
    raise NotImplementedError("Milestone 4: implement pseudo-label generation.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate pseudo-labels from a trained ASR model.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--confidence_threshold", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    generate_pseudolabels(
        ckpt_path=args.ckpt,
        audio_dir=args.audio_dir,
        out_tsv=args.out,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        batch_size=args.batch_size,
    )
