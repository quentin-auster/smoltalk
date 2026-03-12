#!/usr/bin/env bash
# Supervised CTC training on a single CUDA GPU with LibriSpeech 100h.
set -euo pipefail

uv run python -m asr.train.run \
    data=librispeech_100h \
    model=conformer_tiny \
    trainer=gpu \
    logger=wandb \
    "trainer.max_epochs=1000"
