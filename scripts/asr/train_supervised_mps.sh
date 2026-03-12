#!/usr/bin/env bash
# Supervised CTC training on MPS (Apple Silicon) with dev-clean.
set -euo pipefail

uv run python -m asr.train.run \
    data=librispeech_dev \
    model=conformer_tiny \
    trainer=mps \
    logger=tensorboard \
    "trainer.max_epochs=200"
