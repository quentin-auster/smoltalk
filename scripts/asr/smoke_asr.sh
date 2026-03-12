#!/usr/bin/env bash
# Quick smoke test: verify the ASR pipeline runs end-to-end on CPU.
# Uses LibriSpeech dev-clean (auto-downloads ~700MB on first run).
set -euo pipefail

echo "=== ASR smoke test (CPU, 2 epochs) ==="
uv run python -m asr.train.run \
    data=librispeech_dev \
    model=conformer_tiny \
    trainer=cpu \
    logger=none \
    "trainer.max_epochs=2" \
    "trainer.limit_train_batches=2" \
    "trainer.limit_val_batches=2"

echo "=== Smoke test passed ==="
