#!/usr/bin/env bash
# Download LibriSpeech splits via torchaudio (Python).
# Adjust SPLITS to download what you need.
set -euo pipefail

DATA_ROOT="${1:-data/librispeech}"
SPLITS="${2:-dev-clean dev-other}"

echo "Downloading LibriSpeech splits: $SPLITS → $DATA_ROOT"

uv run python - <<EOF
import torchaudio
splits = "$SPLITS".split()
for split in splits:
    print(f"Downloading {split}...")
    torchaudio.datasets.LIBRISPEECH("$DATA_ROOT", url=split, download=True)
    print(f"  done.")
print("All splits downloaded.")
EOF
