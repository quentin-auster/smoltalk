"""Hydra entrypoint for InfoNCE contrastive pretraining (Milestone 3).

Pretrains the ConformerEncoder on unlabeled audio using InfoNCE loss,
then saves encoder weights for fine-tuning in the supervised CTC stage.

Usage:
    uv run python -m asr.train.ssl_run \\
        data=librispeech_unlabeled \\
        model=conformer_tiny \\
        trainer=cpu \\
        trainer.max_epochs=100

TODO (Milestone 3):
  - Implement LitSSL LightningModule wrapping ConformerEncoder + InfoNCELoss
  - Add unlabeled DataModule (audio only, no transcripts)
  - Add target network / quantizer (optional wav2vec2-style)
  - Export pretrained encoder weights after training
"""
from __future__ import annotations

import logging

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(config_path="../../../configs/asr", config_name="ssl_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("SSL pretraining config:\n%s", OmegaConf.to_yaml(cfg))
    L.seed_everything(cfg.get("seed", 42), workers=True)

    # TODO (Milestone 3): instantiate LitSSL + unlabeled DataModule
    raise NotImplementedError(
        "SSL pretraining entrypoint is a stub. "
        "Implement LitSSL in src/asr/models/lit_ssl.py first."
    )


if __name__ == "__main__":
    main()
