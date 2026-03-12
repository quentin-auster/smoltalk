"""Hydra entrypoint for supervised CTC ASR training.

Usage:
    uv run python -m asr.train.run \\
        data=librispeech_dev \\
        model=conformer_tiny \\
        trainer=cpu \\
        trainer.max_epochs=10

Supports checkpoint resuming:
    uv run python -m asr.train.run \\
        run.ckpt_path=outputs/.../last.ckpt \\
        trainer.max_epochs=50
"""
from __future__ import annotations

import logging
import os

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
from decouple import config as decouple_config

log = logging.getLogger(__name__)


@hydra.main(config_path="../../../configs/asr", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("ASR training config:\n%s", OmegaConf.to_yaml(cfg))

    # Load W&B API key from .env when using the wandb logger.
    if (cfg.get("logger") or {}).get("_target_", "").endswith("WandbLogger"):
        wandb_key = str(decouple_config("WANDB_API_KEY", default=""))
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key

    L.seed_everything(cfg.get("seed", 42), workers=True)

    # ── DataModule ────────────────────────────────────────────────────────────
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    # ── Model ─────────────────────────────────────────────────────────────────
    # Pass vocab_size from the DataModule so the config stays DRY.
    datamodule.setup("fit")
    vocab_size = datamodule.vocab_size

    from asr.models.lit_asr import LitASR
    model = LitASR(vocab_size=vocab_size, **OmegaConf.to_container(cfg.model, resolve=True))

    # ── Callbacks & Logger ────────────────────────────────────────────────────
    callbacks = [hydra.utils.instantiate(c) for c in cfg.get("callbacks", {}).values()]
    logger = hydra.utils.instantiate(cfg.logger) if "logger" in cfg else True

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    ckpt_path: str | None = cfg.get("run", {}).get("ckpt_path", None)
    if ckpt_path:
        log.info("Resuming from checkpoint: %s", ckpt_path)

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    log.info("Training complete.")


if __name__ == "__main__":
    main()
