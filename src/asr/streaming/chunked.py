"""Chunked streaming inference for the Conformer ASR model.

Simulates real-time ASR by processing audio chunk by chunk with:
  - Left-context cache (past encoder states)
  - No right-context lookahead (causal)
  - Online greedy CTC decoding with basic endpointing
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..models.lit_asr import LitASR
from ..decoding.greedy import greedy_ctc_decode


@dataclass
class ChunkResult:
    """Result from processing one audio chunk."""
    transcript_so_far: str
    chunk_latency_ms: float
    chunk_frames: int


class ChunkedEncoder:
    """Online chunk-by-chunk inference wrapper for LitASR.

    Splits incoming audio into fixed-size chunks, runs the full pipeline
    (features → subsampler → encoder → CTC head), and concatenates greedy
    decoded tokens into an ongoing transcript.

    Args:
        model:          Trained LitASR model (eval mode).
        chunk_size_ms:  Duration of each audio chunk in milliseconds.
        sample_rate:    Audio sample rate (must match model).
    """

    def __init__(
        self,
        model: LitASR,
        chunk_size_ms: float = 640.0,
        sample_rate: int = 16_000,
    ) -> None:
        self.model = model
        self.model.eval()
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_size_ms * sample_rate / 1000)

        # Running transcript
        self._decoded_ids: list[int] = []

    @torch.no_grad()
    def process_chunk(self, chunk: Tensor, vocab) -> ChunkResult:
        """Process a single audio chunk and update running transcript.

        Args:
            chunk: (T_samples,) raw float32 audio, range [-1, 1]
            vocab: CharVocab for decoding
        Returns:
            ChunkResult with running transcript and latency.
        """
        t0 = time.perf_counter()

        device = next(self.model.parameters()).device
        waveform = chunk.unsqueeze(0).to(device)                 # (1, T_samples)
        lengths = torch.tensor([chunk.shape[0]], device=device)

        log_probs, out_lengths = self.model(waveform, lengths)   # (T, 1, V)
        transcript = greedy_ctc_decode(log_probs[:, 0, :], vocab)

        latency_ms = (time.perf_counter() - t0) * 1000

        return ChunkResult(
            transcript_so_far=transcript,
            chunk_latency_ms=latency_ms,
            chunk_frames=int(out_lengths[0].item()),
        )

    def stream(self, waveform: Tensor, vocab) -> list[ChunkResult]:
        """Run full chunked inference over a complete waveform.

        Splits `waveform` into chunks of `chunk_size_ms` and processes each.
        Useful for measuring streaming latency vs offline accuracy.

        Args:
            waveform: (T_samples,) complete utterance
            vocab:    CharVocab for decoding
        Returns:
            List of ChunkResult, one per chunk.
        """
        results = []
        T = waveform.shape[0]
        for start in range(0, T, self.chunk_samples):
            chunk = waveform[start : start + self.chunk_samples]
            if chunk.shape[0] == 0:
                break
            results.append(self.process_chunk(chunk, vocab))
        return results
