"""Log-mel spectrogram extraction and SpecAugment."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

try:
    import torchaudio
    import torchaudio.transforms as T
    _TORCHAUDIO_AVAILABLE = True
except ImportError:
    _TORCHAUDIO_AVAILABLE = False


class LogMelFeatures(nn.Module):
    """Compute log-mel spectrogram features from raw waveform.

    Standard settings for ASR (matching ESPnet / k2 / whisper defaults):
      - 16 kHz sample rate
      - 25 ms window, 10 ms hop  →  100 frames/second
      - 80 mel bins
      - log(max(x, 1e-10)) normalization

    Input:  (B, T_samples)  float32, values in [-1, 1]
    Output: (B, T_frames, n_mels)
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_mels: int = 80,
        n_fft: int = 512,
        win_length_ms: float = 25.0,
        hop_length_ms: float = 10.0,
        f_min: float = 80.0,
        f_max: float = 7_600.0,
    ) -> None:
        super().__init__()
        if not _TORCHAUDIO_AVAILABLE:
            raise ImportError("torchaudio is required: pip install torchaudio")

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        win_length = int(sample_rate * win_length_ms / 1000)
        hop_length = int(sample_rate * hop_length_ms / 1000)

        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
        )
        self.hop_length = hop_length

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Args:
            waveform: (B, T_samples) or (T_samples,)
        Returns:
            features: (B, T_frames, n_mels)
        """
        squeeze = waveform.dim() == 1
        if squeeze:
            waveform = waveform.unsqueeze(0)

        # mel: (B, n_mels, T_frames)
        mel = self.mel(waveform)
        log_mel = torch.log(mel.clamp(min=1e-10))
        log_mel = log_mel.transpose(1, 2)  # → (B, T_frames, n_mels)

        if squeeze:
            log_mel = log_mel.squeeze(0)
        return log_mel


class SpecAugment(nn.Module):
    """SpecAugment: frequency and time masking for ASR data augmentation.

    Applies F frequency masks and T time masks per sample.
    Reference: Park et al. 2019, "SpecAugment".

    Applied to features of shape (B, T_frames, n_mels).
    Only applied during training (set model.train() mode).
    """

    def __init__(
        self,
        freq_mask_param: int = 27,  # max width of each frequency mask (F)
        time_mask_param: int = 100,  # max width of each time mask (T)
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        time_mask_ratio: float | None = 0.05,  # optionally limit mask to p*T
    ) -> None:
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.time_mask_ratio = time_mask_ratio

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T_frames, n_mels)
        Returns:
            x: (B, T_frames, n_mels)  with masks applied
        """
        if not self.training:
            return x

        B, T, F = x.shape

        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = torch.randint(0, self.freq_mask_param + 1, (B,))
            f0 = torch.randint(0, max(1, F - self.freq_mask_param), (B,))
            for b in range(B):
                x[b, :, f0[b] : f0[b] + f[b]] = 0.0

        # Time masking
        t_max = self.time_mask_param
        if self.time_mask_ratio is not None:
            t_max = min(t_max, int(self.time_mask_ratio * T))

        for _ in range(self.n_time_masks):
            t = torch.randint(0, max(1, t_max + 1), (B,))
            t0 = torch.randint(0, max(1, T - t_max), (B,))
            for b in range(B):
                x[b, t0[b] : t0[b] + t[b], :] = 0.0

        return x
