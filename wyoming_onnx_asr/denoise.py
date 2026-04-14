"""Speech-enhancement (denoise) backend for pre-ASR audio cleanup.

Uses ClearVoice (modelscope/ClearerVoice-Studio) with FRCRN_SE_16K, which runs
natively at 16 kHz — matching typical Wyoming ASR input without resampling.
"""

from typing import Optional

import numpy as np

_NATIVE_SR = 16000
_MODEL_NAME = "FRCRN_SE_16K"


class ClearVoiceDenoiser:
    def __init__(self) -> None:
        from clearvoice import ClearVoice

        self._cv = ClearVoice(
            task="speech_enhancement", model_names=[_MODEL_NAME]
        )

    def process(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        assert waveform.ndim == 1, "denoise expects mono 1-D waveform"
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32, copy=False)

        if sample_rate != _NATIVE_SR:
            import soxr

            work = soxr.resample(waveform, sample_rate, _NATIVE_SR)
        else:
            work = waveform

        batched = work.reshape(1, -1)
        out = self._cv(batched, False)
        out = np.asarray(out, dtype=np.float32).reshape(-1)

        if sample_rate != _NATIVE_SR:
            import soxr

            out = soxr.resample(out, _NATIVE_SR, sample_rate)

        return np.ascontiguousarray(out, dtype=np.float32)

    @property
    def name(self) -> str:
        return f"clearvoice:{_MODEL_NAME}"


def build_denoiser(enabled: bool) -> Optional[ClearVoiceDenoiser]:
    if not enabled:
        return None
    try:
        return ClearVoiceDenoiser()
    except ImportError as e:
        raise RuntimeError(
            f"--denoise requires the 'denoise' extra "
            f"(pip install 'wyoming-onnx-asr[denoise]'); import failed: {e!r}"
        ) from e
