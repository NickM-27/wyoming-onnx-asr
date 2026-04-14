"""Speech-enhancement (denoise) backend for pre-ASR audio cleanup.

Uses noisereduce in stationary mode — spectral gating built on a noise
profile learned from the signal itself. This is the textbook fix for
constant baseline speaker noise (hiss, hum, fan). Pure numpy/scipy, fast,
and preserves voice at moderate strength.
"""

from typing import Optional

import numpy as np


class NoiseReduceDenoiser:
    def __init__(self, prop_decrease: float = 0.9) -> None:
        import noisereduce  # noqa: F401  (fail fast if missing)

        self._prop_decrease = float(prop_decrease)

    def process(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        import noisereduce as nr

        assert waveform.ndim == 1, "denoise expects mono 1-D waveform"
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32, copy=False)

        out = nr.reduce_noise(
            y=waveform,
            sr=sample_rate,
            stationary=True,
            prop_decrease=self._prop_decrease,
        )
        return np.ascontiguousarray(out, dtype=np.float32)

    @property
    def name(self) -> str:
        return f"noisereduce(stationary,prop_decrease={self._prop_decrease})"


def build_denoiser(enabled: bool) -> Optional[NoiseReduceDenoiser]:
    if not enabled:
        return None
    try:
        return NoiseReduceDenoiser()
    except ImportError as e:
        raise RuntimeError(
            f"--denoise requires the 'denoise' extra "
            f"(pip install 'wyoming-onnx-asr[denoise]'); import failed: {e!r}"
        ) from e
