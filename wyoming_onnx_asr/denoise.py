"""Speech-enhancement (denoise) backend for pre-ASR audio cleanup.

Uses noisereduce in stationary mode — spectral gating built on a noise
profile learned from the signal itself. This is the textbook fix for
constant baseline speaker noise (hiss, hum, fan). Pure numpy/scipy, fast,
and preserves voice at moderate strength.
"""

from typing import Optional

import numpy as np


_TARGET_LUFS = -16.0
_PEAK_CEILING_DBFS = -1.0


class NoiseReduceDenoiser:
    def __init__(self, prop_decrease: float = 0.75) -> None:
        import noisereduce  # noqa: F401  (fail fast if missing)
        import pyloudnorm  # noqa: F401

        self._prop_decrease = float(prop_decrease)
        self._peak_ceiling = 10 ** (_PEAK_CEILING_DBFS / 20.0)

    def process(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        import noisereduce as nr
        import pyloudnorm as pyln

        assert waveform.ndim == 1, "denoise expects mono 1-D waveform"
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32, copy=False)

        out = nr.reduce_noise(
            y=waveform,
            sr=sample_rate,
            stationary=True,
            prop_decrease=self._prop_decrease,
        )

        # BS.1770 integrated loudness needs >= 0.4s; shorter clips skip LUFS.
        if out.shape[0] >= int(sample_rate * 0.4):
            meter = pyln.Meter(sample_rate)
            loudness = meter.integrated_loudness(out)
            if np.isfinite(loudness):
                out = pyln.normalize.loudness(out, loudness, _TARGET_LUFS)

        peak = float(np.max(np.abs(out)))
        if peak > self._peak_ceiling:
            out = out * (self._peak_ceiling / peak)

        return np.ascontiguousarray(out, dtype=np.float32)

    @property
    def name(self) -> str:
        return (
            f"noisereduce(stationary,prop_decrease={self._prop_decrease})"
            f"+loudnorm({_TARGET_LUFS}LUFS)"
        )


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
