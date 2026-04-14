"""Speech-enhancement (denoise) backends for pre-ASR audio cleanup."""

import logging
from typing import Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)


class DeepFilterNetDenoiser:
    """DeepFilterNet (DF3) speech enhancement.

    Runs at 48 kHz natively; input is resampled in/out via soxr.
    """

    def __init__(self) -> None:
        from df.enhance import init_df

        self._model, self._df_state, _ = init_df()
        self._native_sr = int(self._df_state.sr())

    def process(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        from df.enhance import enhance

        assert waveform.ndim == 1, "denoise expects mono 1-D waveform"
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32, copy=False)

        if sample_rate != self._native_sr:
            import soxr

            up = soxr.resample(waveform, sample_rate, self._native_sr)
        else:
            up = waveform

        import torch

        tensor = torch.from_numpy(up).unsqueeze(0)
        enhanced = enhance(self._model, self._df_state, tensor)
        out = enhanced.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

        if sample_rate != self._native_sr:
            import soxr

            out = soxr.resample(out, self._native_sr, sample_rate)

        return np.ascontiguousarray(out, dtype=np.float32)

    @property
    def name(self) -> str:
        return "deepfilternet"


def build_denoiser(enabled: bool) -> Optional[DeepFilterNetDenoiser]:
    if not enabled:
        return None
    try:
        return DeepFilterNetDenoiser()
    except ImportError as e:
        raise RuntimeError(
            f"--denoise requires the 'denoise' extra "
            f"(pip install 'wyoming-onnx-asr[denoise]'); import failed: {e!r}"
        ) from e
