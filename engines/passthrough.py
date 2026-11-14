"""Identity + optional trivial gain. Proves the audio loop end to end.

This is NOT a voice changer; it exists to validate capture -> process -> output,
chunking, crossfade and latency before the real neural engines are wired in.
"""
from __future__ import annotations
import numpy as np
from .base import Converter


class PassthroughConverter(Converter):
    name = "passthrough"

    def __init__(self, samplerate: int, gain: float = 1.0):
        super().__init__(samplerate)
        self.gain = gain

    def convert(self, block: np.ndarray) -> np.ndarray:
        if self.gain == 1.0:
            return block
        return np.clip(block * self.gain, -1.0, 1.0).astype(np.float32)
