"""Pluggable voice-conversion engine interface.

Every engine takes a mono float32 numpy block at a known samplerate and returns
a mono float32 block of the SAME length and samplerate. Engines own any internal
resampling (e.g. RVC works at 16k internally but must return the caller's rate).
"""
from __future__ import annotations
import numpy as np


class Converter:
    name = "base"

    def __init__(self, samplerate: int):
        self.samplerate = samplerate

    def warmup(self, samplerate: int | None = None) -> None:
        """Optional: run a dummy conversion so the first real block isn't slow."""
        pass

    def convert(self, block: np.ndarray) -> np.ndarray:
        """block: float32 mono, shape (n,). Return float32 mono, shape (n,)."""
        raise NotImplementedError

    def close(self) -> None:
        pass
