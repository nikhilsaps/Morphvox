"""Offline self-test: run synthetic audio through the engine's worker logic
(no audio devices) to validate chunking, crossfade and converter wiring."""
import numpy as np
from engines.passthrough import PassthroughConverter
from audio_engine import _Ring

# ring buffer wraparound test
r = _Ring(10)
assert r.write(np.arange(7, dtype=np.float32)) == 7
assert np.allclose(r.read(4), [0, 1, 2, 3])
assert r.write(np.arange(100, 106, dtype=np.float32)) == 6  # wraps
got = r.read(9)
assert np.allclose(got, [4, 5, 6, 100, 101, 102, 103, 104, 105]), got
print("ring buffer OK")

# converter test
c = PassthroughConverter(48000, gain=2.0)
x = np.array([0.1, -0.2, 0.9], dtype=np.float32)
y = c.convert(x)
assert np.allclose(y, [0.2, -0.4, 1.0]), y  # 0.9*2 clipped to 1.0
print("passthrough+gain+clip OK")
print("SELFTEST PASSED")
