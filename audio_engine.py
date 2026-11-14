"""VoiceChanger streaming audio engine — context-buffer (low-latency) design.

Latency model:  mouth-to-ear  ~=  hop_ms + crossfade_ms + convert_ms + I/O.
Instead of "buffer a whole block then convert", we slide:
  - every tick we pull `hop` of NEW audio from the mic,
  - prepend `context` of PAST audio (already buffered -> costs no latency) so the
    model converts coherently across phoneme boundaries,
  - convert the whole window on the GPU,
  - emit only the newest `hop`, crossfading the seam with the previous tick.
So latency is driven by `hop`, not by how much context the model needs.

Decoupled threads so a slow converter never stalls the audio callbacks:
  input cb -> input ring -> worker (context window, convert, crossfade)
           -> monitor ring + cable ring -> output cbs
Device indices are negotiated at open time (USB headsets shuffle indices).
Internally mono float32 at the input device's native rate.
"""
from __future__ import annotations
import threading
import time
import numpy as np
import sounddevice as sd

from engines.base import Converter


class _Ring:
    def __init__(self, capacity: int):
        self.buf = np.zeros(capacity, dtype=np.float32)
        self.cap = capacity
        self.w = 0
        self.r = 0
        self.count = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray) -> int:
        with self.lock:
            n = min(len(data), self.cap - self.count)
            if n:
                first = min(n, self.cap - self.w)
                self.buf[self.w:self.w + first] = data[:first]
                rest = n - first
                if rest:
                    self.buf[:rest] = data[first:first + rest]
                self.w = (self.w + n) % self.cap
                self.count += n
            return n

    def read(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        with self.lock:
            m = min(n, self.count)
            if m:
                first = min(m, self.cap - self.r)
                out[:first] = self.buf[self.r:self.r + first]
                rest = m - first
                if rest:
                    out[first:first + rest] = self.buf[:rest]
                self.r = (self.r + m) % self.cap
                self.count -= m
            return out

    def available(self) -> int:
        with self.lock:
            return self.count

    def clear(self):
        with self.lock:
            self.w = self.r = self.count = 0


def negotiate(device, is_input, want_sr):
    """Return (samplerate, channels) that actually opens, or raise."""
    d = sd.query_devices(device)
    maxch = d["max_input_channels"] if is_input else d["max_output_channels"]
    if maxch < 1:
        raise RuntimeError(f"device {device!r} has no {'input' if is_input else 'output'} channels")
    sr_candidates = [want_sr, int(d["default_samplerate"]), 48000, 44100]
    ch_order = [1, 2] if is_input else [min(2, maxch), 1]
    ch_order = [c for c in ch_order if 1 <= c <= maxch] or [maxch]
    last = None
    for sr in dict.fromkeys(sr_candidates):
        for ch in ch_order:
            try:
                sd.check_input_settings(device=device, channels=ch, samplerate=sr, dtype="float32") \
                    if is_input else \
                    sd.check_output_settings(device=device, channels=ch, samplerate=sr, dtype="float32")
                return sr, ch
            except Exception as e:
                last = e
    raise RuntimeError(f"no working format for device {device!r}: {last}")


class AudioEngine:
    def __init__(self, converter: Converter, in_device, monitor_device=None,
                 cable_device=None, samplerate=48000,
                 block_ms=160, context_ms=200, crossfade_ms=40):
        self.converter = converter
        self.in_device = in_device
        self.monitor_device = monitor_device
        self.cable_device = cable_device
        self.sr = samplerate
        self.block_ms = block_ms          # hop: new audio emitted per tick
        self.context_ms = context_ms      # past audio fed to model (no latency cost)
        self.crossfade_ms = crossfade_ms  # seam blend between consecutive hops

        self.monitor_on = True
        self.cable_on = False
        self.bypass = False
        # Noise gate: RVC hallucinates static on silence, so below threshold we
        # emit clean silence instead of converting. hold keeps it open through
        # brief pauses so word-tails aren't clipped.
        self.gate_on = True
        self.gate_thresh = 0.008     # input RMS (linear) to count as speech
        self.gate_hold_ms = 450

        self.running = False
        self._worker = None
        self._streams = []
        self.in_ch = 1
        self.mon_ch = 1
        self.cable_ch = 1

        self.last_convert_ms = 0.0
        self.in_peak = 0.0
        self.out_peak = 0.0
        self.overruns = 0
        self.error = None

        cap = max(1, samplerate) * 4
        self.in_ring = _Ring(cap)
        self.mon_ring = _Ring(cap)
        self.cable_ring = _Ring(cap)

    def _in_cb(self, indata, frames, t, status):
        mono = indata[:, 0].copy() if self.in_ch == 1 else indata.mean(axis=1).astype(np.float32)
        if frames:
            self.in_peak = max(self.in_peak * 0.8, float(np.max(np.abs(mono))))
        if len(mono) - self.in_ring.write(mono):
            self.overruns += 1

    def _mon_cb(self, outdata, frames, t, status):
        data = self.mon_ring.read(frames) if self.monitor_on else np.zeros(frames, np.float32)
        outdata[:] = np.repeat(data[:, None], self.mon_ch, axis=1) if self.mon_ch > 1 else data[:, None]

    def _cable_cb(self, outdata, frames, t, status):
        data = self.cable_ring.read(frames) if self.cable_on else np.zeros(frames, np.float32)
        outdata[:] = np.repeat(data[:, None], self.cable_ch, axis=1) if self.cable_ch > 1 else data[:, None]

    def _run(self):
        hop, ctx, xf = self.hop, self.ctx, self.xf
        history = np.zeros(ctx + xf, dtype=np.float32)   # past raw input
        prev_tail = np.zeros(xf, dtype=np.float32)        # converted seam tail
        fade = np.linspace(0, 1, xf, dtype=np.float32) if xf else np.zeros(0, np.float32)
        fade_in, fade_out = fade, 1.0 - fade
        hold_blocks = max(1, int(self.gate_hold_ms / max(1, self.block_ms)))
        hold = 0
        ramp_dn = np.linspace(1, 0, hop, dtype=np.float32)
        gate_was_open = False
        while self.running:
            if self.in_ring.available() < hop:
                time.sleep(0.002)
                continue
            new = self.in_ring.read(hop)
            window = np.concatenate([history, new])       # (ctx+xf)+hop
            history = window[-(ctx + xf):].copy() if (ctx + xf) else np.zeros(0, np.float32)

            # noise gate: skip convert entirely on silence -> clean silence out
            rms = float(np.sqrt(np.mean(new * new))) if len(new) else 0.0
            if self.gate_on and not self.bypass:
                if rms >= self.gate_thresh:
                    hold = hold_blocks
                else:
                    hold -= 1
                gate_open = hold > 0
            else:
                gate_open = True
            if not gate_open:
                emit = np.zeros(hop, dtype=np.float32)
                if gate_was_open:            # smooth close on the first silent block
                    emit[:] = 0.0
                prev_tail = prev_tail * 0.5
                gate_was_open = False
                self.last_convert_ms = 0.0
                if self.monitor_on:
                    self.mon_ring.write(emit)
                if self.cable_on:
                    self.cable_ring.write(emit)
                continue
            gate_was_open = True

            t0 = time.perf_counter()
            conv = window if self.bypass else self.converter.convert(window)
            self.last_convert_ms = (time.perf_counter() - t0) * 1000.0
            conv = np.asarray(conv, dtype=np.float32).reshape(-1)
            if len(conv) != len(window):
                conv = np.resize(conv, len(window)).astype(np.float32)

            fresh = conv[-(hop + xf):] if (hop + xf) <= len(conv) else conv
            if len(fresh) < hop + xf:
                fresh = np.pad(fresh, (hop + xf - len(fresh), 0))
            if xf:
                head = fresh[:xf] * fade_in + prev_tail * fade_out
                emit = np.concatenate([head, fresh[xf:hop]])
                prev_tail = fresh[hop:hop + xf].copy()
            else:
                emit = fresh[:hop]
            emit = emit[:hop]
            if len(emit) < hop:
                emit = np.pad(emit, (0, hop - len(emit)))

            if len(emit):
                self.out_peak = max(self.out_peak * 0.8, float(np.max(np.abs(emit))))
            if self.monitor_on:
                self.mon_ring.write(emit)
            if self.cable_on:
                self.cable_ring.write(emit)

    def start(self):
        if self.running:
            return
        self.error = None
        try:
            in_sr, self.in_ch = negotiate(self.in_device, True, self.sr)
            self.sr = in_sr
            self.hop = max(1, int(self.sr * self.block_ms / 1000))
            self.ctx = int(self.sr * self.context_ms / 1000)
            self.xf = min(int(self.sr * self.crossfade_ms / 1000), self.hop // 2)
            for r in (self.in_ring, self.mon_ring, self.cable_ring):
                r.clear()

            try:
                self.converter.warmup(self.sr)
            except Exception as e:
                print("warmup failed:", e)

            self.running = True
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()

            s = sd.InputStream(device=self.in_device, channels=self.in_ch, samplerate=self.sr,
                               dtype="float32", callback=self._in_cb)
            s.start(); self._streams.append(s)

            if self.monitor_device is not None:
                _, self.mon_ch = negotiate(self.monitor_device, False, self.sr)
                s = sd.OutputStream(device=self.monitor_device, channels=self.mon_ch, samplerate=self.sr,
                                    dtype="float32", callback=self._mon_cb)
                s.start(); self._streams.append(s)
            if self.cable_device is not None:
                _, self.cable_ch = negotiate(self.cable_device, False, self.sr)
                s = sd.OutputStream(device=self.cable_device, channels=self.cable_ch, samplerate=self.sr,
                                    dtype="float32", callback=self._cable_cb)
                s.start(); self._streams.append(s)
        except Exception as e:
            self.error = str(e)
            self.running = False
            self.stop()
            raise

    def swap_converter(self, new_conv):
        """Hot-swap the voice model while streaming. Warms the new converter
        first (old voice keeps playing meanwhile), then atomically replaces it.
        Only valid when hop/context/crossfade are unchanged (same quality)."""
        try:
            new_conv.warmup(self.sr)
        except Exception as e:
            print("swap warmup failed:", e)
            try:
                new_conv.close()
            except Exception:
                pass
            return False
        old = self.converter
        self.converter = new_conv          # worker picks it up next tick
        try:
            old.close()
        except Exception:
            pass
        return True

    def stop(self):
        self.running = False
        if self._worker:
            self._worker.join(timeout=1.0)
            self._worker = None
        for s in self._streams:
            try:
                s.stop(); s.close()
            except Exception:
                pass
        self._streams = []
        try:
            self.converter.close()
        except Exception:
            pass

    @property
    def latency_ms(self) -> float:
        hop = getattr(self, "hop", int(self.sr * self.block_ms / 1000))
        xf = getattr(self, "xf", int(self.sr * self.crossfade_ms / 1000))
        return (hop + xf) / self.sr * 1000.0 + self.last_convert_ms
