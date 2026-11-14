"""Real-time RVC converter: feeds audio directly into the RVC pipeline
(bypassing rvc-python's per-call ffmpeg load) for low-latency streaming."""
import os
import numpy as np
from math import gcd
from .base import Converter

# torch 2.6+ defaults weights_only=True; our downloaded checkpoints need False.
import torch as _torch
if not getattr(_torch, "_vc_load_patched", False):
    _ol = _torch.load
    def _pl(*a, **k):
        k.setdefault("weights_only", False)
        return _ol(*a, **k)
    _torch.load = _pl
    _torch._vc_load_patched = True


class RVCConverter(Converter):
    name = "RVC"

    def __init__(self, model_name, models_dir, device="cuda:0",
                 f0method="rmvpe", f0up_key=0, index_rate=0.5,
                 protect=0.33, rms_mix_rate=0.25, filter_radius=3):
        self.model_name = model_name
        self.models_dir = models_dir
        self.device = device
        self.f0method = f0method
        self.f0up_key = int(f0up_key)
        self.index_rate = index_rate
        self.protect = protect
        self.rms_mix_rate = rms_mix_rate
        self.filter_radius = filter_radius
        self.rvc = None
        self._sr = None
        self.tgt_sr = None
        self.file_index = ""

    def warmup(self, samplerate):
        import tempfile, soundfile as sf
        from rvc_python.infer import RVCInference
        # Swap faiss's CPU read_index for a GPU-resident kNN index (huge latency win
        # on 'Max' quality: index retrieval drops from ~300-450ms to ~25ms). Must run
        # before load_model, which calls faiss.read_index. No-op if gpu_index missing.
        try:
            import gpu_index
            gpu_index.patch_faiss(self.device)
        except Exception:
            pass
        self._sr = int(samplerate)
        self.rvc = RVCInference(models_dir=self.models_dir, device=self.device)
        self.rvc.load_model(self.model_name, version="v2")
        self.rvc.set_params(f0method=self.f0method, f0up_key=self.f0up_key,
                            index_rate=self.index_rate, protect=self.protect,
                            rms_mix_rate=self.rms_mix_rate,
                            filter_radius=self.filter_radius)
        self.file_index = self.rvc.models[self.rvc.current_model].get("index", "") or ""
        # One dummy inference: forces hubert load + CUDA JIT warmup.
        tmp = os.path.join(tempfile.gettempdir(), "_rvc_warm.wav")
        sf.write(tmp, np.zeros(self._sr, dtype=np.float32), self._sr)
        try:
            self.rvc.infer_file(tmp, tmp + ".out.wav")
        except Exception:
            pass
        self.tgt_sr = self.rvc.vc.tgt_sr

    def _rs(self, x, src, dst):
        if src == dst or x.size == 0:
            return x.astype(np.float32)
        from scipy.signal import resample_poly
        g = gcd(int(src), int(dst))
        return resample_poly(x, int(dst) // g, int(src) // g).astype(np.float32)

    def convert(self, block):
        vc = self.rvc.vc
        mono = np.asarray(block, dtype=np.float32).reshape(-1)
        a16 = self._rs(mono, self._sr, 16000)
        mx = np.abs(a16).max() / 0.95 if a16.size else 0
        if mx > 1:
            a16 = a16 / mx
        if vc.hubert_model is None:
            from rvc_python.modules.vc.utils import load_hubert
            vc.hubert_model = load_hubert(vc.config, vc.lib_dir)
        out = vc.pipeline.pipeline(
            vc.hubert_model, vc.net_g, 0, a16, "", [0, 0, 0],
            self.f0up_key, self.f0method, self.file_index, self.index_rate,
            vc.if_f0, self.filter_radius, vc.tgt_sr, 0, self.rms_mix_rate,
            vc.version, self.protect, "")
        out = np.asarray(out, dtype=np.float32)
        if out.size and np.abs(out).max() > 1.5:
            out = out / 32768.0
        return self._rs(out, vc.tgt_sr, self._sr)

    def close(self):
        try:
            if self.rvc:
                self.rvc.unload_model()
        except Exception:
            pass
