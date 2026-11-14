"""GPU faiss replacement for RVC real-time index retrieval.

rvc-python re-reads the .index file AND reconstructs every stored vector on EVERY
frame, then runs a CPU faiss kNN -> ~300-450ms. We monkeypatch faiss.read_index to
return a cached, GPU-resident index whose .search() is a torch distance+topk on the
GPU (exact kNN), and whose .reconstruct_n() returns the cached vectors instantly.

Call patch_faiss(device) once before any RVC model loads. Idempotent.
"""
from __future__ import annotations
import threading
import numpy as np

_cache = {}
_lock = threading.Lock()


class _GPUIndex:
    def __init__(self, real_index, device):
        import torch
        self._torch = torch
        self._device = device
        self.ntotal = real_index.ntotal
        self.d = real_index.d
        # IVF indexes need a direct map before reconstruct_n.
        try:
            real_index.make_direct_map()
        except Exception:
            pass
        big = np.ascontiguousarray(real_index.reconstruct_n(0, real_index.ntotal)).astype("float32")
        self._big_npy = big                         # kept for the CPU gather/blend in vc()
        self._vecs = torch.from_numpy(big).to(device)       # [N, d]
        self._vn = (self._vecs * self._vecs).sum(1)          # [N]  precomputed ||b||^2

    def reconstruct_n(self, i0, n):
        return self._big_npy                         # instant (cached)

    def search(self, npy, k):
        t = self._torch
        q = t.from_numpy(np.ascontiguousarray(npy, dtype="float32")).to(self._device)  # [T, d]
        qn = (q * q).sum(1, keepdim=True)                     # [T, 1]
        # squared L2 = ||q||^2 + ||b||^2 - 2 q.b  (matches faiss IndexFlatL2 'score')
        d2 = qn + self._vn.unsqueeze(0) - 2.0 * (q @ self._vecs.t())   # [T, N]
        d2.clamp_(min=1e-8)
        score, ix = t.topk(d2, k, dim=1, largest=False)       # [T, k]
        return (score.detach().cpu().numpy().astype("float32"),
                ix.detach().cpu().numpy().astype("int64"))


def patch_faiss(device: str = "cuda:0") -> bool:
    try:
        import faiss, torch
        if not torch.cuda.is_available():
            return False
        if getattr(faiss, "_gpu_patched", False):
            return True
        _real = faiss.read_index

        def read_index(path, *a, **k):
            with _lock:
                if path not in _cache:
                    _cache[path] = _GPUIndex(_real(path, *a, **k), device)
                return _cache[path]

        faiss.read_index = read_index
        faiss._gpu_patched = True
        return True
    except Exception as e:
        print("patch_faiss failed:", e)
        return False
