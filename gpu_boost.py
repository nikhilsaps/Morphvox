"""GPU latency helpers (Windows / consumer laptop GPU).

- keep_alive(): a tiny op on a side CUDA stream every ~30ms stops the GPU from
  dropping to its idle clock (225 MHz on the RTX 5060), so the FIRST real convert
  is already at full boost -> kills the 80ms<->160ms cold/warm swing.
- set_high_priority(): bumps this process to Windows HIGH priority (ctypes, no deps)
  so the OS scheduler doesn't let background work steal the audio thread.
- try_lock_clocks(): best-effort nvidia-smi -lgc (usually blocked on WDDM consumer
  GPUs; keep_alive is the reliable path, this is a bonus when it works).
"""
from __future__ import annotations
import threading, time, ctypes, subprocess


def set_high_priority() -> bool:
    try:
        HIGH_PRIORITY_CLASS = 0x00000080
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        return bool(k.SetPriorityClass(k.GetCurrentProcess(), HIGH_PRIORITY_CLASS))
    except Exception:
        return False


def start_keepalive(interval: float = 0.03):
    """Returns a threading.Event; set() it to stop. None if no CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
    except Exception:
        return None
    stop = threading.Event()

    def loop():
        import torch
        torch.cuda.init()
        s = torch.cuda.Stream()
        a = torch.ones((512, 512), device="cuda", dtype=torch.float16)
        while not stop.is_set():
            try:
                with torch.cuda.stream(s):
                    _ = a @ a
                s.synchronize()
            except Exception:
                break
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True, name="gpu-keepalive").start()
    return stop


def try_lock_clocks(floor_sm: int = 1800, ceil_sm: int = 3090) -> str:
    """Best-effort: force a HIGH clock floor so the GPU never idles to 225 MHz
    (needs admin). Returns status string. Reset with reset_clocks()."""
    try:
        r = subprocess.run(["nvidia-smi", "-lgc", f"{floor_sm},{ceil_sm}"],
                           capture_output=True, text=True, timeout=10)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"lock unavailable: {e}"


def reset_clocks() -> str:
    try:
        r = subprocess.run(["nvidia-smi", "-rgc"], capture_output=True, text=True, timeout=10)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"reset unavailable: {e}"
