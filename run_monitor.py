"""CLI test: real mic -> converter -> headphones (and optionally VB-Cable).

Usage:
  python run_monitor.py --in 1 --monitor 14              # passthrough to headphones
  python run_monitor.py --in 1 --monitor 14 --gain 1.5   # louder, still passthrough
  python run_monitor.py --in 1 --monitor 14 --cable 20   # also feed VB-Cable Input

Speak into the mic; you should hear yourself on the monitor device with the
engine's ~block+overlap latency. Ctrl+C to stop. This proves the pipeline
before any neural engine is attached.
"""
from __future__ import annotations
import argparse
import time
import sys

from audio_engine import AudioEngine
from engines.passthrough import PassthroughConverter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indev", type=int, required=True, help="input device index")
    ap.add_argument("--monitor", type=int, default=None, help="monitor output index (headphones)")
    ap.add_argument("--cable", type=int, default=None, help="VB-Cable Input output index")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--block-ms", type=int, default=160)
    ap.add_argument("--overlap-ms", type=int, default=40)
    ap.add_argument("--gain", type=float, default=1.0)
    args = ap.parse_args()

    if args.monitor is None and args.cable is None:
        print("Pick at least --monitor and/or --cable", file=sys.stderr)
        sys.exit(2)

    conv = PassthroughConverter(args.sr, gain=args.gain)
    eng = AudioEngine(
        converter=conv,
        in_device=args.indev,
        monitor_device=args.monitor,
        cable_device=args.cable,
        samplerate=args.sr,
        block_ms=args.block_ms,
        overlap_ms=args.overlap_ms,
    )
    eng.monitor_on = args.monitor is not None
    eng.cable_on = args.cable is not None

    print(f"engine: {conv.name}  sr={args.sr}  block={args.block_ms}ms overlap={args.overlap_ms}ms")
    print(f"base latency ~{(eng.block + eng.overlap)/eng.sr*1000:.0f} ms (+ convert time)")
    print("Speak now. Ctrl+C to stop.\n")
    eng.start()
    try:
        while True:
            time.sleep(0.5)
            bar_in = "#" * int(eng.in_peak * 20)
            bar_out = "#" * int(eng.out_peak * 20)
            print(f"\rin[{bar_in:20}] out[{bar_out:20}] "
                  f"conv={eng.last_convert_ms:5.1f}ms lat~{eng.latency_ms:5.0f}ms "
                  f"overruns={eng.overruns}   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        eng.stop()


if __name__ == "__main__":
    main()
