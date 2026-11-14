"""VoiceChanger GUI (Tkinter). Pick devices by name, start/stop, live meters.

Device dropdowns are populated live and refreshable (USB headsets reshuffle
indices on plug/unplug). Engine is pluggable; passthrough for now, RVC later.
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import sounddevice as sd

from audio_engine import AudioEngine
from engines.passthrough import PassthroughConverter

ENGINES = {"Passthrough (test)": PassthroughConverter}
NONE_LABEL = "(none)"


def device_rows(want_input):
    rows = []
    for i, d in enumerate(sd.query_devices()):
        ch = d["max_input_channels"] if want_input else d["max_output_channels"]
        if ch < 1:
            continue
        ha = sd.query_hostapis(d["hostapi"])["name"].replace("Windows ", "")
        rows.append((i, f"[{i}] {d['name']}  ({ha})", ha))
    return rows


def autopick(rows, *needles, prefer_hostapi="WASAPI"):
    pref = [r for r in rows if r[2] == prefer_hostapi]
    for pool in (pref, rows):
        for idx, label, ha in pool:
            low = label.lower()
            if all(n in low for n in needles):
                return label
    return None


class App:
    def __init__(self, root):
        self.root = root
        self.eng = None
        root.title("VoiceChanger")
        root.configure(bg="#14161c")
        root.geometry("560x540")

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TLabel", background="#14161c", foreground="#e6e6e6")
        style.configure("TButton", padding=6)
        style.configure("TCheckbutton", background="#14161c", foreground="#e6e6e6")

        pad = dict(padx=10, pady=4)
        r = 0
        ttk.Label(root, text="VoiceChanger", font=("Segoe UI", 16, "bold")).grid(row=r, column=0, columnspan=3, sticky="w", **pad); r += 1

        ttk.Label(root, text="Engine").grid(row=r, column=0, sticky="w", **pad)
        self.engine_cb = ttk.Combobox(root, values=list(ENGINES), state="readonly", width=40)
        self.engine_cb.current(0); self.engine_cb.grid(row=r, column=1, columnspan=2, sticky="w", **pad); r += 1

        ttk.Label(root, text="Input mic").grid(row=r, column=0, sticky="w", **pad)
        self.in_cb = ttk.Combobox(root, state="readonly", width=46); self.in_cb.grid(row=r, column=1, columnspan=2, sticky="w", **pad); r += 1

        ttk.Label(root, text="Monitor (headphones)").grid(row=r, column=0, sticky="w", **pad)
        self.mon_cb = ttk.Combobox(root, state="readonly", width=46); self.mon_cb.grid(row=r, column=1, columnspan=2, sticky="w", **pad); r += 1

        ttk.Label(root, text="Virtual mic (VB-Cable)").grid(row=r, column=0, sticky="w", **pad)
        self.cab_cb = ttk.Combobox(root, state="readonly", width=46); self.cab_cb.grid(row=r, column=1, columnspan=2, sticky="w", **pad); r += 1

        ttk.Button(root, text="Refresh devices", command=self.refresh).grid(row=r, column=0, sticky="w", **pad)
        self.gain = tk.DoubleVar(value=1.0)
        ttk.Label(root, text="Gain").grid(row=r, column=1, sticky="e", **pad)
        ttk.Scale(root, from_=0.0, to=3.0, variable=self.gain, length=140).grid(row=r, column=2, sticky="w", **pad); r += 1

        self.mon_on = tk.BooleanVar(value=True)
        self.cab_on = tk.BooleanVar(value=False)
        self.bypass = tk.BooleanVar(value=False)
        ttk.Checkbutton(root, text="Monitor to headphones", variable=self.mon_on, command=self.apply_toggles).grid(row=r, column=0, sticky="w", **pad)
        ttk.Checkbutton(root, text="Send to VB-Cable", variable=self.cab_on, command=self.apply_toggles).grid(row=r, column=1, sticky="w", **pad)
        ttk.Checkbutton(root, text="Bypass", variable=self.bypass, command=self.apply_toggles).grid(row=r, column=2, sticky="w", **pad); r += 1

        self.start_btn = ttk.Button(root, text="Start", command=self.toggle)
        self.start_btn.grid(row=r, column=0, columnspan=3, sticky="we", **pad); r += 1

        ttk.Label(root, text="Input level").grid(row=r, column=0, sticky="w", **pad)
        self.in_canvas = tk.Canvas(root, width=300, height=16, bg="#0b0d12", highlightthickness=0)
        self.in_canvas.grid(row=r, column=1, columnspan=2, sticky="w", **pad); r += 1
        ttk.Label(root, text="Output level").grid(row=r, column=0, sticky="w", **pad)
        self.out_canvas = tk.Canvas(root, width=300, height=16, bg="#0b0d12", highlightthickness=0)
        self.out_canvas.grid(row=r, column=1, columnspan=2, sticky="w", **pad); r += 1

        self.stat = ttk.Label(root, text="idle", foreground="#8b90a0")
        self.stat.grid(row=r, column=0, columnspan=3, sticky="w", **pad); r += 1

        self.refresh()
        self.tick()

    def refresh(self):
        ins = device_rows(True)
        outs = device_rows(False)
        self._in = {l: i for i, l, _ in ins}
        self._out = {l: i for i, l, _ in outs}
        self.in_cb["values"] = [l for _, l, _ in ins]
        self.mon_cb["values"] = [NONE_LABEL] + [l for _, l, _ in outs]
        self.cab_cb["values"] = [NONE_LABEL] + [l for _, l, _ in outs]
        pin = autopick(ins, "microphone") or autopick(ins, "mic") or (ins[0][1] if ins else "")
        self.in_cb.set(pin)
        self.mon_cb.set(autopick(outs, "jabra") or autopick(outs, "headphone") or (outs[0][1] if outs else NONE_LABEL))
        self.cab_cb.set(autopick(outs, "cable", "input") or NONE_LABEL)
        self.stat.configure(text=f"{len(ins)} inputs, {len(outs)} outputs. Pick devices and Start.")

    def apply_toggles(self):
        if self.eng:
            self.eng.monitor_on = self.mon_on.get()
            self.eng.cable_on = self.cab_on.get()
            self.eng.bypass = self.bypass.get()
        if hasattr(self, "conv") and self.conv:
            self.conv.gain = self.gain.get()

    def toggle(self):
        if self.eng and self.eng.running:
            self.eng.stop(); self.eng = None
            self.start_btn.configure(text="Start")
            self.stat.configure(text="stopped", foreground="#8b90a0")
            return
        try:
            indev = self._in.get(self.in_cb.get())
            if indev is None:
                self.stat.configure(text="pick an input mic", foreground="#e5a0a0"); return
            mon = self._out.get(self.mon_cb.get()) if self.mon_cb.get() != NONE_LABEL else None
            cab = self._out.get(self.cab_cb.get()) if self.cab_cb.get() != NONE_LABEL else None
            self.conv = ENGINES[self.engine_cb.get()](48000)
            if hasattr(self.conv, "gain"):
                self.conv.gain = self.gain.get()
            self.eng = AudioEngine(self.conv, in_device=indev, monitor_device=mon, cable_device=cab)
            self.eng.monitor_on = self.mon_on.get() and mon is not None
            self.eng.cable_on = self.cab_on.get() and cab is not None
            self.eng.bypass = self.bypass.get()
            self.eng.start()
            self.start_btn.configure(text="Stop")
            self.stat.configure(text=f"running @ {self.eng.sr} Hz  in_ch={self.eng.in_ch} mon_ch={self.eng.mon_ch}", foreground="#00e5c7")
        except Exception as e:
            self.eng = None
            self.start_btn.configure(text="Start")
            self.stat.configure(text=f"ERROR: {e}", foreground="#e5a0a0")

    def _bar(self, canvas, level, color):
        canvas.delete("all")
        w = int(min(level, 1.0) * 300)
        canvas.create_rectangle(0, 0, w, 16, fill=color, width=0)

    def tick(self):
        if self.eng and self.eng.running:
            self._bar(self.in_canvas, self.eng.in_peak, "#4aa3ff")
            self._bar(self.out_canvas, self.eng.out_peak, "#00e5c7")
            self.stat.configure(
                text=f"running @ {self.eng.sr} Hz  lat~{self.eng.latency_ms:.0f}ms  "
                     f"conv {self.eng.last_convert_ms:.0f}ms  overruns {self.eng.overruns}",
                foreground="#00e5c7")
        else:
            self._bar(self.in_canvas, 0, "#4aa3ff")
            self._bar(self.out_canvas, 0, "#00e5c7")
        self.root.after(80, self.tick)


def main():
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.eng and app.eng.stop(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
