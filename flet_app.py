"""VoiceChanger - Flet GUI (light theme).

Flet 1.1.2 API: entrypoint ft.run(main); ft.Colors / ft.Icons; ft.ThemeMode;
ft.Dropdown + ft.dropdown.Option; ft.ProgressBar for level meters.
Device dropdowns are by name and refreshable (USB headsets shuffle indices).

Voice (WHO) and Quality (HOW) are independent dropdowns:
  * changing Voice while running HOT-SWAPS the model (no stop/start, old voice
    keeps playing during the few-second warmup) via AudioEngine.swap_converter.
  * changing Quality restarts the engine (hop/window may change).
"""
from __future__ import annotations
import threading
import time
import flet as ft
import sounddevice as sd

import os
from audio_engine import AudioEngine
from engines.passthrough import PassthroughConverter
try:
    import gpu_boost
except Exception:
    gpu_boost = None

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rvc_models")


# --- Voice = which model + pitch shift (f0up_key: +12 female/anime, 0 male, +2 cartoon)
VOICES = {
    "Amiya (anime, F)":   ("amiya-rvc", 12),
    "Hatsune Miku (F)":   ("Hatsune_Miku__RVC_v2_", 12),
    "Tomori (anime, F)":  ("tomori-rvc", 12),
    "Donald Trump (M)":   ("Donald_Trump__RVC_v2_", 0),
    "Drake (M)":          ("Drake_RVC", 0),
    "SpongeBob (M)":      ("SpongeBob_SquarePants__RVC_v2_", 2),
    "Stewie (baby)":      ("Stewie_Griffin_RVCv2", 3),
    "Wonder Kid (child)": ("wonder-kid-potsworthandco-rvc-v2", 5),
    "Neha Kakkar (IN, F)":("NehaKakkar", 12),
    "Atif Aslam (IN, M)": ("AtifAslamHighQualityNewVoiceAiModelNew", 0),
    "Shivangi (IN, F)":   ("Shivangi_Voice_RVC_v2", 12),
}
PASSTHROUGH = "Passthrough (test)"

# --- Quality = HOW it converts. (f0method, index_rate, rms_mix, hop_ms, ctx_ms, xf_ms)
# Measured on box (RTX 5060): pm ~80ms convert, rmvpe ~122ms, faiss index +~300ms.
# Convert cost is ~fixed regardless of window, so context is free.
QUALITY = {
    "Fast (pm)":          ("pm",    0.0, 0.25, 140, 320, 30),
    "Natural (rmvpe)":    ("rmvpe", 0.0, 0.50, 160, 320, 30),
    "Max (rmvpe+index)":  ("rmvpe", 0.5, 0.50, 400, 320, 30),
}
DEFAULT_VOICE = "Amiya (anime, F)"
DEFAULT_QUALITY = "Natural (rmvpe)"
NONE = "(none)"


def build_conv(voice_name, quality_name, samplerate):
    """Returns (converter, hop_ms, ctx_ms, xf_ms) for a voice+quality combo."""
    f0method, index_rate, rms_mix, hop, ctx, xf = QUALITY[quality_name]
    if voice_name == PASSTHROUGH:
        return PassthroughConverter(samplerate), 60, 0, 20
    model_name, f0up_key = VOICES[voice_name]
    from engines.rvc import RVCConverter
    conv = RVCConverter(model_name, _MODELS_DIR, device="cuda:0",
                        f0method=f0method, f0up_key=f0up_key,
                        index_rate=index_rate, protect=0.33, rms_mix_rate=rms_mix)
    return conv, hop, ctx, xf


def device_opts(want_input):
    opts, lookup = [], {}
    for i, d in enumerate(sd.query_devices()):
        ch = d["max_input_channels"] if want_input else d["max_output_channels"]
        if ch < 1:
            continue
        ha = sd.query_hostapis(d["hostapi"])["name"].replace("Windows ", "")
        label = f"[{i}] {d['name']} ({ha})"
        opts.append(ft.dropdown.Option(key=str(i), text=label))
        lookup[str(i)] = (i, ha)
    return opts, lookup


def autopick(lookup, want_input, *needles, prefer="WASAPI"):
    items = []
    for i, d in enumerate(sd.query_devices()):
        ch = d["max_input_channels"] if want_input else d["max_output_channels"]
        if ch < 1:
            continue
        ha = sd.query_hostapis(d["hostapi"])["name"].replace("Windows ", "")
        items.append((str(i), d["name"].lower(), ha))
    for pool in ([x for x in items if x[2] == prefer], items):
        for key, name, ha in pool:
            if all(n in name for n in needles):
                return key
    return None


def main(page: ft.Page):
    page.title = "Morphvox"
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(color_scheme_seed=ft.Colors.CYAN)
    page.bgcolor = "#0d1117"
    page.padding = 24
    page.window.width = 600
    page.window.height = 860
    page.scroll = ft.ScrollMode.AUTO

    state = {"eng": None, "conv": None, "alive": True, "busy": False}
    if gpu_boost:
        gpu_boost.set_high_priority()  # steadier audio thread scheduling

    voice_opts = [ft.dropdown.Option(k) for k in VOICES] + [ft.dropdown.Option(PASSTHROUGH)]
    engine_dd = ft.Dropdown(label="Voice", options=voice_opts, value=DEFAULT_VOICE, width=500)
    quality_dd = ft.Dropdown(label="Quality", options=[ft.dropdown.Option(k) for k in QUALITY],
                             value=DEFAULT_QUALITY, width=500)
    in_dd = ft.Dropdown(label="Input mic", width=500)
    mon_dd = ft.Dropdown(label="Monitor (headphones)", width=500)
    cab_dd = ft.Dropdown(label="Virtual mic (VB-Cable)", width=500)

    gain = ft.Slider(min=0, max=3, value=1.0, divisions=30, label="{value}", width=260)
    monitor_sw = ft.Switch(label="Monitor to headphones", value=True)
    cable_sw = ft.Switch(label="Send to VB-Cable", value=False)
    bypass_sw = ft.Switch(label="Bypass", value=False)

    in_bar = ft.ProgressBar(value=0, width=500, bar_height=10, border_radius=6,
                            color=ft.Colors.CYAN_400, bgcolor="#1b2330")
    out_bar = ft.ProgressBar(value=0, width=500, bar_height=10, border_radius=6,
                             color=ft.Colors.TEAL_ACCENT_400, bgcolor="#1b2330")
    status = ft.Text("idle", color=ft.Colors.GREY_500, size=12, selectable=True)
    start_btn = ft.FilledButton("Start", width=500, height=52,
                                icon=ft.Icons.PLAY_ARROW_ROUNDED)

    in_lk = {}
    mon_lk = {}
    cab_lk = {}

    def set_status(msg, color=ft.Colors.GREY_600):
        status.value = msg
        status.color = color
        page.update()

    def refresh(_=None):
        nonlocal in_lk, mon_lk, cab_lk
        io, in_lk = device_opts(True)
        oo, mon_lk = device_opts(False)
        co, cab_lk = device_opts(False)
        in_dd.options = io
        mon_dd.options = [ft.dropdown.Option(key=NONE, text=NONE)] + oo
        cab_dd.options = [ft.dropdown.Option(key=NONE, text=NONE)] + co
        in_dd.value = autopick(in_lk, True, "microphone") or (io[0].key if io else None)
        mon_dd.value = autopick(mon_lk, False, "jabra") or autopick(mon_lk, False, "headphone") or (oo[0].key if oo else NONE)
        cab_dd.value = autopick(cab_lk, False, "cable", "input") or NONE
        set_status(f"{len(io)} inputs, {len(oo)} outputs. Pick devices and Start.")

    def apply_toggles(_=None):
        eng = state["eng"]
        if eng:
            eng.monitor_on = monitor_sw.value
            eng.cable_on = cable_sw.value
            eng.bypass = bypass_sw.value
        if state["conv"] is not None and hasattr(state["conv"], "gain"):
            state["conv"].gain = gain.value

    def _start_worker():
        try:
            indev = in_lk.get(in_dd.value, (None,))[0]
            if indev is None:
                set_status("pick an input mic", ft.Colors.RED_400); return
            mon = mon_lk.get(mon_dd.value, (None,))[0] if mon_dd.value != NONE else None
            cab = cab_lk.get(cab_dd.value, (None,))[0] if cab_dd.value != NONE else None
            set_status(f"loading {engine_dd.value} / {quality_dd.value} ...", ft.Colors.AMBER_700)
            conv, hop_ms, ctx_ms, xf_ms = build_conv(engine_dd.value, quality_dd.value, 48000)
            if hasattr(conv, "gain"):
                conv.gain = gain.value
            eng = AudioEngine(conv, in_device=indev, monitor_device=mon,
                              cable_device=cab, block_ms=hop_ms,
                              context_ms=ctx_ms, crossfade_ms=xf_ms)
            eng.monitor_on = monitor_sw.value and mon is not None
            eng.cable_on = cable_sw.value and cab is not None
            eng.bypass = bypass_sw.value
            eng.start()
            state["eng"], state["conv"] = eng, conv
            start_btn.text = "Stop"
            set_status(f"running @ {eng.sr} Hz  in_ch={eng.in_ch} mon_ch={eng.mon_ch}", ft.Colors.TEAL_700)
        except Exception as e:
            state["eng"] = None
            start_btn.text = "Start"
            set_status(f"ERROR: {e}", ft.Colors.RED_400)
        finally:
            state["busy"] = False
            page.update()

    def stop_engine():
        eng = state["eng"]
        if eng:
            eng.stop()
        state["eng"] = None
        state["conv"] = None
        start_btn.text = "Start"
        set_status("stopped")

    def toggle(_=None):
        if state["busy"]:
            return
        eng = state["eng"]
        if eng and eng.running:
            stop_engine()
            return
        state["busy"] = True
        start_btn.text = "Loading..."
        page.update()
        threading.Thread(target=_start_worker, daemon=True).start()

    def _swap_worker(voice_name, quality_name):
        try:
            conv, hop_ms, ctx_ms, xf_ms = build_conv(voice_name, quality_name, state["eng"].sr)
            if hasattr(conv, "gain"):
                conv.gain = gain.value
            ok = state["eng"].swap_converter(conv)
            if ok:
                state["conv"] = conv
                set_status(f"switched to {voice_name}", ft.Colors.TEAL_700)
            else:
                set_status("swap failed (kept previous voice)", ft.Colors.RED_400)
        except Exception as e:
            set_status(f"swap ERROR: {e}", ft.Colors.RED_400)
        finally:
            state["busy"] = False
            page.update()

    def on_voice_change(_=None):
        eng = state["eng"]
        if not (eng and eng.running):
            return                      # just a selection; takes effect on Start
        if state["busy"]:
            return
        state["busy"] = True
        set_status(f"switching to {engine_dd.value} (old voice plays until ready)...", ft.Colors.AMBER_700)
        threading.Thread(target=_swap_worker, args=(engine_dd.value, quality_dd.value), daemon=True).start()

    def on_quality_change(_=None):
        eng = state["eng"]
        if not (eng and eng.running):
            return
        if state["busy"]:
            return
        # Quality may change hop/window -> full restart.
        stop_engine()
        state["busy"] = True
        start_btn.text = "Loading..."
        page.update()
        threading.Thread(target=_start_worker, daemon=True).start()

    def meter_loop():
        while state["alive"]:
            eng = state["eng"]
            if eng and eng.running and not state["busy"]:
                in_bar.value = min(eng.in_peak, 1.0)
                out_bar.value = min(eng.out_peak, 1.0)
                status.value = (f"running @ {eng.sr} Hz  lat~{eng.latency_ms:.0f}ms  "
                                f"conv {eng.last_convert_ms:.0f}ms  overruns {eng.overruns}")
                status.color = ft.Colors.TEAL_700
            elif not (eng and eng.running):
                in_bar.value = 0
                out_bar.value = 0
            try:
                page.update()
            except Exception:
                break
            time.sleep(0.1)

    def on_close(_):
        state["alive"] = False
        if state["eng"]:
            state["eng"].stop()

    engine_dd.on_change = on_voice_change
    quality_dd.on_change = on_quality_change
    gain.on_change = apply_toggles
    monitor_sw.on_change = apply_toggles
    cable_sw.on_change = apply_toggles
    bypass_sw.on_change = apply_toggles
    start_btn.on_click = toggle
    page.on_close = on_close

    def card(title, *controls, icon=None):
        head = ft.Row(
            [ft.Icon(icon, size=16, color=ft.Colors.CYAN_300)] if icon else [],
            spacing=8)
        head.controls.append(
            ft.Text(title, size=12, weight=ft.FontWeight.W_700,
                    color=ft.Colors.CYAN_200))
        return ft.Container(
            content=ft.Column([head, *controls], spacing=12, tight=True),
            padding=18, border_radius=16, bgcolor="#161b22",
            border=ft.Border.all(1, "#232c38"),
        )

    header = ft.Row([
        ft.Icon(ft.Icons.GRAPHIC_EQ_ROUNDED, size=30, color=ft.Colors.CYAN_300),
        ft.Column([
            ft.Text("Morphvox", size=24, weight=ft.FontWeight.BOLD),
            ft.Text("real-time AI voice changer", size=12, color=ft.Colors.GREY_500),
        ], spacing=0),
    ], spacing=12)

    meters = ft.Column([
        ft.Row([ft.Text("Input", size=11, color=ft.Colors.GREY_400, width=50), in_bar]),
        ft.Row([ft.Text("Output", size=11, color=ft.Colors.GREY_400, width=50), out_bar]),
    ], spacing=10)

    page.add(
        ft.Column([
            header,
            card("VOICE  &  QUALITY", engine_dd, quality_dd, icon=ft.Icons.RECORD_VOICE_OVER_ROUNDED),
            card("AUDIO DEVICES",
                 in_dd, mon_dd, cab_dd,
                 ft.OutlinedButton("Refresh devices", icon=ft.Icons.REFRESH, on_click=refresh),
                 icon=ft.Icons.HEADSET_MIC_ROUNDED),
            card("CONTROLS",
                 ft.Row([monitor_sw, cable_sw, bypass_sw], wrap=True),
                 ft.Row([ft.Icon(ft.Icons.VOLUME_UP_ROUNDED, size=16, color=ft.Colors.GREY_400),
                         ft.Text("Gain", size=12, color=ft.Colors.GREY_400), gain]),
                 icon=ft.Icons.TUNE_ROUNDED),
            start_btn,
            card("LEVELS", meters, status, icon=ft.Icons.EQUALIZER_ROUNDED),
        ], spacing=16),
    )
    refresh()
    threading.Thread(target=meter_loop, daemon=True).start()


if __name__ == "__main__":
    ft.run(main)
