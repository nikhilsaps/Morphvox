<div align="center">

# 🎙️ Morphvox

### Real-time AI voice changer for calls, streams and VOIP

Turn your mic into any voice in real time, route it into Discord / Zoom / Teams / OBS,
and switch voices on the fly without stopping the stream.

![Python](https://img.shields.io/badge/python-3.10-3776AB?logo=python&logoColor=white)
![RVC](https://img.shields.io/badge/engine-RVC%20v2-00bcd4)
![GPU](https://img.shields.io/badge/CUDA-accelerated-76B900?logo=nvidia&logoColor=white)
![Flet](https://img.shields.io/badge/UI-Flet-0098FF)
![Platform](https://img.shields.io/badge/platform-Windows-0078D6?logo=windows&logoColor=white)

</div>

---

## ✨ What it does

Morphvox captures your microphone, converts your voice to a target speaker with a neural
**RVC v2** model on your GPU, and plays the result to your headphones **and/or** a virtual
microphone that any app can use as its "mic".

```
🎤 mic ──▶ context-buffer stream ──▶ 🧠 RVC v2 (GPU) ──▶ crossfade ──▶ 🎧 monitor
                                                                   └─▶ 🔊 VB-Cable ──▶ Discord / Zoom / OBS
```

## 🚀 Features

- **11 built-in voices** — anime, celebrity, cartoon, child and Indian playback singers (bring your own too).
- **Live voice hot-swap** — change the voice mid-call; the new model warms up in the background while the old voice keeps playing, then switches seamlessly. No stop/start.
- **Quality toggle** — pick your latency/quality trade-off on the fly:
  | Mode | Pitch engine | Index | Latency | Use for |
  |------|-------------|-------|---------|---------|
  | **Fast** | `pm` | off | ~270 ms | lowest latency, calls |
  | **Natural** *(default)* | `rmvpe` | off | ~330 ms | best balance |
  | **Max** | `rmvpe` | on | ~310 ms | highest fidelity, GPU-accelerated index |
- **Virtual mic routing** via [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) — works with any app that picks a microphone.
- **Live monitor + VU meters + A/B bypass** — hear yourself, watch input/output levels, compare raw vs converted instantly.
- **Noise gate** — silences model hallucinations between words.
- **Device-aware** — dropdowns by name, auto-picks your headset, refreshes when you plug/unplug.

## 🧠 How it's fast

Real-time RVC is hard because each conversion has a ~fixed overhead regardless of chunk length.
Morphvox exploits that:

- **Context buffering** — each tick converts a long window (past context + new audio) but only emits the newest slice, so you get quality context *for free* with no added latency.
- **GPU-resident retrieval index** — stock rvc-python re-reads the `.index` file and reconstructs every stored vector on *every frame*, then runs a CPU faiss kNN (~300–450 ms). Morphvox monkeypatches `faiss.read_index` to keep the index on the GPU and runs the kNN as an exact torch distance+topk, so retrieval costs **~0 ms**. "Max" quality now runs at the same latency as "Natural".
- **`pm` vs `rmvpe`** — `pm` pitch tracking is ~60 ms cheaper per frame; pick it in "Fast".
- **Crossfaded seams** — overlapping tails are blended to kill clicks between chunks.

Honest latency on an RTX 5060 laptop: **~270–330 ms** mouth-to-ear across all three quality modes (Fast–Max). Call-usable, not zero.

## 📦 Requirements

- **Windows** with an **NVIDIA GPU** (CUDA). Tested on an RTX 5060 Laptop (Blackwell / cu128).
- **Python 3.10** (RVC's `fairseq` dependency is happiest here).
- **[VB-Audio Virtual Cable](https://vb-audio.com/Cable/)** installed (for the virtual mic).
- A headset (so the monitor doesn't feed back into your mic).

## 🛠️ Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# install a CUDA build of torch matching your GPU, e.g.:
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Add voice models

Morphvox loads RVC v2 models from `rvc_models/<name>/`. Each folder needs a `model.pth`
(and optionally a `model.index` for "Max" quality). Grab models from
[HuggingFace](https://huggingface.co/models?search=rvc) or [weights.gg](https://weights.gg),
or train your own, then drop them in:

```
rvc_models/
  amiya-rvc/
    model.pth
    model.index   (optional)
```

Register the voice in `VOICES` inside `flet_app.py`:

```python
VOICES = {
    "My Voice (F)": ("my-model-folder", 12),   # (folder, pitch shift in semitones)
}
```

> The helper scripts in `tools/` can auto-fetch models from HuggingFace by search query.

## ▶️ Run

```bash
run.bat          # Flet UI (recommended)
run_tk.bat       # Tkinter fallback UI
```

Then: pick your **mic**, **monitor** (headphones) and **VB-Cable** output, hit **Start**, and
select any app's microphone as **"CABLE Output (VB-Audio Virtual Cable)"**.

## 🎭 Built-in voices

| Voice | Type | Voice | Type |
|-------|------|-------|------|
| Amiya | anime, F | Stewie | baby |
| Hatsune Miku | F | Wonder Kid | child |
| Tomori | anime, F | Neha Kakkar | Indian, F |
| Donald Trump | M | Atif Aslam | Indian, M |
| Drake | M | Shivangi | Indian, F |
| SpongeBob | cartoon, M | | |

## 🧩 Architecture

| File | Role |
|------|------|
| `flet_app.py` | Flet GUI, device routing, voice/quality selection, hot-swap |
| `audio_engine.py` | threaded context-buffer streaming engine, noise gate, crossfade |
| `engines/rvc.py` | RVC v2 converter (wraps `rvc-python`) |
| `engines/passthrough.py` | raw mic pass-through (test) |
| `gui.py` | Tkinter fallback UI |
| `run_monitor.py` | headless CLI monitor |

## 🙏 Credits

Built on [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) /
[rvc-python](https://github.com/daswer123/rvc-python), [fairseq](https://github.com/facebookresearch/fairseq) HuBERT,
[Flet](https://flet.dev) and [VB-Audio](https://vb-audio.com/Cable/).

## ⚖️ Responsible use

Voice conversion can impersonate real people. Use it for creative, consensual fun — not fraud,
harassment or deception. You are responsible for how you use the voices you load.
