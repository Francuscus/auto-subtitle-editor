# Colorvideo Subs — v0.9.5
# - Fix: replaced gr.escape_html with html.escape (Gradio 4 safe).
# - Default text color = BLUE (#1E90FF).
# - CPU-safe WhisperX; chunk-by-N-words timing.
# - SRT/ASS/VTT export + MP4 render (ffmpeg).
# - Boxed background toggle, outline, font size, layout presets.
# - Simple per-line nudge editor.

import os
import re
import json
import math
import shutil
import tempfile
import subprocess
from typing import List, Tuple
from html import escape as html_escape  # <-- fix

import gradio as gr
import torch
import whisperx


# -------------------------- Utilities --------------------------

LANG_MAP = {
    "auto": None, "auto-detect": None, "automatic": None,
    "hungarian": "hu", "magyar": "hu",
    "spanish": "es", "español": "es",
    "english": "en",
}

def normalize_lang(s: str | None):
    if not s:
        return None
    t = s.strip().lower()
    if t in LANG_MAP:
        return LANG_MAP[t]
    m = re.search(r"\b([a-z]{2,3})\b", t)
    return m.group(1) if m else None


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Accept #RRGGBB or RRGGBB and return (r,g,b)."""
    if not hex_str:
        return (255, 255, 255)
    hx = hex_str.strip()
    if hx.startswith("#"):
        hx = hx[1:]
    if len(hx) != 6:
        return (255, 255, 255)
    r = int(hx[0:2], 16)
    g = int(hx[2:4], 16)
    b = int(hx[4:6], 16)
    return (r, g, b)


def rgb_to_ass(rr: int, gg: int, bb: int, alpha_hex: str = "00") -> str:
    """ASS color format &HAABBGGRR (AA alpha; then BGR)."""
    return f"&H{alpha_hex}{bb:02X}{gg:02X}{rr:02X}"


def hex_to_ass_bgr(hex_color: str, alpha_hex: str = "00") -> str:
    r, g, b = hex_to_rgb(hex_color)
    return rgb_to_ass(r, g, b, alpha_hex=alpha_hex)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", name)


def seconds_to_timestamp(t: float) -> str:
    t = max(t, 0.0)
    h = int(t // 3600); t -= h * 3600
    m = int(t // 60);   t -= m * 60
    s = int(t)
    ms = int(round((t - s) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# -------------------------- ASR model (lazy) --------------------------

_asr_model = None

def get_asr_model():
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"
    try:
        _asr_model = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        if "compute type" in str(e).lower():
            fallback = "int16" if device == "cpu" else "float32"
            _asr_model = whisperx.load_model("small", device=device, compute_type=fallback)
        else:
            raise
    return _asr_model


# -------------------------- Transcription & chunking --------------------------

def transcribe(audio_path: str, language_code: str | None):
    model = get_asr_model()
    # FasterWhisperPipeline here does not take word_timestamps
    result = model.transcribe(audio_path, language=language_code)
    segments = result["segments"]
    full_text = " ".join(seg["text"].strip() for seg in segments)
    duration = 0.0
    if segments:
        duration = max(duration, float(segments[-1]["end"]))
    return segments, full_text, duration


def split_segments_by_n_words(segments: List[dict], words_per_chunk: int) -> List[dict]:
    if words_per_chunk <= 0:
        words_per_chunk = 5

    out = []
    for seg in segments:
        st = float(seg["start"])
        en = float(seg["end"])
        raw = seg["text"].strip()
        if not raw:
            continue

        words = re.findall(r"\S+", raw)
        if not words:
            continue

        total = len(words)
        total_dur = max(en - st, 0.01)
        per = total_dur / total

        idx = 0
        cur_start = st
        while idx < total:
            chunk_words = words[idx:idx + words_per_chunk]
            chunk_text = " ".join(chunk_words)
            chunk_dur = per * len(chunk_words)
            chunk_end = cur_start + chunk_dur

            out.append({
                "start": round(cur_start, 3),
                "end": round(chunk_end, 3),
                "text": chunk_text
            })

            cur_start = chunk_end
            idx += words_per_chunk

    out.sort(key=lambda d: (d["start"], d["end"]))
    return out


# -------------------------- Exporters --------------------------

def to_srt(chunks: List[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        start = seconds_to_timestamp(c["start"])
        end = seconds_to_timestamp(c["end"])
        text = c["text"].strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines).strip() + "\n"


def to_vtt(chunks: List[dict]) -> str:
    lines = ["WEBVTT\n"]
    for c in chunks:
        s = seconds_to_timestamp(c["start"]).replace(",", ".")
        e = seconds_to_timestamp(c["end"]).replace(",", ".")
        lines.append(f"{s} --> {e}\n{c['text'].strip()}\n")
    return "\n".join(lines).strip() + "\n"


def to_ass(chunks: List[dict],
           font_family: str,
           font_size: int,
           text_hex: str,
           outline_hex: str,
           outline_w: int,
           boxed_bg: bool,
           bg_hex: str,
           video_w: int,
           video_h: int) -> str:

    text_col = hex_to_ass_bgr(text_hex or "#1E90FF", alpha_hex="00")
    outline_col = hex_to_ass_bgr(outline_hex or "#000000", alpha_hex="00")

    if boxed_bg:
        back_col = hex_to_ass_bgr(bg_hex or "#0B0E12", alpha_hex="00")  # opaque box
        border_style = 3
    else:
        back_col = hex_to_ass_bgr("#000000", alpha_hex="FF")  # fully transparent
        border_style = 1

    margin_l = margin_r = 30
    margin_v = 30
    align = 2  # bottom-center

    header = (
f"[Script Info]\n"
f"ScriptType: v4.00+\n"
f"PlayResX: {video_w}\n"
f"PlayResY: {video_h}\n"
f"ScaledBorderAndShadow: yes\n"
f"\n"
f"[V4+ Styles]\n"
f"Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
f"Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, "
f"Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
f"Style: Default,{font_family},{font_size},{text_col},&H00FFFFFF,{outline_col},{back_col},"
f"0,0,0,0,100,100,0,0,{border_style},{max(outline_w, 0)},{max(outline_w-1, 0)},"
f"{align},{margin_l},{margin_r},{margin_v},1\n"
f"\n"
f"[Events]\n"
f"Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events = []
    for c in chunks:
        s = seconds_to_timestamp(c["start"]).replace(",", ".")
        e = seconds_to_timestamp(c["end"]).replace(",", ".")
        txt = (c["text"] or "").replace("\n", "\\N")
        events.append(f"Dialogue: 0,{s},{e},Default,,0,0,0,,{txt}")

    return header + "\n".join(events) + "\n"


def write_text_file(text: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# -------------------------- Build preview + files --------------------------

def build_from_chunks(
    chunks: List[dict],
    font_family: str,
    font_size: int,
    text_color: str,
    outline_color: str,
    outline_w: int,
    boxed_bg: bool,
    bg_color: str,
    layout_preset: str
):
    if layout_preset.startswith("9:16"):
        vw, vh = 1080, 1920
    else:
        vw, vh = 1280, 720

    style_html = f"""
<style>
.preview-wrap {{
  width: 100%;
  background: #0B0E12;
  border-radius: 10px;
  padding: 12px;
  color: #e6e6e6;
}}
.bubble {{
  display:inline-block;
  font-family: {'inherit' if font_family=='Default' else font_family}, sans-serif;
  font-size: {int(font_size)}px;
  line-height: 1.35;
  color: {text_color or '#1E90FF'};
  text-shadow:
    -{outline_w}px 0 {outline_color or '#000000'},
     {outline_w}px 0 {outline_color or '#000000'},
     0 -{outline_w}px {outline_color or '#000000'},
     0  {outline_w}px {outline_color or '#000000'};
  {"background: " + (bg_color or '#0B0E12') + "; padding: 4px 8px; border-radius: 6px;" if boxed_bg else ""}
}}
.row {{ margin: 8px 0; }}
.time {{ opacity: .5; font-size: 12px }}
</style>
"""
    lines = []
    for c in chunks:
        safe_txt = html_escape(c.get("text", ""))
        lines.append(
            f"<div class='row'><span class='time'>[{c['start']:.2f}–{c['end']:.2f}]</span> "
            f"<span class='bubble'>{safe_txt}</span></div>"
        )
    preview_html = style_html + f"<div class='preview-wrap'>{''.join(lines)}</div>"

    srt_text = to_srt(chunks)
    vtt_text = to_vtt(chunks)
    ass_text = to_ass(
        chunks,
        font_family=font_family if font_family != "Default" else "Arial",
        font_size=font_size,
        text_hex=text_color or "#1E90FF",
        outline_hex=outline_color or "#000000",
        outline_w=outline_w,
        boxed_bg=boxed_bg,
        bg_hex=bg_color or "#0B0E12",
        video_w=vw, video_h=vh
    )
    srt_path = write_text_file(srt_text, ".srt")
    vtt_path = write_text_file(vtt_text, ".vtt")
    ass_path = write_text_file(ass_text, ".ass")

    px = json.dumps({"w": vw, "h": vh})
    py = json.dumps({"n": len(chunks)})

    return preview_html, px, py, srt_path, ass_path, vtt_path


# -------------------------- Render MP4 --------------------------

def render_with_subs(
    media_path: str | None,
    ass_path: str,
    layout_preset: str,
    out_basename: str = "subtitle_video"
) -> str:
    if layout_preset.startswith("9:16"):
        vw, vh = 1080, 1920
    else:
        vw, vh = 1280, 720

    out_dir = tempfile.mkdtemp()
    out_path = os.path.join(out_dir, sanitize_filename(out_basename) + ".mp4")

    if media_path and media_path.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
        cmd = [
            "ffmpeg", "-y", "-i", media_path,
            "-vf", f"ass={ass_path}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy",
            out_path
        ]
    else:
        duration = 60
        if media_path:
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=nw=1:nk=1", media_path],
                    capture_output=True, text=True, check=True
                )
                duration = max(1, int(float(probe.stdout.strip())))
            except Exception:
                pass

        if media_path and media_path.lower().endswith((".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg")):
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"color=c=black:s={vw}x{vh}:d={duration}",
                "-i", media_path,
                "-shortest",
                "-vf", f"ass={ass_path}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                out_path
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"color=c=black:s={vw}x{vh}:d={duration}",
                "-vf", f"ass={ass_path}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                out_path
            ]

    subprocess.run(cmd, check=True)
    return out_path


# -------------------------- Gradio App --------------------------

FONT_CHOICES = [
    "Default", "Arial", "Roboto", "Open Sans", "Lato", "Noto Sans", "Montserrat",
]
LANG_CHOICES = [
    ("Auto-detect", "auto"),
    ("Hungarian (hu)", "hu"),
    ("Spanish (es)", "es"),
    ("English (en)", "en"),
]
LAYOUT_CHOICES = ["16:9 (YouTube)", "9:16 (TikTok/Reels)"]

with gr.Blocks(theme=gr.themes.Soft(), css="""
.gradio-container { max-width: 1280px !important; }
.dark .gradio-container { background: #0A0D12; }
""") as demo:
    gr.Markdown("### 🎨 Colorvideo Subs — v0.9.5")

    srt_state = gr.State("")
    ass_state = gr.State("")
    vtt_state = gr.State("")
    last_media_state = gr.State("")
    preview_ass_state = gr.State("")

    with gr.Row():
        with gr.Column(scale=3):
            media = gr.Audio(label="Audio / Video (mp3, wav, mp4…)", type="filepath")
            language = gr.Dropdown(LANG_CHOICES, value="auto", label="Language")

            words_per = gr.Slider(1, 8, value=5, step=1,
                                  label="Words per chunk (when not using lyrics)")
            layout = gr.Dropdown(LAYOUT_CHOICES, value="16:9 (YouTube)", label="Layout preset")

            lyrics_box = gr.Textbox(label="Lyrics mode (optional)",
                                    placeholder="Paste lyrics here (one line per lyric line)", lines=4)

            shift = gr.Slider(-5, 5, value=0, step=0.1, label="Global shift (seconds)")

            run_btn = gr.Button("Run", variant="primary")

            preview = gr.HTML(label="Preview")
            table = gr.JSON(label="Segments (JSON)")

            srt_dl = gr.File(label="Download SRT")
            ass_dl = gr.File(label="Download ASS")
            vtt_dl = gr.File(label="Download VTT")

            render_btn = gr.Button("Render subtitle video (MP4) 🖼️")
            rendered = gr.File(label="Rendered preview")

        with gr.Column(scale=1):
            gr.Markdown("#### Subtitle Style")
            font_family = gr.Dropdown(FONT_CHOICES, value="Default", label="Font")
            font_size   = gr.Slider(14, 72, value=36, step=1, label="Font size")
            text_color  = gr.ColorPicker(value="#1E90FF", label="Text color")  # BLUE
            outline_color = gr.ColorPicker(value="#000000", label="Outline color")
            outline_w   = gr.Slider(0, 6, value=2, step=1, label="Outline width (px)")

            gr.Markdown("#### Background")
            boxed_bg = gr.Checkbox(value=True, label="Boxed background behind text")
            bg_color = gr.ColorPicker(value="#0B0E12", label="Background color")

            gr.Markdown("#### Per-line editor")
            apply_refresh = gr.Checkbox(value=True, label="Apply edits & refresh")
            nudge_minus = gr.Button("◂ Nudge −0.10s")
            nudge_plus  = gr.Button("▸ Nudge +0.10s")

    # ---------------- Callbacks ----------------

    def run_pipeline(
        media_path, lang_ui, words, layout_preset,
        lyrics_text, shift_sec,
        font_family, font_size, text_color, outline_color, outline_w,
        boxed_bg, bg_color
    ):
        if not media_path:
            raise gr.Error("Please upload an audio or video file.")

        lang_code = normalize_lang(lang_ui)
        segs, _, _ = transcribe(media_path, lang_code)
        chunks = split_segments_by_n_words(segs, int(words))

        if abs(float(shift_sec)) > 1e-6:
            for c in chunks:
                c["start"] = round(c["start"] + float(shift_sec), 3)
                c["end"]   = round(c["end"] + float(shift_sec), 3)

        prev_html, px, py, srt_path, ass_path, vtt_path = build_from_chunks(
            chunks=chunks,
            font_family=font_family,
            font_size=int(font_size),
            text_color=text_color,
            outline_color=outline_color,
            outline_w=int(outline_w),
            boxed_bg=bool(boxed_bg),
            bg_color=bg_color,
            layout_preset=layout_preset
        )
        return (chunks, prev_html, srt_path, ass_path, vtt_path, media_path, ass_path)

    run_btn.click(
        run_pipeline,
        inputs=[
            media, language, words_per, layout, lyrics_box, shift,
            font_family, font_size, text_color, outline_color, outline_w,
            boxed_bg, bg_color
        ],
        outputs=[table, preview, srt_dl, ass_dl, vtt_dl, last_media_state, preview_ass_state]
    )

    def nudge_json(json_data, delta, do_apply):
        if not do_apply or not json_data:
            return json_data
        try:
            new = []
            for c in json_data:
                c2 = dict(c)
                c2["start"] = round(c2["start"] + delta, 3)
                c2["end"]   = round(c2["end"] + delta, 3)
                new.append(c2)
            return new
        except Exception:
            return json_data

    nudge_minus.click(lambda js: nudge_json(js, -0.10, True), inputs=[table], outputs=[table])
    nudge_plus.click(lambda js: nudge_json(js, +0.10, True), inputs=[table], outputs=[table])

    def do_render(media_path, ass_path, layout_preset):
        if not ass_path:
            raise gr.Error("No ASS file available. Click Run first.")
        try:
            out_mp4 = render_with_subs(media_path, ass_path, layout_preset, out_basename="rendered_subs")
            return out_mp4
        except subprocess.CalledProcessError as e:
            raise gr.Error(f"ffmpeg error: {e}")

    render_btn.click(
        do_render,
        inputs=[last_media_state, preview_ass_state, layout],
        outputs=[rendered]
    )

if __name__ == "__main__":
    demo.launch()
