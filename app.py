# app.py  —  Five-Word Subtitle Chunker + Exports
# Version: v0.9.3 (2025-10-25)

import os
import re
import json
import uuid
import math
import tempfile
from typing import List, Dict, Tuple, Optional

import gradio as gr
import torch

# ---------- ASR (whisperx) ----------
import whisperx

# Map friendly names → whisper language codes
LANG_MAP = {
    "auto": None,
    "hungarian": "hu",
    "spanish": "es",
}

def normalize_lang(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    t = label.strip().lower()
    if t in LANG_MAP:
        return LANG_MAP[t]
    # accept "hu (Hungarian)" or "Hungarian (hu)" or plain "hu"
    m = re.search(r"\b([a-z]{2,3})\b", t)
    if m:
        return m.group(1)
    return None

_ASR_MODEL = None

def get_asr_model():
    """Load whisperx once, choosing safe compute_type for CPU/GPU."""
    global _ASR_MODEL
    if _ASR_MODEL is not None:
        return _ASR_MODEL

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"  # CPU: int8 is fast & safe

    try:
        _ASR_MODEL = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        # Some CPUs can’t do int8 kernels; fall back automatically.
        if "compute type" in str(e).lower():
            fallback = "int16" if device == "cpu" else "float32"
            _ASR_MODEL = whisperx.load_model("small", device=device, compute_type=fallback)
        else:
            raise
    return _ASR_MODEL

# ---------- Chunking ----------
def split_segment_into_word_chunks(
    text: str, start: float, end: float, max_words: int
) -> List[Dict]:
    """
    Split one segment (with start/end) into sub-segments of up to max_words,
    distributing time proportionally by token count (simple & robust).
    """
    # naive tokenization by whitespace; keeps punctuation attached (good enough for timing)
    tokens = [t for t in text.strip().split() if t]
    if not tokens:
        return []

    duration = max(0.0, (end or 0.0) - (start or 0.0))
    per_token = duration / max(len(tokens), 1)

    chunks = []
    idx = 0
    while idx < len(tokens):
        group = tokens[idx : idx + max_words]
        n = len(group)
        c_start = start + idx * per_token
        c_end = start + (idx + n) * per_token
        chunks.append(
            {
                "start": round(c_start, 3),
                "end": round(c_end, 3),
                "text": " ".join(group),
            }
        )
        idx += max_words
    return chunks


def make_five_word_chunks(segments: List[Dict], max_words: int) -> List[Dict]:
    """Apply 5-word (or N-word) chunking across the whole transcript."""
    out = []
    for seg in segments:
        seg_text = str(seg.get("text", "")).strip()
        s = float(seg.get("start", 0.0) or 0.0)
        e = float(seg.get("end", 0.0) or s)
        out.extend(split_segment_into_word_chunks(seg_text, s, e, max_words))
    return out


# ---------- Transcription ----------
def transcribe(audio_path: str, language_code: Optional[str]) -> Tuple[List[Dict], str]:
    """
    Run ASR and return (segments, full_text).
    We do NOT request word timestamps (keeps it fast & compatible).
    """
    model = get_asr_model()
    result = model.transcribe(audio_path, language=language_code)
    segments = result["segments"]
    # Join raw text
    full_text = " ".join((seg.get("text") or "").strip() for seg in segments).strip()
    return segments, full_text


# ---------- Formatting: SRT / ASS ----------
def to_srt(chunks: List[Dict]) -> str:
    def fmt_time(t: float) -> str:
        t = max(0.0, t)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int(round((t - math.floor(t)) * 1000))
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(str(i))
        lines.append(f"{fmt_time(c['start'])} --> {fmt_time(c['end'])}")
        lines.append(c["text"])
        lines.append("")  # blank line
    return "\n".join(lines)


def hex_to_ass_bgr(hex_color: str) -> str:
    """
    Convert #RRGGBB → &HBBGGRR& (ASS BGR)
    """
    hx = hex_color.lstrip("#")
    if len(hx) == 3:
        hx = "".join(ch*2 for ch in hx)
    r = int(hx[0:2], 16)
    g = int(hx[2:4], 16)
    b = int(hx[4:6], 16)
    return f"&H{b:02X}{g:02X}{r:02X}&"


def to_ass(
    chunks: List[Dict],
    playres_x: int,
    playres_y: int,
    font: str,
    size: int,
    text_hex: str,
    outline_hex: str,
    outline_w: int,
    bg_box: bool,
    bg_hex: str,
    vertical_margin: int = 40,
) -> str:
    """
    Build a simple ASS with a single style.
    We approximate a "boxed background" by using a large border in the background color.
    """
    text_col = hex_to_ass_bgr(text_hex)
    outline_col = hex_to_ass_bgr(outline_hex)
    bg_col = hex_to_ass_bgr(bg_hex)

    # If bg box is on, we use the outline as a "fill" by setting border color to bg.
    style_outline_color = bg_col if bg_box else outline_col
    style_bord = max(0, outline_w if not bg_box else max(outline_w, 6))

    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {playres_x}",
        f"PlayResY: {playres_y}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font},{size},{text_col},&H00FFFFFF,{style_outline_color},&H7F000000,"
        f"0,0,0,0,100,100,0,0,1,{style_bord},0,2,30,30,{vertical_margin},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    def ass_time(t: float) -> str:
        t = max(0.0, t)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        cs = int(round((t - math.floor(t)) * 100))  # centiseconds
        return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

    events = []
    for c in chunks:
        txt = (c["text"] or "").replace("\n", "\\N")
        events.append(
            f"Dialogue: 0,{ass_time(c['start'])},{ass_time(c['end'])},Default,,0,0,0,,{txt}"
        )

    return "\n".join(header + events)


# ---------- Preview HTML ----------
def compute_canvas_size(preset: str) -> Tuple[int, int]:
    if preset == "9:16 (phone/tiktok)":
        return (1080, 1920)
    # default 16:9
    return (1920, 1080)

def preview_html(
    chunks: List[Dict],
    layout_preset: str,
    font: str,
    size: int,
    text_hex: str,
    outline_hex: str,
    outline_w: int,
    bg_box: bool,
    bg_hex: str,
) -> str:
    """
    Simple visual preview: shows all text in one frame with chosen style.
    (Not a time-playback preview — keeps frontend simple and fast.)
    """
    playres_x, playres_y = compute_canvas_size(layout_preset)

    # CSS for a center-bottom caption block
    box_bg = bg_hex if bg_box else "transparent"
    outline_css = (
        f" -{outline_w}px 0 {outline_hex}, {outline_w}px 0 {outline_hex},"
        f" 0 -{outline_w}px {outline_hex}, 0 {outline_w}px {outline_hex}"
        if outline_w > 0 else " none"
    )

    joined = " ".join(c.get("text", "") for c in chunks).strip()
    if not joined:
        joined = "<i>(no text)</i>"

    return f"""
<div style="
  background:#0b1020;
  color:#e8f1ff;
  padding:16px;
  border-radius:16px;
">
  <div style="
    width: 100%;
    max-width: 1000px;
    aspect-ratio: {playres_x} / {playres_y};
    margin: 0 auto;
    border-radius: 16px;
    background: #11151f;
    position: relative;
    box-shadow: 0 8px 30px rgba(0,0,0,.35);
    overflow: hidden;
  ">
    <div style="
      position:absolute; left:0; right:0; bottom:6%;
      display:flex; justify-content:center;
    ">
      <div style="
        font-family: {'inherit' if font=='Default' else font}, system-ui, sans-serif;
        font-size: {int(size)}px;
        line-height: 1.35;
        color: {text_hex};
        text-align:center;
        padding: 8px 14px;
        background: {box_bg};
        border-radius: 12px;
        text-shadow:{outline_css};
        max-width: 90%;
      ">
        {joined}
      </div>
    </div>
  </div>
</div>
"""


# ---------- Pipeline (UI callback) ----------
def run_pipeline(
    audio_path,
    language_label,
    words_per_chunk,
    layout_preset,
    font,
    size,
    text_hex,
    outline_hex,
    outline_w,
    bg_box,
    bg_hex,
):
    if not audio_path:
        raise gr.Error("Please upload an audio or video file (wav/mp3/mp4, etc.).")

    lang_code = normalize_lang(language_label)
    segments, _ = transcribe(audio_path, lang_code)
    chunks = make_five_word_chunks(segments, int(words_per_chunk))

    # preview & canvas dims
    prev_html = preview_html(
        chunks, layout_preset, font, int(size), text_hex, outline_hex, int(outline_w), bool(bg_box), bg_hex
    )
    playres_x, playres_y = compute_canvas_size(layout_preset)

    # write SRT + ASS to temp files for downloads
    srt_text = to_srt(chunks)
    ass_text = to_ass(
        chunks, playres_x, playres_y, font if font != "Default" else "Arial",
        int(size), text_hex, outline_hex, int(outline_w), bool(bg_box), bg_hex
    )

    srt_path = os.path.join(tempfile.gettempdir(), f"subs_{uuid.uuid4().hex}.srt")
    ass_path = os.path.join(tempfile.gettempdir(), f"subs_{uuid.uuid4().hex}.ass")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_text)
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_text)

    return prev_html, chunks, playres_x, playres_y, srt_path, ass_path


# ---------- UI ----------
FONT_CHOICES = [
    "Default", "Arial", "Roboto", "Open Sans", "Lato", "Noto Sans", "Montserrat",
]
LANG_CHOICES = [
    ("Auto-detect", "auto"),
    ("Hungarian (hu)", "hungarian"),
    ("Spanish (es)", "spanish"),
]
LAYOUT_CHOICES = ["16:9 (YouTube)", "9:16 (phone/tiktok)"]

custom_css = """
/* dark canvas + readable titles */
.gradio-container { max-width: 1180px !important; }
body { background: #070b16; }
h1, h2, h3, .prose h1, .prose h2, .prose h3 { color: #cfe2ff !important; }
.prose p, label, .gradio-container { color: #c3d1ea; }
.settings-card { position: sticky; top: 12px; border-radius: 16px; }
#header_version { color:#85b6ff; font-weight:600; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("### <span id='header_version'>Build: v0.9.3</span>")
    gr.Markdown("# 🎬 Five-Word Subtitle Chunker  \nFast ASR → clean chunks → SRT/ASS export")

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group():
                audio = gr.Audio(label="Audio / Video", type="filepath")
                language = gr.Dropdown(
                    choices=[c[0] for c in LANG_CHOICES],
                    value="Auto-detect",
                    label="Language"
                )
                words_per_chunk = gr.Slider(3, 8, value=5, step=1, label="Words per chunk")
                layout_choice = gr.Dropdown(LAYOUT_CHOICES, value=LAYOUT_CHOICES[0], label="Layout preset")

                run_btn = gr.Button("Run", variant="primary")

            preview_html_box = gr.HTML(label="Preview")
            chunks_json = gr.JSON(label="Chunks (debug / export)")
            playres_x_box = gr.Number(label="PlayResX", interactive=False)
            playres_y_box = gr.Number(label="PlayResY", interactive=False)

            with gr.Row():
                srt_dl = gr.DownloadButton("Download SRT")
                ass_dl = gr.DownloadButton("Download ASS")

        with gr.Column(scale=1):
            with gr.Group(elem_classes=["settings-card"]):
                gr.Markdown("### Subtitle Style")
                font_family   = gr.Dropdown(FONT_CHOICES, value="Default", label="Font")
                font_size     = gr.Slider(14, 72, value=36, step=1, label="Font size")
                text_color    = gr.ColorPicker(value="#FFFFFF", label="Text color")
                outline_color = gr.ColorPicker(value="#000000", label="Outline color")
                outline_w     = gr.Slider(0, 6, value=2, step=1, label="Outline width (px)")

                gr.Markdown("---")
                gr.Markdown("### Background")
                bg_box   = gr.Checkbox(value=True, label="Boxed background behind text")
                bg_color = gr.ColorPicker(value="#111111", label="Background color")

    def _run_and_return(*args):
        prev_html, chunks, px, py, srt_path, ass_path = run_pipeline(*args)
        # For Gradio 4.44, provide file paths as return values to DownloadButton
        return prev_html, chunks, px, py, srt_path, ass_path

    run_btn.click(
        _run_and_return,
        inputs=[
            audio, language, words_per_chunk,
            layout_choice, font_family, font_size,
            text_color, outline_color, outline_w,
            bg_box, bg_color
        ],
        outputs=[
            preview_html_box, chunks_json, playres_x_box, playres_y_box,
            srt_dl, ass_dl
        ],
    )

if __name__ == "__main__":
    demo.launch()
