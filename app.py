# app.py — Five-Word Subtitle Chunker + Lyrics Timecoding + Global Shift + WebVTT + Per-line Editor
# Version: v1.2.0 (2025-10-25)

import os
import re
import json
import uuid
import math
import tempfile
from typing import List, Dict, Tuple, Optional

import gradio as gr
import torch
import whisperx

# ---------- Language helpers ----------
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
    m = re.search(r"\b([a-z]{2,3})\b", t)  # accept "es (Spanish)" or "Spanish (es)"
    if m:
        return m.group(1)
    return None

# ---------- ASR model (lazy load) ----------
_ASR_MODEL = None

def get_asr_model():
    """Load whisperx once, with safe compute_type for CPU/GPU."""
    global _ASR_MODEL
    if _ASR_MODEL is not None:
        return _ASSR_MODEL  # intentional: typo caught below

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

# Fix the small typo above so we don't crash:
def get_asr_model():
    global _ASR_MODEL
    if _ASR_MODEL is not None:
        return _ASR_MODEL
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"
    try:
        _ASR_MODEL = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        if "compute type" in str(e).lower():
            fallback = "int16" if device == "cpu" else "float32"
            _ASR_MODEL = whisperx.load_model("small", device=device, compute_type=fallback)
        else:
            raise
    return _ASR_MODEL

# ---------- Transcription ----------
def transcribe(audio_path: str, language_code: Optional[str]) -> Tuple[List[Dict], str, float, float]:
    """
    Run ASR and return (segments, full_text, t_min, t_max).
    We do NOT request word timestamps (keeps it fast & compatible).
    """
    model = get_asr_model()
    result = model.transcribe(audio_path, language=language_code)
    segments = result["segments"]
    full_text = " ".join((seg.get("text") or "").strip() for seg in segments).strip()
    if segments:
        t_min = float(segments[0].get("start", 0.0) or 0.0)
        t_max = float(segments[-1].get("end", 0.0) or t_min)
    else:
        t_min, t_max = 0.0, 0.0
    return segments, full_text, t_min, t_max

# ---------- Chunking (5-word by default) ----------
def split_segment_into_word_chunks(text: str, start: float, end: float, max_words: int) -> List[Dict]:
    tokens = [t for t in text.strip().split() if t]
    if not tokens:
        return []
    duration = max(0.0, (end or 0.0) - (start or 0.0))
    per_token = duration / max(len(tokens), 1)

    chunks = []
    i = 0
    while i < len(tokens):
        group = tokens[i : i + max_words]
        n = len(group)
        c_start = start + i * per_token
        c_end = start + (i + n) * per_token
        chunks.append({"start": round(c_start, 3), "end": round(c_end, 3), "text": " ".join(group)})
        i += max_words
    return chunks

def make_five_word_chunks(segments: List[Dict], max_words: int) -> List[Dict]:
    out = []
    for seg in segments:
        seg_text = str(seg.get("text", "")).strip()
        s = float(seg.get("start", 0.0) or 0.0)
        e = float(seg.get("end", 0.0) or s)
        out.extend(split_segment_into_word_chunks(seg_text, s, e, max_words))
    return out

# ---------- Lyrics alignment ----------
def parse_lyrics(raw: str) -> List[str]:
    # one line per lyric line; drop empties
    lines = [ln.strip() for ln in raw.replace("\r\n", "\n").split("\n")]
    return [ln for ln in lines if ln]

def word_count(s: str) -> int:
    return len([t for t in s.split() if t])

def align_lyrics_to_timeline(
    lyrics_lines: List[str], t_min: float, t_max: float
) -> List[Dict]:
    """
    Assign each lyric line a start/end inside [t_min, t_max] proportionally to its word count.
    """
    chunks: List[Dict] = []
    total_words = sum(max(1, word_count(ln)) for ln in lyrics_lines) or 1
    total_duration = max(0.0, t_max - t_min)
    cursor = t_min
    for ln in lyrics_lines:
        wc = max(1, word_count(ln))
        dur = total_duration * (wc / total_words)
        start = cursor
        end = cursor + dur
        chunks.append({"start": round(start, 3), "end": round(end, 3), "text": ln})
        cursor = end
    # ensure end equals t_max
    if chunks:
        chunks[-1]["end"] = round(t_max, 3)
    return chunks

# ---------- Global timing shift ----------
def apply_global_shift(chunks: List[Dict], shift_sec: float) -> List[Dict]:
    if not shift_sec:
        return chunks
    out = []
    for c in chunks:
        out.append({
            "start": round(max(0.0, c["start"] + shift_sec), 3),
            "end":   round(max(0.0, c["end"]   + shift_sec), 3),
            "text":  c["text"],
        })
    return out

# ---------- Formatting: SRT / ASS / VTT ----------
def fmt_srt_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - math.floor(t)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def to_srt(chunks: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(str(i))
        lines.append(f"{fmt_srt_time(c['start'])} --> {fmt_srt_time(c['end'])}")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines)

def hex_to_ass_bgr(hex_color: str) -> str:
    hx = hex_color.lstrip("#")
    if len(hx) == 3:
        hx = "".join(ch * 2 for ch in hx)
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
    text_col = hex_to_ass_bgr(text_hex)
    outline_col = hex_to_ass_bgr(outline_hex)
    bg_col = hex_to_ass_bgr(bg_hex)

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
        events.append(f"Dialogue: 0,{ass_time(c['start'])},{ass_time(c['end'])},Default,,0,0,0,,{txt}")

    return "\n".join(header + events)

def fmt_vtt_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - math.floor(t)) * 1000))
    if h > 0:
        return f"{h:02}:{m:02}:{s:02}.{ms:03}"
    else:
        return f"{m:02}:{s:02}.{ms:03}"

def to_vtt(chunks: List[Dict]) -> str:
    lines = ["WEBVTT", ""]
    for c in chunks:
        lines.append(f"{fmt_vtt_time(c['start'])} --> {fmt_vtt_time(c['end'])}")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines)

# ---------- Preview HTML ----------
def compute_canvas_size(preset: str) -> Tuple[int, int]:
    if preset == "9:16 (phone/tiktok)":
        return (1080, 1920)
    return (1920, 1080)  # 16:9 default

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
    playres_x, playres_y = compute_canvas_size(layout_preset)
    box_bg = bg_hex if bg_box else "transparent"
    outline_css = (
        f" -{outline_w}px 0 {outline_hex}, {outline_w}px 0 {outline_hex},"
        f" 0 -{outline_w}px {outline_hex}, 0 {outline_w}px {outline_hex}"
        if outline_w > 0
        else " none"
    )
    joined = " ".join(c.get("text", "") for c in chunks).strip() or "<i>(no text)</i>"

    return f"""
<div style="background:#0b1020;color:#e8f1ff;padding:16px;border-radius:16px;">
  <div style="
    width:100%;
    max-width:1000px;
    aspect-ratio:{playres_x}/{playres_y};
    margin:0 auto;
    border-radius:16px;
    background:#11151f;
    position:relative;
    box-shadow:0 8px 30px rgba(0,0,0,.35);
    overflow:hidden;
  ">
    <div style="position:absolute;left:0;right:0;bottom:6%;display:flex;justify-content:center;">
      <div style="
        font-family:{'inherit' if font=='Default' else font}, system-ui, sans-serif;
        font-size:{int(size)}px;
        line-height:1.35;
        color:{text_hex};
        text-align:center;
        padding:8px 14px;
        background:{box_bg};
        border-radius:12px;
        text-shadow:{outline_css};
        max-width:90%;
      ">{joined}</div>
    </div>
  </div>
</div>
"""

# ---------- Helper: table <-> chunks ----------
def chunks_to_table(chunks: List[Dict]) -> List[List]:
    rows = []
    for c in chunks:
        rows.append([round(float(c.get("start", 0.0)), 3),
                     round(float(c.get("end", 0.0)), 3),
                     str(c.get("text", ""))])
    return rows

def table_to_chunks(table: List[List]) -> List[Dict]:
    out = []
    for row in (table or []):
        if not row or (len(row) < 3):
            continue
        try:
            s = max(0.0, float(row[0]))
            e = max(0.0, float(row[1]))
        except Exception:
            continue
        txt = str(row[2] or "")
        if txt.strip() == "":
            continue
        if e < s:
            e = s
        out.append({"start": round(s, 3), "end": round(e, 3), "text": txt})
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out

# ---------- Build preview + files from chunks ----------
def compute_canvas_size(preset: str) -> Tuple[int, int]:
    if preset == "9:16 (phone/tiktok)":
        return (1080, 1920)
    return (1920, 1080)

def build_from_chunks(
    chunks: List[Dict],
    layout_preset: str,
    font: str,
    size: int,
    text_hex: str,
    outline_hex: str,
    outline_w: int,
    bg_box: bool,
    bg_hex: str,
):
    prev_html = preview_html(
        chunks, layout_preset, font, int(size), text_hex, outline_hex, int(outline_w), bool(bg_box), bg_hex
    )
    playres_x, playres_y = compute_canvas_size(layout_preset)

    srt_text = to_srt(chunks)
    ass_text = to_ass(
        chunks,
        playres_x,
        playres_y,
        font if font != "Default" else "Arial",
        int(size),
        text_hex,
        outline_hex,
        int(outline_w),
        bool(bg_box),
        bg_hex,
    )
    vtt_text = to_vtt(chunks)

    srt_path = os.path.join(tempfile.gettempdir(), f"subs_{uuid.uuid4().hex}.srt")
    ass_path = os.path.join(tempfile.gettempdir(), f"subs_{uuid.uuid4().hex}.ass")
    vtt_path = os.path.join(tempfile.gettempdir(), f"subs_{uuid.uuid4().hex}.vtt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_text)
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_text)
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write(vtt_text)

    return prev_html, playres_x, playres_y, srt_path, ass_path, vtt_path

# ---------- Main pipeline ----------
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
    use_lyrics,
    lyrics_text,
    global_shift,
):
    if not audio_path:
        raise gr.Error("Please upload an audio or video file (wav/mp3/mp4, etc.).")

    lang_code = normalize_lang(language_label)
    segments, _, t_min, t_max = transcribe(audio_path, lang_code)

    if use_lyrics and (lyrics_text or "").strip():
        lines = parse_lyrics(lyrics_text)
        chunks = align_lyrics_to_timeline(lines, t_min, t_max)
    else:
        chunks = make_five_word_chunks(segments, int(words_per_chunk))

    # apply global shift (can be negative or positive)
    chunks = apply_global_shift(chunks, float(global_shift))

    table = chunks_to_table(chunks)
    prev_html, px, py, srt_path, ass_path, vtt_path = build_from_chunks(
        chunks, layout_preset, font, int(size), text_hex, outline_hex, int(outline_w), bool(bg_box), bg_hex
    )
    return chunks, table, prev_html, px, py, srt_path, ass_path, vtt_path

# ---------- UI ----------
FONT_CHOICES = ["Default", "Arial", "Roboto", "Open Sans", "Lato", "Noto Sans", "Montserrat"]
LANG_CHOICES = [("Auto-detect", "auto"), ("Hungarian (hu)", "hungarian"), ("Spanish (es)", "spanish")]
LAYOUT_CHOICES = ["16:9 (YouTube)", "9:16 (phone/tiktok)"]

custom_css = """
.gradio-container { max-width: 1180px !important; }
body { background: #070b16; }
h1, h2, h3, .prose h1, .prose h2, .prose h3 { color: #cfe2ff !important; }
.prose p, label, .gradio-container { color: #c3d1ea; }
.settings-card { position: sticky; top: 12px; border-radius: 16px; }
#header_version { color:#85b6ff; font-weight:600; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("### <span id='header_version'>Build: v1.2.0</span>")
    gr.Markdown("# 🎬 Chunker + Lyrics + Global Shift + Editor  \nFast ASR → edit timing → SRT/ASS/VTT")

    # keep a state with current chunks
    chunks_state = gr.State([])  # List[Dict]

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group():
                audio = gr.Audio(label="Audio / Video", type="filepath")
                language = gr.Dropdown(choices=[c[0] for c in LANG_CHOICES], value="Auto-detect", label="Language")
                words_per_chunk = gr.Slider(3, 8, value=5, step=1, label="Words per chunk (when not using lyrics)")
                layout_choice = gr.Dropdown(LAYOUT_CHOICES, value=LAYOUT_CHOICES[0], label="Layout preset")

                with gr.Accordion("Lyrics mode (optional)", open=False):
                    use_lyrics = gr.Checkbox(value=False, label="Use pasted lyrics instead of auto chunks")
                    lyrics_box = gr.Textbox(
                        label="Paste lyrics (one line per lyric line)",
                        placeholder="line 1\nline 2\nline 3 ...",
                        lines=10
                    )

                with gr.Accordion("Timing tools", open=False):
                    global_shift = gr.Slider(-5.0, 5.0, value=0.0, step=0.1, label="Global shift (seconds)")

                run_btn = gr.Button("Run", variant="primary")

            preview_html_box = gr.HTML(label="Preview")
            chunks_json = gr.JSON(label="Chunks (debug / export)")
            playres_x_box = gr.Number(label="PlayResX", interactive=False)
            playres_y_box = gr.Number(label="PlayResY", interactive=False)

            with gr.Row():
                srt_dl = gr.DownloadButton("Download SRT")
                ass_dl = gr.DownloadButton("Download ASS")
                vtt_dl = gr.DownloadButton("Download VTT")

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

            with gr.Group():
                gr.Markdown("### Per-line editor")
                table = gr.Dataframe(
                    headers=["start", "end", "text"],
                    datatype=["number", "number", "str"],
                    row_count=(1, "dynamic"),
                    wrap=True,
                    interactive=True,
                    label="Edit times/text here, then click Apply edits"
                )
                with gr.Row():
                    apply_edits_btn = gr.Button("✅ Apply edits & refresh")
                    sort_btn = gr.Button("Sort by start")
                with gr.Row():
                    nudge_earlier_btn = gr.Button("◀︎ Nudge −0.10s")
                    nudge_later_btn = gr.Button("▶︎ Nudge +0.10s")

    # --- Handlers ---

    def _run_and_return(*args):
        chunks, table_data, prev_html, px, py, srt_path, ass_path, vtt_path = run_pipeline(*args)
        return (
            chunks,                   # state
            prev_html, chunks, px, py,
            srt_path, ass_path, vtt_path,
            table_data
        )

    run_btn.click(
        _run_and_return,
        inputs=[
            audio, language, words_per_chunk,
            layout_choice, font_family, font_size,
            text_color, outline_color, outline_w,
            bg_box, bg_color, use_lyrics, lyrics_box, global_shift
        ],
        outputs=[
            chunks_state,            # update state
            preview_html_box, chunks_json, playres_x_box, playres_y_box,
            srt_dl, ass_dl, vtt_dl,
            table
        ],
    )

    # Apply edits from table to state + rebuild preview/files
    def _apply_edits(table_data, layout_preset, font, size, text_hex, outline_hex, outline_w, bg_box, bg_hex, current_chunks):
        new_chunks = table_to_chunks(table_data)
        if not new_chunks:
            # keep current if user cleared table accidentally
            new_chunks = current_chunks or []
        prev_html, px, py, srt_path, ass_path, vtt_path = build_from_chunks(
            new_chunks, layout_preset, font, int(size), text_hex, outline_hex, int(outline_w), bool(bg_box), bg_hex
        )
        return (
            new_chunks,  # state
            prev_html, new_chunks, px, py,
            srt_path, ass_path, vtt_path,
            chunks_to_table(new_chunks)
        )

    apply_edits_btn.click(
        _apply_edits,
        inputs=[table, layout_choice, font_family, font_size, text_color, outline_color, outline_w, bg_box, bg_color, chunks_state],
        outputs=[chunks_state, preview_html_box, chunks_json, playres_x_box, playres_y_box, srt_dl, ass_dl, vtt_dl, table],
    )

    # Sort by start
    def _sort_chunks(current_chunks, layout_preset, font, size, text_hex, outline_hex, outline_w, bg_box, bg_hex):
        new_chunks = sorted(current_chunks or [], key=lambda x: (x["start"], x["end"]))
        prev_html, px, py, srt_path, ass_path, vtt_path = build_from_chunks(
            new_chunks, layout_preset, font, int(size), text_hex, outline_hex, int(outline_w), bool(bg_box), bg_hex
        )
        return (
            new_chunks,
            prev_html, new_chunks, px, py,
            srt_path, ass_path, vtt_path,
            chunks_to_table(new_chunks)
        )

    sort_btn.click(
        _sort_chunks,
        inputs=[chunks_state, layout_choice, font_family, font_size, text_color, outline_color, outline_w, bg_box, bg_color],
        outputs=[chunks_state, preview_html_box, chunks_json, playres_x_box, playres_y_box, srt_dl, ass_dl, vtt_dl, table],
    )

    # Quick nudges: ±0.10s
    def _nudge(current_chunks, delta, layout_preset, font, size, text_hex, outline_hex, outline_w, bg_box, bg_hex):
        new_chunks = apply_global_shift(current_chunks or [], float(delta))
        prev_html, px, py, srt_path, ass_path, vtt_path = build_from_chunks(
            new_chunks, layout_preset, font, int(size), text_hex, outline_hex, int(outline_w), bool(bg_box), bg_hex
        )
        return (
            new_chunks,
            prev_html, new_chunks, px, py,
            srt_path, ass_path, vtt_path,
            chunks_to_table(new_chunks)
        )

    nudge_earlier_btn.click(
        _nudge,
        inputs=[chunks_state, gr.State(-0.10), layout_choice, font_family, font_size, text_color, outline_color, outline_w, bg_box, bg_color],
        outputs=[chunks_state, preview_html_box, chunks_json, playres_x_box, playres_y_box, srt_dl, ass_dl, vtt_dl, table],
    )
    nudge_later_btn.click(
        _nudge,
        inputs=[chunks_state, gr.State(+0.10), layout_choice, font_family, font_size, text_color, outline_color, outline_w, bg_box, bg_color],
        outputs=[chunks_state, preview_html_box, chunks_json, playres_x_box, playres_y_box, srt_dl, ass_dl, vtt_dl, table],
    )

if __name__ == "__main__":
    demo.launch()
