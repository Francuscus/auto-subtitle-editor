import os
import re
import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

import gradio as gr
import torch
from faster_whisper import WhisperModel

# --------------------- Config ---------------------
TITLE = "🎨 Colorvideo Subs"
THEME = gr.themes.Soft()

# Pick a small model for CPU. Change to "base" if you want a bit more accuracy.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "tiny")

# --------------------- Language helpers ---------------------
LANG_CHOICES = [
    ("Auto-detect", None),
    ("Spanish (es)", "es"),
    ("Hungarian (hu)", "hu"),
    ("English (en)", "en"),
]

def normalize_lang(v: Optional[str]) -> Optional[str]:
    if not v or v == "None":
        return None
    t = str(v).strip().lower()
    # accept "es (Spanish)" or "Spanish (es)"
    m = re.search(r"\b([a-z]{2,3})\b", t)
    return m.group(1) if m else None

# --------------------- Load model (CPU-friendly) ---------------------
_model: Optional[WhisperModel] = None

def get_model() -> WhisperModel:
    global _model
    if _model is not None:
        return _model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    # vad_filter off = faster startup, we’ll rely on Whisper itself
    _model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=device,
        compute_type=compute
    )
    return _model

# --------------------- Types ---------------------
@dataclass
class Word:
    text: str
    start: float
    end: float

@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: List[Word]

# --------------------- Transcription ---------------------
def transcribe(audio_path: str, language_code: Optional[str]) -> Tuple[List[Segment], str]:
    model = get_model()
    # beam_size=1 for speed, word_timestamps=True to enable later word highlighting
    segments_iter, _ = model.transcribe(
        audio_path,
        language=language_code,
        beam_size=1,
        word_timestamps=True
    )
    segs: List[Segment] = []
    all_text = []
    for s in segments_iter:
        words = []
        for w in (s.words or []):
            # words might be None for very short segments
            words.append(Word(text=w.word, start=max(0.0, w.start or s.start), end=max(0.0, w.end or s.end)))
        text_clean = (s.text or "").strip()
        if not text_clean:
            continue
        segs.append(Segment(text=text_clean, start=float(s.start), end=float(s.end), words=words))
        all_text.append(text_clean)
    return segs, " ".join(all_text)

# --------------------- Simple lyrics retime ---------------------
def split_into_lines(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    lines = [ln.strip() for ln in raw.replace("\r", "").split("\n") if ln.strip()]
    # Allow a paragraph: break on sentence-ish boundaries
    if len(lines) == 1:
        para = lines[0]
        lines = [p.strip() for p in re.split(r"(?<=[\.\!\?…])\s+", para) if p.strip()]
    return lines

def retime_lyrics(lines: List[str], speech_spans: List[Tuple[float, float]]) -> List[Tuple[str, float, float]]:
    """
    Very simple alignment:
    - Take total speech duration (sum of segment durations)
    - Distribute lines evenly across that duration in order.
    Good enough for karaoke-style quick timing. You can refine later.
    """
    if not lines or not speech_spans:
        return []
    # flatten contiguous speech span into one big span
    total_start = speech_spans[0][0]
    total_end   = speech_spans[-1][1]
    total_dur   = max(0.01, total_end - total_start)
    n = len(lines)
    out = []
    for i, line in enumerate(lines):
        a = total_start + (i / n) * total_dur
        b = total_start + ((i + 1) / n) * total_dur
        out.append((line, a, b))
    return out

# --------------------- Exporters ---------------------
def to_srt(items: List[Tuple[str, float, float]]) -> str:
    def ts(x: float) -> str:
        h = int(x // 3600); m = int((x % 3600) // 60); s = int(x % 60); ms = int((x - int(x)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    lines = []
    for i, (text, a, b) in enumerate(items, 1):
        lines.append(str(i))
        lines.append(f"{ts(a)} --> {ts(b)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)

def to_ass(items: List[Tuple[str, float, float]],
           font="Arial", size=32, color="#FFFFFF", outline="#000000", outline_px=2) -> str:
    def ass_time(x: float) -> str:
        h = int(x // 3600); m = int((x % 3600) // 60); s = x % 60
        return f"{h:d}:{m:02d}:{s:05.2f}"
    def ass_rgb(hexcol: str) -> str:
        hexcol = hexcol.lstrip("#")
        r = int(hexcol[0:2],16); g = int(hexcol[2:4],16); b = int(hexcol[4:6],16)
        return f"&H{0:02X}{b:02X}{g:02X}{r:02X}"
    hdr = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font},{size},{ass_rgb(color)},{ass_rgb(color)},{ass_rgb(outline)},{ass_rgb('#000000')},"
        f"0,0,0,0,100,100,0,0,1,{outline_px},0,2,10,10,10,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    ev = []
    for text, a, b in items:
        ev.append(f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Default,,0,0,0,,{text}")
    return "\n".join(hdr + ev)

# --------------------- Gradio app ---------------------
custom_css = """
:root { --bg: #0f1220; --panel: #151a2e; --ink: #e9ecff; }
.gradio-container { max-width: 1200px !important; }
body { background: var(--bg); }
.dark .prose, .gradio-container * { color: var(--ink); }
.settings-card { position: sticky; top: 12px; background: var(--panel); border-radius: 16px; padding: 8px; }
.label, .wrap .label-wrap span { color: #c5ccff !important; }
"""

def run_pipeline(audio_path, language_ui, font_family, font_size, txt_color, outline_color, outline_w,
                 use_lyrics, lyrics_text):
    # 1) Transcribe
    lang_code = normalize_lang(language_ui)
    segments, full_text = transcribe(audio_path, lang_code)

    # 2) Build speech spans for optional retime
    speech_spans = [(s.start, s.end) for s in segments]
    preview_html = f"""
<div style="font-family:{'inherit' if font_family=='Default' else font_family}, sans-serif;
            font-size:{int(font_size)}px; line-height:1.35; color:{txt_color};
            text-shadow:
              -{outline_w}px 0 {outline_color},
               {outline_w}px 0 {outline_color},
               0 -{outline_w}px {outline_color},
               0 {outline_w}px {outline_color};">
    {full_text}
</div>"""

    # 3) Decide what to export
    if use_lyrics and (lyrics := split_into_lines(lyrics_text)):
        items = retime_lyrics(lyrics, speech_spans)
    else:
        # export the detected segments as-is
        items = [(s.text, s.start, s.end) for s in segments]

    srt_txt = to_srt(items)
    ass_txt = to_ass(items, font=font_family if font_family != "Default" else "Arial",
                     size=int(font_size), color=txt_color, outline=outline_color, outline_px=int(outline_w))
    return preview_html, [s.__dict__ for s in segments], srt_txt, ass_txt

with gr.Blocks(theme=THEME, css=custom_css, fill_height=True) as demo:
    gr.Markdown(f"# {TITLE}\nSmall, fast CPU subtitles with style panel →")

    with gr.Row():
        with gr.Column(scale=3):
            audio = gr.Audio(label="Audio / Video (mp3, wav, mp4)", type="filepath")
            language = gr.Dropdown(LANG_CHOICES, value=None, label="Language")
            run = gr.Button("Run", variant="primary")

            preview = gr.HTML(label="Preview")
            segs = gr.JSON(label="Segments")
            with gr.Tab("Export"):
                srt_box = gr.Code(label="SRT", language="text")
                ass_box = gr.Code(label="ASS", language="text")
                gr.Markdown("Copy, or **right-click → Save as…** to download.")

        with gr.Column(scale=1):
            with gr.Group(elem_classes=["settings-card"]):
                gr.Markdown("### Subtitle Settings")
                font_family = gr.Dropdown(["Default","Arial","Roboto","Open Sans","Lato","Montserrat","Noto Sans"],
                                          value="Default", label="Font")
                font_size = gr.Slider(14, 72, value=34, step=1, label="Font size")
                txt_color = gr.ColorPicker(value="#FFFFFF", label="Text color")
                outline_color = gr.ColorPicker(value="#000000", label="Outline color")
                outline_w = gr.Slider(0, 4, value=2, step=1, label="Outline width (px)")

            with gr.Group(elem_classes=["settings-card"]):
                gr.Markdown("### Optional lyrics (retime)")
                use_lyrics = gr.Checkbox(label="Use pasted lyrics to retime")
                lyrics_text = gr.Textbox(label="Paste lyrics or lines here",
                                         placeholder="One line per subtitle… or paste a paragraph.",
                                         lines=10)

    run.click(
        run_pipeline,
        inputs=[audio, language, font_family, font_size, txt_color, outline_color, outline_w, use_lyrics, lyrics_text],
        outputs=[preview, segs, srt_box, ass_box]
    )

if __name__ == "__main__":
    demo.launch()
