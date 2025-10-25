# app.py — Colorvideo Subs v1.1
# - Fixes State wiring and render button path
# - Robust color parsing (#hex or rgb/rgba)
# - Text color now affects preview and ASS export

import os, re, json, subprocess, tempfile, math
from typing import List, Dict, Tuple, Optional

import gradio as gr
import torch
import whisperx

# ----------------------------
# Config / constants
# ----------------------------
WHISPER_MODEL_SIZE = "small"
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"
COMPUTE_TYPE = "float16" if USE_CUDA else "int8"

# Supported languages menu (kept short as you asked)
LANG_CHOICES = [
    ("Auto-detect", "auto"),
    ("Spanish (es)", "es"),
    ("Hungarian (hu)", "hu"),
    ("English (en)", "en"),
]

LAYOUT_CHOICES = [
    "16:9 (YouTube)",
    "9:16 (TikTok)",
    "1:1 (Square)"
]

# ----------------------------
# Helpers: language + model
# ----------------------------

def normalize_lang(lang_ui: Optional[str]) -> Optional[str]:
    if not lang_ui or lang_ui == "auto":
        return None
    t = lang_ui.strip().lower()
    # accept 'es', 'hu', 'en' directly
    if t in {"es","hu","en"}:
        return t
    # accept "spanish (es)" etc.
    m = re.search(r"\b([a-z]{2,3})\b", t)
    return m.group(1) if m else None

_asr_model = None
def get_asr():
    global _asr_model
    if _asr_model is not None:
        return _asr_model
    try:
        _asr_model = whisperx.load_model(
            WHISPER_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE
        )
    except ValueError as e:
        # CPU with no int8 kernels fallback
        if "compute type" in str(e).lower():
            fallback = "int16" if DEVICE == "cpu" else "float32"
            _asr_model = whisperx.load_model(
                WHISPER_MODEL_SIZE, device=DEVICE, compute_type=fallback
            )
        else:
            raise
    return _asr_model

# ----------------------------
# Helpers: colors
# ----------------------------

def parse_color(s: str) -> Tuple[int,int,int]:
    """
    Accept '#rrggbb', '#rgb', 'rgb(r,g,b)', 'rgba(r,g,b,a)' and return (r,g,b).
    """
    if not s:
        return (255,255,255)
    s = s.strip()
    if s.startswith("#"):
        hx = s[1:]
        if len(hx) == 3:
            r = int(hx[0]*2, 16); g = int(hx[1]*2, 16); b = int(hx[2]*2, 16)
            return (r,g,b)
        if len(hx) >= 6:
            return (int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16))
    m = re.match(r"rgba?\(([^)]+)\)", s, flags=re.I)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        r = int(float(parts[0])); g = int(float(parts[1])); b = int(float(parts[2]))
        return (r,g,b)
    # fallback white
    return (255,255,255)

def rgb_to_ass_bgr(r:int,g:int,b:int, alpha: int = 0) -> str:
    """
    ASS color format is &HAABBGGRR (hex). We’ll keep alpha 00 (opaque).
    """
    aa = max(0, min(alpha, 255))
    return f"&H{aa:02X}{b:02X}{g:02X}{r:02X}"

# ----------------------------
# Subtitle formatting
# ----------------------------

def to_srt(chunks: List[Dict]) -> str:
    lines = []
    for i,c in enumerate(chunks, start=1):
        t1 = c["start"]; t2 = c["end"]
        def fmt(t):
            ms = int((t - int(t)) * 1000)
            s = int(t) % 60
            m = (int(t) // 60) % 60
            h = int(t) // 3600
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        lines.append(str(i))
        lines.append(f"{fmt(t1)} --> {fmt(t2)}")
        lines.append(c["text"].strip())
        lines.append("")
    return "\n".join(lines)

def to_vtt(chunks: List[Dict]) -> str:
    out = ["WEBVTT",""]
    for c in chunks:
        def fmt(t):
            ms = int((t - int(t)) * 1000)
            s = int(t) % 60
            m = (int(t) // 60) % 60
            h = int(t) // 3600
            return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
        out.append(f"{fmt(c['start'])} --> {fmt(c['end'])}")
        out.append(c["text"].strip())
        out.append("")
    return "\n".join(out)

def to_ass(chunks: List[Dict], font:str, size:int,
           text_hex: str, outline_hex: str, outline_w:int,
           boxed_bg: bool, bg_hex: str) -> str:
    # Parse colors robustly
    tr,tg,tb = parse_color(text_hex)
    or_,og,ob = parse_color(outline_hex)
    br,bg,bb = parse_color(bg_hex)

    primary = rgb_to_ass_bgr(tr,tg,tb, 0)
    outline = rgb_to_ass_bgr(or_,og,ob, 0)
    back    = rgb_to_ass_bgr(br,bg,bb, 0)

    # BorderStyle: 1=Outline, 3=Opaque box
    border_style = 3 if boxed_bg else 1

    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1280",
        "PlayResY: 720",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font},{size},{primary},&H00FFFFFF,{outline},{back},"
        f"0,0,0,0,100,100,0,0,{border_style},{max(outline_w,0)},0,2,30,30,24,0",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    def ts(t: float) -> str:
        cs = int((t - int(t)) * 100)
        s = int(t) % 60
        m = (int(t) // 60) % 60
        h = int(t) // 3600
        return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

    lines = []
    for c in chunks:
        text = c["text"].replace("\n"," ").strip()
        lines.append(f"Dialogue: 0,{ts(c['start'])},{ts(c['end'])},Default,,0,0,0,,{text}")

    return "\n".join(header + lines)

# ----------------------------
# Chunking: cap ~5 words each
# ----------------------------

def split_into_chunks(segments: List[Dict], max_words:int = 5) -> List[Dict]:
    chunks = []
    for seg in segments:
        words = seg["text"].strip().split()
        if not words:
            continue
        cur = []
        cur_start = seg["start"]
        span = max(seg["end"] - seg["start"], 0.001) / max(len(words),1)
        for i,w in enumerate(words):
            if not cur:
                cur_start = seg["start"] + i*span
            cur.append(w)
            if len(cur) >= max_words:
                start = cur_start
                end   = seg["start"] + (i+1)*span
                chunks.append({"start":start, "end":end, "text":" ".join(cur)})
                cur = []
        if cur:
            end = seg["end"]
            chunks.append({"start":cur_start, "end":end, "text":" ".join(cur)})
    return chunks

# ----------------------------
# ASR
# ----------------------------

def transcribe(audio_path: str, language_code: Optional[str]) -> Tuple[List[Dict], str, float]:
    model = get_asr()
    # faster_whisper pipeline in whisperx ignores word_timestamps kw; keep it simple
    result = model.transcribe(audio_path, language=language_code)
    segments: List[Dict] = result["segments"]
    # Total duration fallback
    total = segments[-1]["end"] if segments else 0.0
    text = " ".join(s["text"].strip() for s in segments)
    return segments, text, total

# ----------------------------
# Build outputs from chunks
# ----------------------------

def build_from_chunks(
    media_path: str,
    chunks: List[Dict],
    font: str, fsize: int,
    text_color: str, outline_color: str, outline_w: int,
    boxed_bg: bool, bg_color: str,
    layout: str
) -> Tuple[str, str, str]:
    # Preview HTML (uses CSS color directly)
    styled_html = f"""
<div style="
  font-family:{font if font!='Default' else 'inherit'}, sans-serif;
  font-size:{int(fsize)}px;
  line-height:1.35;
  color:{text_color};
  text-shadow:
    -{outline_w}px 0 {outline_color},
    {outline_w}px 0 {outline_color},
    0 -{outline_w}px {outline_color},
    0 {outline_w}px {outline_color};
  {'background:'+bg_color+'; padding:4px 8px; border-radius:6px;' if boxed_bg else ''}
">
{" ".join([c['text'] for c in chunks])}
</div>
""".strip()

    # Save subtitle files
    srt_text = to_srt(chunks)
    vtt_text = to_vtt(chunks)
    ass_text = to_ass(chunks, font if font!="Default" else "Roboto",
                      int(fsize), text_color, outline_color, int(outline_w),
                      bool(boxed_bg), bg_color)

    tmp = tempfile.mkdtemp(prefix="subs_")
    srt_path = os.path.join(tmp, "subtitles.srt")
    ass_path = os.path.join(tmp, "subtitles.ass")
    vtt_path = os.path.join(tmp, "subtitles.vtt")
    with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_text)
    with open(ass_path, "w", encoding="utf-8") as f: f.write(ass_text)
    with open(vtt_path, "w", encoding="utf-8") as f: f.write(vtt_text)

    return styled_html, srt_path, ass_path, vtt_path

# ----------------------------
# Render with ffmpeg (burn ASS)
# ----------------------------

def render_video(media_path: str, ass_path: str, layout:str, duration: float) -> str:
    if not media_path or not os.path.exists(media_path):
        raise gr.Error("No input media.")
    if not ass_path or not os.path.exists(ass_path):
        raise gr.Error("No ASS file. Click Run first.")

    # Pick resolution
    if layout.startswith("16:9"):
        w,h = 1280,720
    elif layout.startswith("9:16"):
        w,h = 1080,1920
    else:
        w,h = 1080,1080

    out_mp4 = os.path.join(tempfile.mkdtemp(prefix="render_"), "out.mp4")
    cmd = [
        "ffmpeg","-y",
        "-i", media_path,
        "-vf", f"scale={w}:{h},ass='{ass_path}'",
        "-c:v","libx264","-preset","veryfast","-crf","18",
        "-c:a","aac","-b:a","192k",
        out_mp4
    ]
    subprocess.run(cmd, check=True)
    return out_mp4

# ----------------------------
# Pipeline (Run button)
# ----------------------------

def run_pipeline(
    media_path: str,
    language_ui: str,
    words_per_chunk: int,
    layout: str,
    font: str, fsize: int, text_color: str,
    outline_color: str, outline_w: int,
    boxed_bg: bool, bg_color: str,
    global_shift: float
):
    if not media_path:
        raise gr.Error("Please upload audio or video first.")

    lang_code = normalize_lang(language_ui)
    segments, _, duration = transcribe(media_path, lang_code)
    chunks = split_into_chunks(segments, max(1,int(words_per_chunk)))

    # global shift
    if abs(global_shift) > 1e-6:
        for c in chunks:
            c["start"] = max(0.0, c["start"] + global_shift)
            c["end"] = max(c["start"] + 0.01, c["end"] + global_shift)

    preview_html, srt_path, ass_path, vtt_path = build_from_chunks(
        media_path, chunks, font, fsize, text_color, outline_color, outline_w, boxed_bg, bg_color, layout
    )

    # simple table
    table = [{"start":round(c["start"],2),"end":round(c["end"],2),"text":c["text"]} for c in chunks]
    return chunks, table, preview_html, srt_path, ass_path, vtt_path, duration

# ----------------------------
# UI
# ----------------------------

theme = gr.themes.Soft(primary_hue="violet", neutral_hue="slate")
custom_css = """
.gradio-container { max-width: 1200px !important; }
"""

with gr.Blocks(theme=theme, css=custom_css) as demo:
    gr.Markdown("## 🎨 Colorvideo Subs — v1.1")

    with gr.Row():
        with gr.Column(scale=3):
            media = gr.Audio(label="Audio / Video (mp3, wav, mp4…)", type="filepath")
            language = gr.Dropdown(choices=LANG_CHOICES, value="auto", label="Language")

            words_per_chunk = gr.Slider(1,8,value=5,step=1, label="Words per chunk (when not using lyrics)")
            layout = gr.Dropdown(choices=LAYOUT_CHOICES, value="16:9 (YouTube)", label="Layout preset")

            with gr.Accordion("Timing tools", open=False):
                global_shift = gr.Slider(-5,5,value=0,step=0.1, label="Global shift (seconds)")

            run = gr.Button("Run", variant="primary")

            preview = gr.HTML(label="Preview")
            table = gr.JSON(label="Segments (table)")

        with gr.Column(scale=1):
            gr.Markdown("### Subtitle Style")
            font = gr.Dropdown(["Default","Roboto","Arial","Open Sans","Lato","Montserrat"], value="Default", label="Font")
            fsize = gr.Slider(14,72,value=36,step=1, label="Font size")
            text_color = gr.ColorPicker(value="#FFFFFF", label="Text color")
            outline_color = gr.ColorPicker(value="#000000", label="Outline color")
            outline_w = gr.Slider(0,6,value=2,step=1, label="Outline width (px)")

            gr.Markdown("#### Background")
            boxed_bg = gr.Checkbox(value=True, label="Boxed background behind text")
            bg_color = gr.ColorPicker(value="#000000", label="Background color")

            with gr.Accordion("Per-line editor", open=False):
                gr.Markdown("Apply edits in the table soon (placeholder).")
                gr.Checkbox(value=True, label="Apply edits & refresh")

    # Download buttons
    srt_dl = gr.DownloadButton("Download SRT")
    ass_dl = gr.DownloadButton("Download ASS")
    vtt_dl = gr.DownloadButton("Download VTT")

    # Render section
    render_btn = gr.Button("Render subtitle video (MP4) 🖼️")
    rendered = gr.File(label="Rendered preview")

    # ------------------------
    # State we need across actions
    # ------------------------
    srt_path_state = gr.State()
    ass_path_state = gr.State()
    vtt_path_state = gr.State()
    duration_state = gr.State()

    # ------------------------
    # Wiring
    # ------------------------

    def _run_and_return(media_path, language_ui, words_per_chunk, layout,
                        font, fsize, text_color, outline_color, outline_w,
                        boxed_bg, bg_color, global_shift):
        try:
            chunks, table_data, prev_html, srt_path, ass_path, vtt_path, duration = run_pipeline(
                media_path, language_ui, words_per_chunk, layout,
                font, fsize, text_color, outline_color, outline_w, boxed_bg, bg_color, global_shift
            )
            return (prev_html, table_data,
                    srt_path, ass_path, vtt_path,
                    duration, srt_path, ass_path, vtt_path)
        except Exception as e:
            return (f"<div style='color:#ff6b6b'>Error: {gr.utils.sanitize_html(str(e))}</div>",
                    gr.update(value=None), None, None, None, 0.0, None, None, None)

    run.click(
        _run_and_return,
        inputs=[media, language, words_per_chunk, layout,
                font, fsize, text_color, outline_color, outline_w,
                boxed_bg, bg_color, global_shift],
        outputs=[
            preview, table,
            srt_dl, ass_dl, vtt_dl,           # download buttons expect file paths
            duration_state,                   # store duration
            srt_path_state, ass_path_state, vtt_path_state  # keep paths for render
        ]
    )

    def _render(media_path, ass_path, layout, duration):
        return render_video(media_path, ass_path, layout, duration or 0.0)

    render_btn.click(
        _render,
        inputs=[media, ass_path_state, layout, duration_state],
        outputs=rendered
    )

if __name__ == "__main__":
    demo.launch()
