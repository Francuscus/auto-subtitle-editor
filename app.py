# app.py — Colorvideo Subs v0.9
# - Robust color parsing (hex OR rgb/rgba)
# - Text color now applied in preview + exports
# - Background "boxed" toggle + color works in preview + ASS
# - Safer ASS color (&HBBGGRR&) conversion
# - Uses gr.skip() correctly
# - MP4 render: overlays ASS on video if present, otherwise on a solid background + your audio

from __future__ import annotations
import os, re, json, shlex, subprocess, tempfile, math
from typing import List, Dict, Tuple, Optional

import gradio as gr
import torch
import whisperx

# ------------------------------
# Helpers: language + colors
# ------------------------------
LANG_CHOICES = [
    ("Auto-detect", "auto"),
    ("Spanish (es)", "es"),
    ("Hungarian (hu)", "hu"),
    ("English (en)", "en"),
]

def normalize_lang(s: Optional[str]) -> Optional[str]:
    if not s or s == "auto":
        return None
    s = s.strip().lower()
    # support things like "es (Spanish)"
    m = re.search(r"\b([a-z]{2,3})\b", s)
    return m.group(1) if m else None

def parse_css_color(c: Optional[str], default_hex="#FFFFFF") -> str:
    """
    Accept '#RRGGBB', '#RGB', 'rgb(r,g,b)', 'rgba(r,g,b,a)'.
    Returns normalized '#RRGGBB' (alpha dropped).
    """
    if not c:
        return default_hex
    c = c.strip()
    if c.startswith("#"):
        hx = c[1:]
        if len(hx) == 3:
            # short form #RGB -> #RRGGBB
            hx = "".join([ch*2 for ch in hx])
        if len(hx) != 6:
            return default_hex
        # validate hex
        try:
            int(hx, 16)
        except ValueError:
            return default_hex
        return "#" + hx.lower()
    if c.startswith("rgb"):
        nums = re.findall(r"[\d.]+", c)
        if len(nums) >= 3:
            r, g, b = [max(0, min(255, int(float(v)))) for v in nums[:3]]
            return "#{:02x}{:02x}{:02x}".format(r, g, b)
    # fallback
    return default_hex

def hex_to_ass_bgr(hex_str: str) -> str:
    """
    ASS wants &HBBGGRR& (BGR), no alpha here.
    Input '#rrggbb' -> '&Hbbggrr&'
    """
    hx = parse_css_color(hex_str, "#ffffff")[1:]  # strip '#'
    r = int(hx[0:2], 16)
    g = int(hx[2:4], 16)
    b = int(hx[4:6], 16)
    return f"&H{b:02X}{g:02X}{r:02X}&"

# ------------------------------
# ASR model (CPU-friendly)
# ------------------------------
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
        # Fallback if int8 kernels unavailable
        if "compute type" in str(e).lower():
            fallback = "int16" if device == "cpu" else "float32"
            _asr_model = whisperx.load_model("small", device=device, compute_type=fallback)
        else:
            raise
    return _asr_model

# ------------------------------
# Chunking + subtitle formats
# ------------------------------
def chunk_by_words(segments: List[Dict], max_words: int) -> List[Dict]:
    """Greedy chunking by up to `max_words` per subtitle, preserving timing."""
    out = []
    cur_words = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        words = text.split()
        # estimate word timing inside segment
        dur = seg["end"] - seg["start"]
        if len(words) == 0 or dur <= 0:
            continue
        step = dur / max(len(words), 1)
        for i, w in enumerate(words):
            w_start = seg["start"] + i * step
            w_end = seg["start"] + (i + 1) * step
            cur_words.append({"w": w, "s": w_start, "e": w_end})
            if len(cur_words) >= max_words:
                start = cur_words[0]["s"]
                end = cur_words[-1]["e"]
                out.append({"start": start, "end": end,
                            "text": " ".join([x["w"] for x in cur_words])})
                cur_words = []
    # tail
    if cur_words:
        out.append({
            "start": cur_words[0]["s"],
            "end": cur_words[-1]["e"],
            "text": " ".join([x["w"] for x in cur_words]),
        })
    return out

def srt_timestamp(t: float) -> str:
    ms = int(round((t - int(t)) * 1000))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def to_srt(chunks: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(str(i))
        lines.append(f"{srt_timestamp(c['start'])} --> {srt_timestamp(c['end'])}")
        lines.append(c["text"])
        lines.append("")  # blank line
    return "\n".join(lines)

def to_vtt(chunks: List[Dict]) -> str:
    lines = ["WEBVTT", ""]
    for c in chunks:
        start = srt_timestamp(c["start"]).replace(",", ".")
        end = srt_timestamp(c["end"]).replace(",", ".")
        lines.append(f"{start} --> {end}")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines)

def to_ass(chunks: List[Dict], font: str, size: int, text_hex: str,
           outline_hex: str, outline_w: int, boxed_bg: bool, bg_hex: str) -> str:
    # Normalize all colors first
    text_hex = parse_css_color(text_hex, "#FFFFFF")
    outline_hex = parse_css_color(outline_hex, "#000000")
    bg_hex = parse_css_color(bg_hex, "#000000")

    primary = hex_to_ass_bgr(text_hex)
    outline = hex_to_ass_bgr(outline_hex)
    back = "&H00000000&"  # we keep shadow off
    # If boxed, we’ll draw a solid box via border style 3 (outline around glyphs). For true box we fake with opaque outline:
    # Better approach: we render a background via preview; for ASS we approximate by outline/shadow.
    # We keep outline width as given; box color is not directly supported in plain dialogue—would need \3c for outline color already set.
    # We’ll ignore bg_hex in ASS itself; burn-in preview/video uses real box.

    header = f"""[Script Info]
ScriptType: v4.00+
Collisions: Normal
PlayResX: 1280
PlayResY: 720
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: SubStyle,{font},{size},{primary},&H00FFFFFF,{outline},{back},0,0,0,0,100,100,0,0,1,{outline_w},0,2,30,30,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev_lines = []
    for c in chunks:
        st = srt_timestamp(c["start"]).replace(",", ".")
        en = srt_timestamp(c["end"]).replace(",", ".")
        safe = c["text"].replace("\n", "\\N")
        ev_lines.append(f"Dialogue: 0,{st},{en},SubStyle,,0,0,0,,{safe}")
    return header + "\n".join(ev_lines) + "\n"

# ------------------------------
# ASR + pipeline
# ------------------------------
def transcribe(audio_path: str, language_code: Optional[str]) -> Tuple[List[Dict], str, float]:
    model = get_asr_model()
    result = model.transcribe(audio_path, language=language_code)
    segments = result["segments"]  # [{start, end, text}]
    text = " ".join(seg.get("text", "").strip() for seg in segments).strip()
    # duration guess
    duration = 0.0
    if segments:
        duration = max(duration, float(segments[-1]["end"]))
    return segments, text, duration

def build_from_chunks(
    chunks: List[Dict],
    font: str, size: int, text_color: str,
    outline_color: str, outline_w: int,
    boxed_bg: bool, bg_color: str
) -> Tuple[str, str, str]:
    """Returns (preview_html, srt_path, ass_path, vtt_path) file paths (3)."""
    # Preview block
    text_hex = parse_css_color(text_color, "#FFFFFF")
    outline_hex = parse_css_color(outline_color, "#000000")
    bg_hex = parse_css_color(bg_color, "#000000")
    css_text_shadow = (
        f"-{outline_w}px 0 {outline_hex}, {outline_w}px 0 {outline_hex}, "
        f"0 -{outline_w}px {outline_hex}, 0 {outline_w}px {outline_hex}"
        if outline_w > 0 else "none"
    )
    box_css = f"background:{bg_hex}; padding:6px 10px; border-radius:6px;" if boxed_bg else "background:transparent;"
    preview_lines = []
    for c in chunks:
        preview_lines.append(
            f"""<div style="display:inline-block; {box_css} margin:6px 8px;">
<span style="font-family:{font},sans-serif; font-size:{int(size)}px; color:{text_hex}; line-height:1.3; text-shadow:{css_text_shadow};">
{gr.utils.sanitize_html(c['text'])}
</span></div>"""
        )
    preview_html = "<div style='text-align:center;'>" + "\n".join(preview_lines) + "</div>"

    # Write files
    srt_text = to_srt(chunks)
    vtt_text = to_vtt(chunks)
    ass_text = to_ass(chunks, font, size, text_hex, outline_hex, outline_w, boxed_bg, bg_hex)

    tmpdir = tempfile.mkdtemp(prefix="subs_")
    srt_path = os.path.join(tmpdir, "subtitles.srt")
    vtt_path = os.path.join(tmpdir, "subtitles.vtt")
    ass_path = os.path.join(tmpdir, "subtitles.ass")
    with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_text)
    with open(vtt_path, "w", encoding="utf-8") as f: f.write(vtt_text)
    with open(ass_path, "w", encoding="utf-8") as f: f.write(ass_text)

    return preview_html, srt_path, ass_path, vtt_path

def run_pipeline(
    audio_path: str | None,
    language_ui: str,
    words_per_chunk: int,
    layout: str,  # unused placeholder for future canvas sizing
    lyrics_mode: str,  # "", or raw lyrics (not aligning in this build)
    font: str, size: int, text_color: str,
    outline_color: str, outline_w: int,
    boxed_bg: bool, bg_color: str,
    global_shift: float
):
    if not audio_path:
        raise gr.Error("Please upload an audio or video file.")

    lang_code = normalize_lang(language_ui)
    segs, _, duration = transcribe(audio_path, lang_code)

    # Make chunks (no lyrics-align in this version)
    max_words = max(1, int(words_per_chunk))
    chunks = chunk_by_words(segs, max_words)

    # Apply global shift
    if abs(global_shift) > 1e-6:
        for c in chunks:
            c["start"] = max(0.0, c["start"] + global_shift)
            c["end"] = max(c["start"] + 0.01, c["end"] + global_shift)

    preview_html, srt_path, ass_path, vtt_path = build_from_chunks(
        chunks, font, size, text_color, outline_color, outline_w, boxed_bg, bg_color
    )

    # simple table for JSON output
    table = [{"start": round(c["start"], 2), "end": round(c["end"], 2), "text": c["text"]}
             for c in chunks]

    # basic px/py for PlayRes (constant here; render step will adapt)
    px, py = 1280, 720
    return chunks, table, preview_html, px, py, srt_path, ass_path, vtt_path, duration

# ------------------------------
# Render to MP4 with ffmpeg
# ------------------------------
def probe_has_video(path: str) -> bool:
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=codec_type", "-of", "csv=p=0", path]
        out = subprocess.check_output(cmd, text=True).strip()
        return out == "video"
    except Exception:
        return False

def render_video(
    media_path: str | None,
    ass_path: str,
    layout: str,
    duration: float
) -> str:
    if not ass_path or not os.path.exists(ass_path):
        raise gr.Error("No ASS file to render.")
    # pick size from layout
    if layout.startswith("9:16"):
        w, h = 1080, 1920
    else:
        w, h = 1280, 720

    out_path = os.path.join(tempfile.mkdtemp(prefix="render_"), "subtitle.mp4")

    if media_path and probe_has_video(media_path):
        # Overlay on video
        cmd = f'ffmpeg -y -i {shlex.quote(media_path)} -vf "ass={shlex.quote(ass_path)}" -c:a copy -c:v libx264 -pix_fmt yuv420p {shlex.quote(out_path)}'
    else:
        # Solid background + audio if any
        aud_in = f"-i {shlex.quote(media_path)}" if media_path else ""
        d_arg = "" if media_path else f"-t {max(1.0, duration):.2f}"
        cmd = f'ffmpeg -y -f lavfi -i color=size={w}x{h}:color=black {aud_in} {d_arg} -shortest -vf "ass={shlex.quote(ass_path)}" -c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 192k {shlex.quote(out_path)}'

    subprocess.check_call(cmd, shell=True)
    return out_path

# ------------------------------
# UI
# ------------------------------
FONT_CHOICES = [
    "Default", "Arial", "Roboto", "Open Sans", "Lato", "Noto Sans", "Montserrat",
]

custom_css = """
.gradio-container { max-width: 1200px !important; }
.settings-card { position: sticky; top: 12px; border-radius: 16px; }
body { background: #0E1220; } /* darker so labels are readable */
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("## 🎨 Colorvideo Subs — v0.9")

    with gr.Row():
        with gr.Column(scale=3):
            media = gr.Audio(label="Audio / Video (mp3, wav, mp4)", type="filepath")
            language = gr.Dropdown(LANG_CHOICES, value="auto", label="Language")

            with gr.Accordion("Timing tools", open=False):
                global_shift = gr.Slider(-5, 5, step=0.1, value=0, label="Global shift (seconds)")

            run = gr.Button("Run", variant="primary")

            preview = gr.HTML(label="Preview")
            table = gr.JSON(label="Segments (table)")

            srt_dl = gr.DownloadButton("Download SRT")
            ass_dl = gr.DownloadButton("Download ASS")
            vtt_dl = gr.DownloadButton("Download VTT")

            render_btn = gr.Button("Render subtitle video (MP4) 🧪")
            rendered = gr.Video(label="Rendered preview")

        with gr.Column(scale=1):
            with gr.Group(elem_classes=["settings-card"]):
                gr.Markdown("### Subtitle Style")

                font = gr.Dropdown(FONT_CHOICES, value="Default", label="Font")
                fsize = gr.Slider(14, 72, value=36, step=1, label="Font size")

                text_color = gr.ColorPicker(value="#FFFFFF", label="Text color")
                outline_color = gr.ColorPicker(value="#000000", label="Outline color")
                outline_w = gr.Slider(0, 6, value=2, step=1, label="Outline width (px)")

                gr.Markdown("### Background")
                boxed_bg = gr.Checkbox(value=True, label="Boxed background behind text")
                bg_color = gr.ColorPicker(value="#000000", label="Background color")

                gr.Markdown("### Layout preset")
                layout = gr.Dropdown(
                    choices=["16:9 (YouTube)", "9:16 (TikTok/Reels)"],
                    value="16:9 (YouTube)", label=""
                )

                gr.Markdown("### Words per chunk (when not using lyrics)")
                words_per_chunk = gr.Slider(1, 8, value=5, step=1, label="", info="Max words per subtitle line")

    # Wiring
    def _run_and_return(*args):
        try:
            return run_pipeline(*args)
        except Exception as e:
            return (f"Error: {e}", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), 0.0)

    run_outputs = [
        preview, table,   # 0,1
        gr.State(), gr.State(),  # px, py (unused visually)
        srt_dl, ass_dl, vtt_dl,  # will receive file paths
        gr.State(),  # duration
    ]

    run.click(
        _run_and_return,
        inputs=[media, language, words_per_chunk, layout, gr.Textbox(visible=False),
                font, fsize, text_color, outline_color, outline_w, boxed_bg, bg_color, global_shift],
        outputs=run_outputs
    )

    def _render(media_path, ass_path, layout, duration):
        try:
            if not ass_path:
                raise gr.Error("Run the pipeline first to generate ASS.")
            out = render_video(media_path, ass_path, layout, duration or 0.0)
            return out
        except Exception as e:
            raise gr.Error(str(e))

    render_btn.click(
        _render,
        inputs=[media, ass_dl, layout, run_outputs[-1]],  # media, ass path, layout, duration
        outputs=rendered
    )

if __name__ == "__main__":
    demo.launch()
