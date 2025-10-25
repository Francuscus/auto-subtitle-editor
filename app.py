import os
import re
import json
import math
from typing import List, Tuple, Optional

import gradio as gr
import torch
import whisperx


# ==============================
# Language helpers
# ==============================
LANG_MAP = {
    "auto": None, "auto-detect": None, "automatic": None,
    "spanish": "es", "es": "es",
    "hungarian": "hu", "hu": "hu",
    # extras if user types them
    "english": "en", "en": "en",
}

def normalize_lang(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = s.strip().lower()
    if t in LANG_MAP:
        return LANG_MAP[t]
    # Accept labels like "Spanish (es)" or "es (Spanish)"
    m = re.search(r"\b([a-z]{2,3})\b", t)
    if m:
        code = m.group(1)
        return LANG_MAP.get(code, code)
    return None


# ==============================
# WhisperX model (lazy / CPU-safe)
# ==============================
_asr_model = None

def get_asr_model():
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"  # int8 on CPU to avoid float16 error

    try:
        _asr_model = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        # Fallback if host CPU lacks required int8 kernels
        if "compute type" in str(e).lower():
            fallback = "int16" if device == "cpu" else "float32"
            _asr_model = whisperx.load_model("small", device=device, compute_type=fallback)
        else:
            raise
    return _asr_model


# ==============================
# ASR & simple retiming
# ==============================
def transcribe(audio_path: str, language_code: Optional[str]) -> Tuple[List[dict], str]:
    """
    Returns (segments, full_text)
    segments: [{start: float, end: float, text: str}, ...]
    """
    model = get_asr_model()
    # Do NOT pass unsupported kwargs (e.g., word_timestamps) on some builds.
    result = model.transcribe(audio_path, language=language_code)
    segs = result.get("segments", [])
    text = " ".join(s.get("text", "").strip() for s in segs).strip()

    cleaned = []
    for s in segs:
        cleaned.append({
            "start": float(max(0.0, s.get("start", 0.0) or 0.0)),
            "end": float(max(0.0, s.get("end", 0.0) or 0.0)),
            "text": (s.get("text") or "").strip(),
        })
    return cleaned, text


def spread_lyrics_over_duration(lines: List[str], duration: float) -> List[dict]:
    """Evenly spread lyric lines across [0, duration]."""
    n = max(1, len(lines))
    step = duration / n
    segs = []
    for i, line in enumerate(lines):
        start = step * i
        end = step * (i + 1) if i < n - 1 else duration
        segs.append({"start": start, "end": end, "text": line.strip()})
    return segs


def retime_with_lyrics(original: List[dict], lyrics_text: str) -> List[dict]:
    """
    Lightweight retime: distribute pasted lyric lines evenly across
    the original timeline length.
    """
    lines = [ln.strip() for ln in lyrics_text.splitlines() if ln.strip()]
    if not lines:
        return original

    if not original:
        total = 60.0  # assume 60s if nothing detected
    else:
        total = max(0.0, original[-1]["end"] - original[0]["start"])

    return spread_lyrics_over_duration(lines, total)


# ==============================
# Export helpers (SRT / ASS)
# ==============================
def secs_to_ts(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - math.floor(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def secs_to_ass_ts(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - math.floor(t)) * 100))  # centiseconds
    return f"{h:01d}:{m:02d}:{s:02d}.{cs:02d}"

def make_srt(segments: List[dict]) -> str:
    lines = []
    for i, s in enumerate(segments, 1):
        start = secs_to_ts(s["start"])
        end = secs_to_ts(s["end"])
        text = (s["text"] or "").strip()
        lines += [str(i), f"{start} --> {end}", text, ""]
    return "\n".join(lines).strip() + "\n"

def make_ass(segments: List[dict], font_family: str, font_size: int,
             text_color: str, outline_color: str, outline_w: int) -> str:
    # Minimal ASS with a single Default style. (Note: not converting HTML hex to ASS BGR here.)
    style = (
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_family if font_family!='Default' else 'Arial'},{int(font_size)},"
        "&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,"
        f"{max(0,int(outline_w))},0,2,10,10,20,1\n"
    )

    events = ["[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for s in segments:
        start = secs_to_ass_ts(s["start"])
        end = secs_to_ass_ts(s["end"])
        text = (s["text"] or "").replace("\n", "\\N")
        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1280\n"
        "PlayResY: 720\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
    )
    return "\n".join([header, style, "\n".join(events)]) + "\n"


# ==============================
# UI callback
# ==============================
def run_pipeline(
    audio_path,
    language_ui,
    font_family, font_size, text_color, outline_color, outline_w,
    use_lyrics_retime, lyrics_text
):
    if not audio_path:
        raise gr.Error("Please upload an audio or video file first.")

    lang_code = normalize_lang(language_ui)
    segments, full_text = transcribe(audio_path, lang_code)

    if use_lyrics_retime and lyrics_text.strip():
        segments_out = retime_with_lyrics(segments, lyrics_text)
    else:
        segments_out = segments

    # Live preview HTML
    styled_html = f"""
<div style="
  font-family:{'inherit' if font_family=='Default' else font_family}, sans-serif;
  font-size:{int(font_size)}px;
  line-height:1.35;
  color:{text_color};
  text-shadow:
    -{outline_w}px 0 {outline_color},
    {outline_w}px 0 {outline_color},
    0 -{outline_w}px {outline_color},
    0 {outline_w}px {outline_color};
  padding: 12px 16px;
  background: rgba(255,255,255,0.03);
  border-radius: 12px;
">
{full_text}
</div>
""".strip()

    srt_text = make_srt(segments_out)
    ass_text = make_ass(segments_out, font_family, int(font_size), text_color, outline_color, int(outline_w))

    return styled_html, segments_out, srt_text, ass_text


def download_file(contents: str, filename: str) -> str:
    path = f"/tmp/{filename}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(contents)
    return path


# ==============================
# Gradio app
# ==============================
custom_css = """
/* Dark UI with clear contrast */
.gradio-container { max-width: 1280px !important; }
body { background: #0f1220; }
#titlebar h1 { color: #FFFFFF !important; }
.grey-note { color: #a7adc0; }
.preview-card .label-wrap > label { color: #cdd3eb !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🎨 Colorvideo Subs", elem_id="titlebar")
    gr.Markdown(
        "<span class='grey-note'>Upload audio/video, pick language (Auto/ES/HU),"
        " tweak font & colors on the right, (optionally) paste lyrics to retime,"
        " then export SRT/ASS.</span>",
        elem_classes=["grey-note"]
    )

    with gr.Row():
        with gr.Column(scale=3):
            audio = gr.Audio(label="Audio / Video (mp3, wav, mp4)", type="filepath")

            language = gr.Dropdown(
                label="Language",
                choices=[("Auto-detect", "auto"), ("Spanish (es)", "es"), ("Hungarian (hu)", "hu")],
                value="auto"
            )

            run_btn = gr.Button("Run", variant="primary")

            preview = gr.HTML(label="Preview", elem_classes=["preview-card"])
            segments_json = gr.JSON(label="Segments")

            with gr.Accordion("Export", open=True):
                # Textboxes instead of Code (wider Gradio compatibility)
                srt_box = gr.Textbox(label="SRT", lines=10, show_copy_button=True)
                ass_box = gr.Textbox(label="ASS", lines=10, show_copy_button=True)
                with gr.Row():
                    # In older Gradio, you just set the label here,
                    # and later populate the button with a file path.
                    srt_dl = gr.DownloadButton("Download SRT")
                    ass_dl = gr.DownloadButton("Download ASS")

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Subtitle Settings")
                font_family = gr.Dropdown(
                    ["Default", "Arial", "Roboto", "Open Sans", "Lato", "Noto Sans", "Montserrat"],
                    value="Default", label="Font"
                )
                font_size = gr.Slider(14, 72, value=34, step=1, label="Font size")
                text_color = gr.ColorPicker(value="#FFFFFF", label="Text color")
                outline_color = gr.ColorPicker(value="#000000", label="Outline color")
                outline_w = gr.Slider(0, 4, value=2, step=1, label="Outline width (px)")

            with gr.Group():
                gr.Markdown("### Optional lyrics (retime)")
                use_lyrics = gr.Checkbox(value=False, label="Use pasted lyrics to retime")
                lyrics_box = gr.Textbox(
                    label="Paste lyrics or lines here",
                    placeholder="One line per subtitle… or paste a paragraph and I'll split by lines.",
                    lines=10
                )

    # Wire up main run
    run_btn.click(
        run_pipeline,
        inputs=[audio, language, font_family, font_size, text_color, outline_color, outline_w, use_lyrics, lyrics_box],
        outputs=[preview, segments_json, srt_box, ass_box]
    )

    # Wire up downloads (return a file path; Gradio uses the basename as the download name)
    srt_dl.click(lambda s: download_file(s, "subtitles.srt"), inputs=[srt_box], outputs=[srt_dl])
    ass_dl.click(lambda a: download_file(a, "subtitles.ass"), inputs=[ass_box], outputs=[ass_dl])


if __name__ == "__main__":
    demo.launch()
