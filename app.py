import os
import re
import torch
import gradio as gr
import whisperx

# ---------------- helpers ----------------

# Map common names -> language codes that faster-whisper/whisperx accept
LANG_MAP = {
    "auto": None, "auto-detect": None, "automatic": None,
    "english": "en", "en": "en",
    "spanish": "es", "es": "es",
    "hungarian": "hu", "magyar": "hu", "hu": "hu",
}

def normalize_lang(s: str | None):
    """Return a 2/3-letter code or None for auto. Accepts 'es (Spanish)' etc."""
    if not s:
        return None
    t = s.strip().lower()
    if t in LANG_MAP:
        return LANG_MAP[t]

    # Pull a 2–3 letter code if it's present anywhere
    m = re.search(r"\b([a-z]{2,3})\b", t)
    if m:
        code = m.group(1)
        # respect known keys when present
        return LANG_MAP.get(code, code)

    # Fallback: first token if it looks like a code
    tok = t.split()[0]
    if tok in LANG_MAP:
        return LANG_MAP[tok]
    if 2 <= len(tok) <= 3:
        return tok
    return None

# Lazy-loaded global model so we don't reload on every click
_asr_model = None

def get_asr_model():
    """
    Load whisperx model once.
    - GPU: float16
    - CPU: int8 (fallback to int16/float32 if needed)
    """
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"

    try:
        _asr_model = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        # Some CPUs may lack efficient int8 kernels — try a safer type.
        if "compute type" in str(e).lower():
            fallback = "int16" if device == "cpu" else "float32"
            _asr_model = whisperx.load_model("small", device=device, compute_type=fallback)
        else:
            raise
    return _asr_model


def transcribe_and_align(audio_path: str, language_code: str | None):
    model = get_asr_model()
    result = model.transcribe(audio_path, language=language_code)
    # Join segment texts for a quick preview
    text = " ".join(seg["text"].strip() for seg in result["segments"])
    return result["segments"], text

# ---------------- UI logic ----------------

FONT_CHOICES = [
    "Default", "Arial", "Roboto", "Open Sans", "Lato", "Noto Sans", "Montserrat",
]

# Focus on your 3 main choices (+ Auto)
LANG_CHOICES = [
    ("Auto-detect", "auto"),
    ("Hungarian (hu)", "hu"),
    ("English (en)", "en"),
    ("Spanish (es)", "es"),
]

def main_interface(
    audio,
    language_ui,
    font_family,
    font_size,
    text_color,
    bg_color,
    outline_color,
    outline_w,
):
    if audio is None:
        raise gr.Error("Please upload an audio/video file.")

    lang_code = normalize_lang(language_ui)
    segments, text = transcribe_and_align(audio, lang_code)

    # Build a styled preview (this is not burned into video—just a preview)
    css_font = "inherit" if font_family == "Default" else font_family
    styled = f"""
<div style="
  display:inline-block;
  padding: 8px 12px;
  border-radius: 8px;
  font-family:{css_font}, sans-serif;
  font-size:{int(font_size)}px;
  line-height:1.35;
  color:{text_color};
  background:{bg_color};
  text-shadow:
    -{outline_w}px 0 {outline_color},
     {outline_w}px 0 {outline_color},
     0 -{outline_w}px {outline_color},
     0  {outline_w}px {outline_color};
">
{text}
</div>
"""
    return styled, segments

custom_css = """
/* cleaner look + right panel sizing */
.gradio-container { max-width: 1200px !important; }
.settings-card { position: sticky; top: 12px; border-radius: 16px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🎨 Colorvideo Subs\nTranscribe + style subtitles. Pick fonts & sizes on the right →")

    with gr.Row():
        with gr.Column(scale=3):
            audio = gr.Audio(label="Audio or Video", type="filepath")
            language = gr.Dropdown(choices=LANG_CHOICES, value="auto", label="Language")
            run = gr.Button("Run", variant="primary")

            transcript_html = gr.HTML(label="Preview")
            segments_json = gr.JSON(label="Segments (for debugging / export)")

        # RIGHT SETTINGS PANEL
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["settings-card"]):
                gr.Markdown("### Subtitle Settings")
                font_family   = gr.Dropdown(FONT_CHOICES, value="Default", label="Font")
                font_size     = gr.Slider(14, 72, value=32, step=1, label="Font size")
                text_color    = gr.ColorPicker(value="#FFFFFF", label="Text color")
                bg_color      = gr.ColorPicker(value="rgba(0,0,0,0)", label="Background color")
                outline_color = gr.ColorPicker(value="#000000", label="Outline color")
                outline_w     = gr.Slider(0, 4, value=2, step=1, label="Outline width (px)")

    run.click(
        main_interface,
        inputs=[audio, language, font_family, font_size, text_color, bg_color, outline_color, outline_w],
        outputs=[transcript_html, segments_json]
    )

if __name__ == "__main__":
    # If your Space complains about SSR, you can pass ssr_mode=False here.
    demo.launch()
