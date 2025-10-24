# app.py
import os
import re
import gradio as gr
import torch
import whisperx

# ----------------------------
# Language helpers
# ----------------------------
LANG_MAP = {
    # explicit
    "auto": None, "auto-detect": None, "automatic": None,
    "hungarian": "hu", "magyar": "hu",
    "spanish": "es", "español": "es",
    "english": "en",
    # direct codes
    "hu": "hu", "es": "es", "en": "en",
}

def normalize_lang(s: str | None):
    """Return a 2/3-letter code or None for auto-detect.
       Accepts names like 'Hungarian', 'es (Spanish)', 'Spanish (es)'. """
    if not s:
        return None
    t = s.strip().lower()

    if t in LANG_MAP:
        return LANG_MAP[t]

    # Try to extract a 2-3 letter code anywhere in the string
    m = re.search(r"\b([a-z]{2,3})\b", t)
    if m:
        return m.group(1)

    # Fallback: first token if it looks like a code
    tok = t.split()[0]
    if tok in LANG_MAP:
        return LANG_MAP[tok]
    if 2 <= len(tok) <= 3:
        return tok

    return None  # default to auto-detect upstream


# ----------------------------
# ASR model (lazy load)
# ----------------------------
_asr_model = None

def get_asr_model():
    """Load WhisperX with a compute type that works on the current hardware."""
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"  # CPU: int8 to avoid float16 error

    try:
        _asr_model = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        # Fallback for environments lacking int8/float16 support
        if "compute type" in str(e).lower():
            fallback = "int16" if device == "cpu" else "float32"
            _asr_model = whisperx.load_model("small", device=device, compute_type=fallback)
        else:
            raise
    return _asr_model


def transcribe_and_align(audio_path: str, language_code: str | None):
    model = get_asr_model()
    result = model.transcribe(audio_path, language=language_code)
    # Join segments into a previewable paragraph
    text = " ".join(seg["text"].strip() for seg in result["segments"])
    return result["segments"], text


# ----------------------------
# UI logic
# ----------------------------
FONT_CHOICES = [
    "Default", "Arial", "Roboto", "Open Sans", "Lato", "Noto Sans", "Montserrat",
]
LANG_CHOICES = [
    ("Auto-detect", "auto"),
    ("Hungarian (hu)", "hu"),
    ("Spanish (es)", "es"),
    ("English (en)", "en"),
]

def main_interface(audio, language_ui, font_family, font_size,
                   text_color, outline_color, outline_w, background_color):
    if audio is None:
        raise gr.Error("Please upload an audio or video file (mp3, wav, mp4...).")

    lang_code = normalize_lang(language_ui)
    segments, text = transcribe_and_align(audio, lang_code)

    # Preview HTML (this is just a visual preview; not burned into video here)
    styled = f"""
<div style="
  background:{background_color};
  padding:16px;
  border-radius:12px;
  font-family:{'inherit' if font_family=='Default' else font_family}, sans-serif;
  font-size:{int(font_size)}px;
  line-height:1.4;
  color:{text_color};
  text-shadow:
    -{outline_w}px 0 {outline_color},
    {outline_w}px 0 {outline_color},
    0 -{outline_w}px {outline_color},
    0 {outline_w}px {outline_color};
">
{text}
</div>
"""
    return styled, segments


# ----------------------------
# Theme / CSS
# ----------------------------
custom_css = """
/* overall palette */
:root {
  --brand-bg: #0b1220;
  --panel-bg: #111827;
  --panel-2: #0d1b2a;
  --text: #e5e7eb;
  --muted: #cbd5e1;
}
body { background: var(--brand-bg); }
.gradio-container { max-width: 1200px !important; color: var(--text); }

/* title bar */
#titlebar {
  background: linear-gradient(90deg, #6d28d9 0%, #4f46e5 40%, #06b6d4 100%);
  color: #fff;
  padding: 14px 18px;
  border-radius: 14px;
  font-weight: 700;
  font-size: 20px;
  letter-spacing: .2px;
  margin-bottom: 10px;
}

/* cards / groups */
.settings-card, #preview_box, #segments_box {
  background: var(--panel-bg) !important;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px !important;
  padding: 12px;
}

/* distinct tones so they differ */
#preview_box { background: var(--panel-2) !important; }
#segments_box { background: var(--panel-bg) !important; }

/* labels and section titles */
.gradio-container .gr-block label,
.gradio-container .gr-form .block-title,
.gradio-container .gr-panel .block-title {
  color: var(--muted) !important;
}

/* ensure content stays readable */
#preview_box, #preview_box * { color: var(--text) !important; }
#segments_box, #segments_box * { color: var(--text) !important; }

/* spacing */
.gradio-container .gr-block { gap: 8px; }
.settings-card { position: sticky; top: 12px; }

/* make dropdowns/pickers readable on dark bg */
input, select, textarea {
  color: #e5e7eb;
}
"""


# ----------------------------
# App
# ----------------------------
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    title = gr.HTML(
        """
        <div id="titlebar">
          <span>🎨 Colorvideo Subs — v0.5</span>
        </div>
        """
    )

    with gr.Row():
        # LEFT: main work area
        with gr.Column(scale=3):
            audio = gr.Audio(label="Audio / Video", type="filepath")
            language = gr.Dropdown(choices=LANG_CHOICES, value="auto", label="Language")
            run = gr.Button("Run", variant="primary")

            transcript_html = gr.HTML(label="Preview", elem_id="preview_box")
            segments_json   = gr.JSON(label="Segments", elem_id="segments_box")

        # RIGHT: settings panel
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["settings-card"]):
                gr.Markdown("### Subtitle Settings")
                font_family = gr.Dropdown(FONT_CHOICES, value="Default", label="Font")
                font_size   = gr.Slider(14, 72, value=32, step=1, label="Font size")
                text_color  = gr.ColorPicker(value="#FFFFFF", label="Text color")
                outline_color = gr.ColorPicker(value="#000000", label="Outline color")
                outline_w   = gr.Slider(0, 4, value=2, step=1, label="Outline width (px)")
                background_color = gr.ColorPicker(value="#1f2937", label="Preview background")

    run.click(
        fn=main_interface,
        inputs=[audio, language, font_family, font_size, text_color, outline_color, outline_w, background_color],
        outputs=[transcript_html, segments_json]
    )

if __name__ == "__main__":
    demo.launch()
