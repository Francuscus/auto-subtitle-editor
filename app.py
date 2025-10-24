import os
import re
import gradio as gr
import torch
import whisperx

# =========================
# Version tag (helps confirm deployment)
APP_TITLE = "CCC Colorvideo Subs — v0.5"
# =========================

# ---------- Language helpers ----------
LANG_MAP = {
    "auto": None, "auto-detect": None, "automatic": None,
    "hungarian": "hu", "magyar": "hu",
    "spanish": "es", "español": "es",
    "english": "en",
    # keep a few common ones to avoid errors if chosen accidentally
    "french": "fr", "german": "de", "italian": "it",
    "japanese": "ja", "korean": "ko", "portuguese": "pt", "russian": "ru",
    "chinese": "zh", "cantonese": "yue", "arabic": "ar", "hindi": "hi",
}

def normalize_lang(s: str | None):
    """Return a 2/3-letter code or None for auto-detect."""
    if not s:
        return None
    t = s.strip().lower()
    if t in LANG_MAP:
        return LANG_MAP[t]
    # Accept labels like "es (Spanish)" or "Spanish (es)" or "es Spanish"
    m = re.search(r"\b([a-z]{2,3})\b", t)
    if m:
        return m.group(1)
    tok = t.split()[0]
    if tok in LANG_MAP:
        return LANG_MAP[tok]
    if 2 <= len(tok) <= 3:
        return tok
    return None

# ---------- Model (lazy load + CPU-safe compute_type) ----------
_asr_model = None

def get_asr_model():
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"   # int8 is safe & fast-ish on CPU

    try:
        _asr_model = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        # Fallback if this CPU lacks int8 kernels
        if "compute type" in str(e).lower():
            fallback = "int16" if device == "cpu" else "float32"
            _asr_model = whisperx.load_model("small", device=device, compute_type=fallback)
        else:
            raise
    return _asr_model

def transcribe_and_align(audio_path, language_code):
    model = get_asr_model()
    result = model.transcribe(audio_path, language=language_code)
    text = " ".join(seg["text"].strip() for seg in result["segments"])
    return result["segments"], text

# ---------- UI options ----------
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
                   text_color, outline_color, outline_w, bg_color):
    if audio is None:
        raise gr.Error("Please upload an audio or video file (mp3, wav, mp4).")
    lang_code = normalize_lang(language_ui)
    segments, text = transcribe_and_align(audio, lang_code)

    # Styled live preview
    styled = f"""
<div style="
  background:{bg_color};
  padding:20px; border-radius:14px;
  font-family:{'inherit' if font_family=='Default' else font_family}, sans-serif;
  font-size:{int(font_size)}px; line-height:1.35;
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

# ---------- CSS (very visible new theme so you can tell it changed) ----------
custom_css = """
/* overall app bg */
body, .gradio-container { background: #0b1220 !important; }

/* card-like main area */
.panel-card {
  background: #0f172a; /* slate-900 */
  border: 1px solid #1f2a44;
  border-radius: 18px;
  padding: 14px;
}

/* right settings card with subtle tint */
.settings-card {
  background: #111b32; /* deeper slate */
  border: 1px solid #1f2a44;
  border-radius: 18px;
  padding: 14px;
  position: sticky;
  top: 12px;
}

/* header banner */
.header-banner {
  background: linear-gradient(90deg, #4f46e5, #06b6d4);
  color: white;
  border-radius: 16px;
  padding: 14px 16px;
  margin-bottom: 10px;
  font-weight: 700;
}
.header-subtle {
  color: #e5f2ffcc;
}

/* component labels */
label, .gr-markdown h3, .gr-markdown h4, .gr-markdown p { color: #e5e7eb !important; }

/* buttons */
button.svelte-1ipelgc, .gr-button {
  border-radius: 10px !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, analytics_enabled=False) as demo:
    # Header with obvious version label
    gr.Markdown(f"""
<div class="header-banner">{APP_TITLE}</div>
<p class="header-subtle">Transcribe and style subtitles. (Now with background color picker & new theme.)</p>
""")

    with gr.Row():
        with gr.Column(scale=3, elem_classes=["panel-card"]):
            audio = gr.Audio(label="Audio or Video (MP3 / WAV / MP4)", type="filepath")
            language = gr.Dropdown(choices=LANG_CHOICES, value="auto", label="Language")
            run = gr.Button("Run", variant="primary")

            gr.Markdown("### Preview")
            transcript_html = gr.HTML()

            gr.Markdown("### Segments (JSON)")
            segments_json = gr.JSON()

        # RIGHT SETTINGS PANEL
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["settings-card"]):
                gr.Markdown("### Subtitle Settings")
                font_family = gr.Dropdown(FONT_CHOICES, value="Default", label="Font")
                font_size   = gr.Slider(14, 72, value=32, step=1, label="Font size")
                text_color  = gr.ColorPicker(value="#FFFFFF", label="Text color")
                outline_color = gr.ColorPicker(value="#000000", label="Outline color")
                outline_w   = gr.Slider(0, 4, value=2, step=1, label="Outline width (px)")
                bg_color    = gr.ColorPicker(value="#0a0a0a", label="Background color (Preview)")

    run.click(
        main_interface,
        inputs=[audio, language, font_family, font_size, text_color, outline_color, outline_w, bg_color],
        outputs=[transcript_html, segments_json]
    )

if __name__ == "__main__":
    # SSR can cause long cold starts; leave default on HF
    demo.launch()
