import os, re, gradio as gr
import whisperx

# ---------- helpers ----------
LANG_MAP = {
    # common names to codes (add more if you like)
    "auto": None, "auto-detect": None, "automatic": None,
    "english": "en", "spanish": "es", "chinese": "zh", "mandarin": "zh",
    "cantonese": "yue", "french": "fr", "german": "de", "italian": "it",
    "japanese": "ja", "korean": "ko", "portuguese": "pt", "russian": "ru",
    "arabic": "ar", "hindi": "hi", "bengali": "bn", "urdu": "ur",
}

def normalize_lang(s: str | None):
    """Return a 2/3-letter code or None for auto."""
    if not s:
        return None
    t = s.strip().lower()
    if t in LANG_MAP:
        return LANG_MAP[t]
    # Accept labels like "es (Spanish)" or "Spanish (es)"
    # -> pull the 2-3 letter code if present
    m = re.search(r"\b([a-z]{2,3})\b", t)
    if m:
        return m.group(1)
    # Accept first token "es" from "es Spanish"
    tok = t.split()[0]
    if tok in LANG_MAP:
        return LANG_MAP[tok]
    if 2 <= len(tok) <= 3:
        return tok
    return None

# Lazy-load model (keeps startup snappy)
_asr_model = None
def get_asr_model():
    global _asr_model
    if _asr_model is None:
        # CPU-friendly default; HF Spaces CPU hardware
        _asr_model = whisperx.load_model("small", device="cpu")
    return _asr_model

def transcribe_and_align(audio_path, language_code):
    model = get_asr_model()
    result = model.transcribe(audio_path, language=language_code)
    # Return plain text draft plus segments for future styling/burn-in
    text = " ".join(seg["text"].strip() for seg in result["segments"])
    return result["segments"], text

# ---------- UI logic (with right-side settings) ----------
FONT_CHOICES = [
    "Default", "Arial", "Roboto", "Open Sans", "Lato", "Noto Sans", "Montserrat",
]
LANG_CHOICES = [
    ("Auto-detect", "auto"),
    ("English (en)", "en"),
    ("Spanish (es)", "es"),
    ("French (fr)", "fr"),
    ("German (de)", "de"),
    ("Italian (it)", "it"),
    ("Portuguese (pt)", "pt"),
    ("Russian (ru)", "ru"),
    ("Chinese (zh)", "zh"),
    ("Cantonese (yue)", "yue"),
    ("Japanese (ja)", "ja"),
    ("Korean (ko)", "ko"),
    ("Hindi (hi)", "hi"),
]

def main_interface(audio, language_ui, font_family, font_size, text_color, outline_color, outline_w):
    if audio is None:
        raise gr.Error("Please upload an audio/video file.")
    lang_code = normalize_lang(language_ui)
    segments, text = transcribe_and_align(audio, lang_code)
    # Preview-only CSS style block for the transcript (burn-in not shown here)
    styled = f"""
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
                font_family = gr.Dropdown(FONT_CHOICES, value="Default", label="Font")
                font_size   = gr.Slider(14, 72, value=32, step=1, label="Font size")
                text_color  = gr.ColorPicker(value="#FFFFFF", label="Text color")
                outline_color = gr.ColorPicker(value="#000000", label="Outline color")
                outline_w   = gr.Slider(0, 4, value=2, step=1, label="Outline width (px)")

    run.click(
        main_interface,
        inputs=[audio, language, font_family, font_size, text_color, outline_color, outline_w],
        outputs=[transcript_html, segments_json]
    )

if __name__ == "__main__":
    demo.launch()
