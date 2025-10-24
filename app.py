import os, re, gradio as gr
import torch
import whisperx

# ---------- Language helpers ----------
LANG_MAP = {
    "auto": None, "auto-detect": None, "automatic": None,
    "english": "en", "spanish": "es", "chinese": "zh", "mandarin": "zh",
    "cantonese": "yue", "french": "fr", "german": "de", "italian": "it",
    "japanese": "ja", "korean": "ko", "portuguese": "pt", "russian": "ru",
    "arabic": "ar", "hindi": "hi", "bengali": "bn", "urdu": "ur",
}

def normalize_lang(s: str | None):
    if not s:
        return None
    t = s.strip().lower()
    if t in LANG_MAP:
        return LANG_MAP[t]
    m = re.search(r"\b([a-z]{2,3})\b", t)
    if m:
        return m.group(1)
    tok = t.split()[0]
    if tok in LANG_MAP:
        return LANG_MAP[tok]
    if 2 <= len(tok) <= 3:
        return tok
    return None

# ---------- ASR model (CPU-safe) ----------
_asr_model = None

def get_asr_model():
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"  # key: int8 on CPU

    try:
        _asr_model = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        # Fallback for CPUs without efficient int8
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

# ---------- Styling / Preview ----------
def hex_to_rgb(hex_str: str):
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def rgba(hex_str: str, alpha: float):
    r, g, b = hex_to_rgb(hex_str)
    alpha = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r}, {g}, {b}, {alpha})"

def render_preview(text,
                   font_family, font_size,
                   text_color, outline_color, outline_w,
                   bg_color, bg_opacity):
    if text is None:
        text = ""
    family = "inherit" if font_family == "Default" else font_family
    bg_rgba = rgba(bg_color, bg_opacity)

    html = f"""
<div class="subtitle-preview" style="
  display:inline-block;
  padding:12px 16px;
  border-radius:10px;
  background:{bg_rgba};
">
  <div style="
    font-family:{family}, sans-serif;
    font-size:{int(font_size)}px;
    line-height:1.35;
    color:{text_color} !important;
    text-shadow:
      -{outline_w}px 0 {outline_color},
       {outline_w}px 0 {outline_color},
       0 -{outline_w}px {outline_color},
       0  {outline_w}px {outline_color};
  ">
    {text}
  </div>
</div>
"""
    return html

# ---------- UI ----------
FONT_CHOICES = ["Default", "Arial", "Roboto", "Open Sans", "Lato", "Noto Sans", "Montserrat"]
LANG_CHOICES = [
    ("Auto-detect", "auto"),
    ("English (en)", "en"), ("Spanish (es)", "es"), ("French (fr)", "fr"),
    ("German (de)", "de"), ("Italian (it)", "it"), ("Portuguese (pt)", "pt"),
    ("Russian (ru)", "ru"), ("Chinese (zh)", "zh"), ("Cantonese (yue)", "yue"),
    ("Japanese (ja)", "ja"), ("Korean (ko)", "ko"), ("Hindi (hi)", "hi"),
]

custom_css = """
/* widen app + ensure inline colors win */
.gradio-container { max-width: 1200px !important; }
.settings-card { position: sticky; top: 12px; border-radius: 16px; }
.subtitle-preview * { color: inherit !important; }  /* stop theme from overriding text color */
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🎨 Colorvideo Subs\nTranscribe + style subtitles. Pick fonts & sizes on the right →")

    # Keep the last transcript so style controls can live-update
    last_text = gr.State("")
    last_segments = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            audio = gr.Audio(label="Audio or Video", type="filepath")
            language = gr.Dropdown(choices=LANG_CHOICES, value="auto", label="Language")
            run = gr.Button("Run", variant="primary")

            transcript_html = gr.HTML(label="Preview")
            segments_json = gr.JSON(label="Segments (export/debug)")

        with gr.Column(scale=1):
            with gr.Group(elem_classes=["settings-card"]):
                gr.Markdown("### Subtitle Settings")
                font_family = gr.Dropdown(FONT_CHOICES, value="Default", label="Font")
                font_size   = gr.Slider(14, 72, value=32, step=1, label="Font size")

                text_color     = gr.ColorPicker(value="#FFFFFF", label="Text color")
                outline_color  = gr.ColorPicker(value="#000000", label="Outline color")
                outline_w      = gr.Slider(0, 4, value=2, step=1, label="Outline width (px)")

                gr.Markdown("#### Background")
                bg_color   = gr.ColorPicker(value="#000000", label="Background color")
                bg_opacity = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Background opacity")

    # --- Actions ---
    def do_transcribe(audio_path, language_ui):
        if audio_path is None:
            raise gr.Error("Please upload an audio/video file.")
        lang_code = normalize_lang(language_ui)
        segments, text = transcribe_and_align(audio_path, lang_code)
        html = render_preview(
            text, font_family.value, font_size.value,
            text_color.value, outline_color.value, outline_w.value,
            bg_color.value, bg_opacity.value
        )
        return html, segments, text, segments

    run.click(
        do_transcribe,
        inputs=[audio, language],
        outputs=[transcript_html, segments_json, last_text, last_segments],
    )

    # Live style updates (no need to press Run)
    def update_style(text, ff, fs, tc, oc, ow, bgc, bgo):
        return render_preview(text, ff, fs, tc, oc, ow, bgc, bgo)

    for ctrl in [font_family, font_size, text_color, outline_color, outline_w, bg_color, bg_opacity]:
        ctrl.change(
            update_style,
            inputs=[last_text, font_family, font_size, text_color, outline_color, outline_w, bg_color, bg_opacity],
            outputs=[transcript_html],
            show_progress=False,
        )

if __name__ == "__main__":
    demo.launch()
