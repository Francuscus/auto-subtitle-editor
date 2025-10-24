# app.py
import re
import gradio as gr
import torch
import whisperx

# ----- Helpers -----
LANG_MAP = {"auto": None, "hungarian": "hu", "spanish": "es"}

def normalize_lang(s: str | None):
    if not s:
        return None
    t = s.strip().lower()
    if t in LANG_MAP:
        return LANG_MAP[t]
    # accept labels like "es (Spanish)" or plain "es"/"hu"
    m = re.search(r"\b([a-z]{2,3})\b", t)
    return m.group(1) if m else None

_asr_model = None

def get_asr_model():
    """Load once; safe compute types for CPU/GPU."""
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"  # CPU-safe default

    try:
        _asr_model = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        # Fallback if the CPU lacks int8 kernels, or any compute mismatch
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

# ----- UI -----
FONT_CHOICES = ["Default", "Arial", "Roboto", "Open Sans", "Lato", "Montserrat"]
LANG_CHOICES = [("Auto-detect", "auto"), ("Spanish (es)", "es"), ("Hungarian (hu)", "hu")]

def main_interface(audio, language_ui, font_family, font_size,
                   text_color, outline_color, outline_w, bg_color):
    if audio is None:
        raise gr.Error("Please upload an audio/video file (mp3, wav, mp4, etc.).")
    lang_code = normalize_lang(language_ui)
    segments, text = transcribe_and_align(audio, lang_code)

    styled = f"""
<div style="background:{bg_color}; padding:16px; border-radius:12px;">
  <div style="
    font-family:{'inherit' if font_family=='Default' else font_family}, sans-serif;
    font-size:{int(font_size)}px; line-height:1.35; color:{text_color};
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
    return styled, segments

custom_css = """
.gradio-container { max-width: 1200px !important; }
.settings-card { position: sticky; top: 12px; border-radius: 16px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🎨 Colorvideo Subs\nUpload → transcribe → style preview (Spanish & Hungarian focus)")

    with gr.Row():
        with gr.Column(scale=3):
            audio = gr.Audio(label="Audio or Video", type="filepath")
            language = gr.Dropdown(choices=LANG_CHOICES, value="auto", label="Language (ASR)")
            run = gr.Button("Transcribe", variant="primary")

            transcript_html = gr.HTML(label="Styled Preview")
            segments_json = gr.JSON(label="Segments (debug/export)")

        with gr.Column(scale=1):
            with gr.Group(elem_classes=["settings-card"]):
                gr.Markdown("### Subtitle Settings")
                font_family = gr.Dropdown(FONT_CHOICES, value="Default", label="Font")
                font_size   = gr.Slider(14, 72, value=32, step=1, label="Font size")
                text_color  = gr.ColorPicker(value="#FFFFFF", label="Text color")
                outline_color = gr.ColorPicker(value="#000000", label="Outline color")
                outline_w   = gr.Slider(0, 4, value=2, step=1, label="Outline width (px)")
                bg_color    = gr.ColorPicker(value="#1A1A1A", label="Preview background")

    run.click(
        main_interface,
        inputs=[audio, language, font_family, font_size, text_color, outline_color, outline_w, bg_color],
        outputs=[transcript_html, segments_json]
    )

if __name__ == "__main__":
    demo.launch()
