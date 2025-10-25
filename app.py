# app.py
import os, re, tempfile, json
import gradio as gr
import torch
import whisperx

# ----------------------------
# Language helpers (Auto/HU/ES)
# ----------------------------
LANG_CHOICES = [
    ("Auto-detect", "auto"),
    ("Hungarian (hu)", "hu"),
    ("Spanish (es)", "es"),
]
LANG_MAP = {
    "auto": None, "auto-detect": None, "automatic": None,
    "hungarian": "hu", "hu": "hu",
    "spanish": "es", "es": "es",
}
def normalize_lang(s: str | None):
    if not s:
        return None
    t = s.strip().lower()
    if t in LANG_MAP:
        return LANG_MAP[t]
    m = re.search(r"\b([a-z]{2,3})\b", t)
    if m:
        code = m.group(1)
        return LANG_MAP.get(code, code)
    return None

# ----------------------------
# Load WhisperX (with fallback)
# ----------------------------
_asr_model = None
def get_asr_model():
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"   # int8 on CPU avoids float16 crash

    try:
        _asr_model = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        if "compute type" in str(e).lower():
            fallback = "int16" if device == "cpu" else "float32"
            _asr_model = whisperx.load_model("small", device=device, compute_type=fallback)
        else:
            raise
    return _asr_model

# ----------------------------
# ASR
# ----------------------------
def transcribe(audio_path, language_code):
    model = get_asr_model()
    result = model.transcribe(audio_path, language=language_code)
    segments = result.get("segments", [])
    return segments

# ----------------------------
# Chunking into ~N words
# ----------------------------
WORD_RE = re.compile(r"\S+")

def split_segments_by_words(segments, max_words=5):
    max_words = max(1, int(max_words))
    out = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        words = WORD_RE.findall(text)
        if not words:
            continue

        groups = [words[i:i+max_words] for i in range(0, len(words), max_words)]
        total_groups = len(groups)
        dur = max(0.0, end - start)

        for idx, group in enumerate(groups):
            chunk_text = " ".join(group).strip()
            g0 = idx / total_groups
            g1 = (idx + 1) / total_groups
            c_start = start + g0 * dur
            c_end = start + g1 * dur
            out.append({
                "start": round(c_start, 3),
                "end": round(c_end, 3),
                "text": chunk_text
            })
    return out

# ----------------------------
# Formatters (SRT / ASS)
# ----------------------------
def srt_timestamp(t):
    ms = int(round((t - int(t)) * 1000.0))
    s = int(t) % 60
    m = (int(t) // 60) % 60
    h = int(t) // 3600
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def format_srt(chunks):
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(str(i))
        lines.append(f"{srt_timestamp(c['start'])} --> {srt_timestamp(c['end'])}")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines).strip() + "\n"

def _ass_hex(hex_rgb):
    # ASS needs &HBBGGRR&
    hex_rgb = (hex_rgb or "").strip().lstrip("#")
    if len(hex_rgb) != 6:
        hex_rgb = "FFFFFF"
    rr, gg, bb = hex_rgb[0:2], hex_rgb[2:4], hex_rgb[4:6]
    return f"&H{bb}{gg}{rr}&"

def format_ass(
    chunks,
    playres_x=1920, playres_y=1080,
    font="Noto Sans", size=48,
    primary="#FFFFFF", outline="#000000", outline_w=2,
    bg_box=False, bg_color="#111111"
):
    ass_primary = _ass_hex(primary)
    ass_outline = _ass_hex(outline)
    ass_back    = _ass_hex(bg_color)

    # BorderStyle 1 = outline/glow; 3 = opaque box behind text
    border_style = 3 if bg_box else 1
    outline_val  = 0 if bg_box else outline_w
    shadow_val   = 0

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {playres_x}\n"
        f"PlayResY: {playres_y}\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{size},{ass_primary},&H000000FF,{ass_outline},{ass_back},"
        f"0,0,0,0,100,100,0,0,{border_style},{outline_val},{shadow_val},2,120,120,80,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = [header]
    for c in chunks:
        start = srt_timestamp(c["start"]).replace(",", ".")
        end   = srt_timestamp(c["end"]).replace(",", ".")
        txt   = c["text"].replace("\n", "\\N")
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{txt}")
    return "\n".join(lines) + "\n"

# ----------------------------
# Export helpers
# ----------------------------
def write_temp_file(content: str, suffix: str):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path

# ----------------------------
# Pipeline
# ----------------------------
def run_pipeline(audio_path, lang_ui, words_per_chunk,
                 layout_choice, font_family, font_size,
                 text_color, outline_color, outline_w,
                 bg_box, bg_color):
    if not audio_path:
        raise gr.Error("Please upload an audio or video file.")

    # layout preset
    if layout_choice == "YouTube 16:9 (1920×1080)":
        playres_x, playres_y = 1920, 1080
        aspect = 16/9
    else:
        playres_x, playres_y = 1080, 1920
        aspect = 9/16

    lang_code = normalize_lang(lang_ui)
    raw_segments = transcribe(audio_path, lang_code)
    chunks = split_segments_by_words(raw_segments, max_words=words_per_chunk)

    # preview (in an aspect-ratio box)
    combined_text = " ".join([c['text'] for c in chunks])

    # Create an aspect-ratio container using padding-top trick
    # and center the text visually
    padding_pct = 100 / aspect
    preview_html = f"""
<div style="position:relative;width:100%;background:#0f172a;border-radius:12px;overflow:hidden;">
  <div style="width:100%;padding-top:{padding_pct}%;"></div>
  <div style="
    position:absolute;inset:0;display:flex;align-items:flex-end;justify-content:center;
    padding:24px;
    ">
    <div style="
      max-width:90%;
      text-align:center;
      font-family:{'inherit' if font_family=='Default' else font_family}, sans-serif;
      font-size:{int(font_size)}px;
      line-height:1.35;
      color:{text_color};
      {'background:'+bg_color+'; padding:8px 12px; border-radius:8px;' if bg_box else ''}
      text-shadow:
        -{outline_w}px 0 {outline_color},
         {outline_w}px 0 {outline_color},
         0 -{outline_w}px {outline_color},
         0  {outline_w}px {outline_color};
    ">
      {combined_text}
    </div>
  </div>
</div>
"""

    return (preview_html,
            json.dumps(chunks, ensure_ascii=False, indent=2),
            playres_x, playres_y)

def make_srt(json_chunks):
    if not json_chunks:
        raise gr.Error("Run a transcription first.")
    chunks = json.loads(json_chunks)
    srt = format_srt(chunks)
    return write_temp_file(srt, ".srt")

def make_ass(json_chunks, layout_choice, font_family, font_size,
             text_color, outline_color, outline_w, bg_box, bg_color):
    if not json_chunks:
        raise gr.Error("Run a transcription first.")
    chunks = json.loads(json_chunks)

    if layout_choice == "YouTube 16:9 (1920×1080)":
        playres_x, playres_y = 1920, 1080
    else:
        playres_x, playres_y = 1080, 1920

    ass = format_ass(
        chunks,
        playres_x=playres_x, playres_y=playres_y,
        font=(None if font_family == "Default" else font_family) or "Noto Sans",
        size=int(font_size),
        primary=text_color,
        outline=outline_color,
        outline_w=int(outline_w),
        bg_box=bool(bg_box),
        bg_color=bg_color
    )
    return write_temp_file(ass, ".ass")

# ----------------------------
# Gradio UI
# ----------------------------
FONT_CHOICES = ["Default", "Arial", "Roboto", "Open Sans", "Lato", "Noto Sans", "Montserrat"]
LAYOUT_CHOICES = ["YouTube 16:9 (1920×1080)", "Phone/TikTok 9:16 (1080×1920)"]

custom_css = """
/* make it obvious this is the new build */
body { background: #07111f; }
.gradio-container { max-width: 1200px !important; }
footer { display:none !important; }
h1, h2, .prose h1, .prose h2 { color: #e8f0ff !important; }
label, .text-gray-600, .prose :where(p):not(:where(.not-prose *)) { color:#d4e0ff !important; }
.settings-card { position: sticky; top: 12px; border-radius: 16px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🎬 Five-Word Subtitle Chunker + Layout/Background")
    with gr.Row():
        with gr.Column(scale=3):
            audio = gr.Audio(label="Audio or Video", type="filepath")
            language = gr.Dropdown(choices=LANG_CHOICES, value="auto", label="Language")
            words_per_chunk = gr.Slider(1, 10, value=5, step=1, label="Words per chunk (max)")
            layout_choice = gr.Dropdown(LAYOUT_CHOICES, value=LAYOUT_CHOICES[0], label="Layout preset")
            run = gr.Button("Run", variant="primary")

            gr.Markdown("### Preview")
            preview_html = gr.HTML()

            gr.Markdown("### Chunks (JSON)")
            chunks_json = gr.Textbox(lines=10, show_label=False)

            # invisible holders for PlayRes from pipeline
            playres_x_box = gr.Number(visible=False)
            playres_y_box = gr.Number(visible=False)

        with gr.Column(scale=1):
            with gr.Group(elem_classes=["settings-card"]):
                gr.Markdown("### Subtitle Style")
                font_family   = gr.Dropdown(FONT_CHOICES, value="Default", label="Font")
                font_size     = gr.Slider(14, 72, value=36, step=1, label="Font size")
                text_color    = gr.ColorPicker(value="#FFFFFF", label="Text color")
                outline_color = gr.ColorPicker(value="#000000", label="Outline color")
                outline_w     = gr.Slider(0, 4, value=2, step=1, label="Outline width (px)")

                gr.Markdown("---")
                gr.Markdown("### Background")
                bg_box   = gr.Checkbox(value=True, label="Boxed background behind text")
                bg_color = gr.ColorPicker(value="#111111", label="Background color")

                gr.Markdown("---")
                gr.Markdown("### Export")
                srt_btn = gr.DownloadButton("Download SRT")
                ass_btn = gr.DownloadButton("Download ASS")

    run.click(
        run_pipeline,
        inputs=[audio, language, words_per_chunk,
                layout_choice, font_family, font_size,
                text_color, outline_color, outline_w, bg_box, bg_color],
        outputs=[preview_html, chunks_json, playres_x_box, playres_y_box]
    )

    srt_btn.click(
        make_srt,
        inputs=[chunks_json],
        outputs=[srt_btn]
    )

    ass_btn.click(
        make_ass,
        inputs=[chunks_json, layout_choice, font_family, font_size,
                text_color, outline_color, outline_w, bg_box, bg_color],
        outputs=[ass_btn]
    )

if __name__ == "__main__":
    demo.launch()
