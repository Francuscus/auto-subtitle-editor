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
    compute = "float16" if use_cuda else "int8"   # int8 on CPU avoids the float16 crash

    try:
        _asr_model = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        # safety fallback for CPUs without int8/AVX2 etc.
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
    """
    Returns raw segments from WhisperX (each has start, end, text).
    We do NOT request word-level timestamps (keeps it fast/stable).
    """
    model = get_asr_model()
    result = model.transcribe(audio_path, language=language_code)
    segments = result.get("segments", [])
    return segments

# ----------------------------
# Chunking into ~N words
# ----------------------------
WORD_RE = re.compile(r"\S+")

def split_segments_by_words(segments, max_words=5):
    """
    Split each ASR segment into ~N-word mini-chunks.
    Timestamps are distributed linearly within the original segment.
    """
    max_words = max(1, int(max_words))
    out = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        # tokenize by "words"
        words = WORD_RE.findall(text)
        if not words:
            continue

        # make groups of <= max_words
        groups = [words[i:i+max_words] for i in range(0, len(words), max_words)]
        total_groups = len(groups)

        # linear time slicing across the segment
        dur = max(0.0, end - start)
        for idx, group in enumerate(groups):
            chunk_text = " ".join(group).strip()
            # proportions
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
        lines.append("")  # blank line
    return "\n".join(lines).strip() + "\n"

def format_ass(chunks, font="Noto Sans", size=48, primary="#FFFFFF", outline="#000000", outline_w=2, shadow=0):
    # convert #RRGGBB -> &HBBGGRR& (ASS BGR hex with &H..&)
    def ass_color(hex_rgb):
        hex_rgb = hex_rgb.strip().lstrip("#")
        if len(hex_rgb) != 6:
            hex_rgb = "FFFFFF"
        rr, gg, bb = hex_rgb[0:2], hex_rgb[2:4], hex_rgb[4:6]
        return f"&H{bb}{gg}{rr}&"

    ass_primary = ass_color(primary)
    ass_outline = ass_color(outline)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1920\n"
        "PlayResY: 1080\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{size},{ass_primary},&H000000FF,{ass_outline},&H00000000,"
        f"0,0,0,0,100,100,0,0,1,{outline_w},{shadow},2,120,120,80,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [header]
    for c in chunks:
        start = srt_timestamp(c["start"]).replace(",", ".")
        end = srt_timestamp(c["end"]).replace(",", ".")
        txt = c["text"].replace("\n", "\\N")
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{txt}")
    return "\n".join(lines) + "\n"

# ----------------------------
# Export helpers (files for DownloadButton)
# ----------------------------
def write_temp_file(content: str, suffix: str):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path

# ----------------------------
# Pipeline for the UI
# ----------------------------
def run_pipeline(audio_path, lang_ui, words_per_chunk, font_family, font_size, text_color, outline_color, outline_w):
    if not audio_path:
        raise gr.Error("Please upload an audio or video file.")
    lang_code = normalize_lang(lang_ui)
    raw_segments = transcribe(audio_path, lang_code)
    chunks = split_segments_by_words(raw_segments, max_words=words_per_chunk)

    # preview HTML (styled)
    combined_text = " ".join([c["text"] for c in chunks])
    preview_html = f"""
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
    ">
    {combined_text}
    </div>
    """

    # also keep segments JSON for debugging/export
    return preview_html, json.dumps(chunks, ensure_ascii=False, indent=2)

def make_srt(json_chunks):
    if not json_chunks:
        raise gr.Error("Run a transcription first.")
    chunks = json.loads(json_chunks)
    srt = format_srt(chunks)
    return write_temp_file(srt, ".srt")

def make_ass(json_chunks, font_family, font_size, text_color, outline_color, outline_w):
    if not json_chunks:
        raise gr.Error("Run a transcription first.")
    chunks = json.loads(json_chunks)
    ass = format_ass(
        chunks,
        font=(None if font_family == "Default" else font_family) or "Noto Sans",
        size=int(font_size),
        primary=text_color,
        outline=outline_color,
        outline_w=int(outline_w),
        shadow=0
    )
    return write_temp_file(ass, ".ass")

# ----------------------------
# Gradio UI
# ----------------------------
FONT_CHOICES = ["Default", "Arial", "Roboto", "Open Sans", "Lato", "Noto Sans", "Montserrat"]

custom_css = """
/* obvious visual difference so you know this build is live */
body { background: #0a0f1a; }                 /* deep navy */
.gradio-container { max-width: 1200px !important; }
footer { display:none !important; }
h1, h2, .prose h1, .prose h2 { color: #e6f0ff !important; }   /* light title */
label, .text-gray-600, .prose :where(p):not(:where(.not-prose *)) { color:#c9d6ff !important; } /* lighter labels */
.settings-card { position: sticky; top: 12px; border-radius: 16px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🎯 Five-Word Subtitle Chunker\nTranscribe + split into small chunks, then export SRT/ASS.")

    with gr.Row():
        with gr.Column(scale=3):
            audio = gr.Audio(label="Audio or Video", type="filepath")
            language = gr.Dropdown(choices=LANG_CHOICES, value="auto", label="Language")
            words_per_chunk = gr.Slider(1, 10, value=5, step=1, label="Words per chunk (max)")
            run = gr.Button("Run", variant="primary")

            gr.Markdown("### Preview")
            preview_html = gr.HTML()

            gr.Markdown("### Chunks (JSON)")
            chunks_json = gr.Textbox(lines=10, show_label=False)

        with gr.Column(scale=1):
            with gr.Group(elem_classes=["settings-card"]):
                gr.Markdown("### Subtitle Style")
                font_family = gr.Dropdown(FONT_CHOICES, value="Default", label="Font")
                font_size   = gr.Slider(14, 72, value=36, step=1, label="Font size")
                text_color  = gr.ColorPicker(value="#FFFFFF", label="Text color")
                outline_color = gr.ColorPicker(value="#000000", label="Outline color")
                outline_w   = gr.Slider(0, 4, value=2, step=1, label="Outline width (px)")

                gr.Markdown("---")
                gr.Markdown("### Export")
                srt_btn = gr.DownloadButton("Download SRT")
                ass_btn = gr.DownloadButton("Download ASS")

    # wire up actions
    run.click(
        run_pipeline,
        inputs=[audio, language, words_per_chunk, font_family, font_size, text_color, outline_color, outline_w],
        outputs=[preview_html, chunks_json]
    )

    srt_btn.click(
        make_srt,
        inputs=[chunks_json],
        outputs=[srt_btn]
    )

    ass_btn.click(
        make_ass,
        inputs=[chunks_json, font_family, font_size, text_color, outline_color, outline_w],
        outputs=[ass_btn]
    )

if __name__ == "__main__":
    demo.launch()
