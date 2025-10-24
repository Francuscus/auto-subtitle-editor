import os, re, tempfile, json
import gradio as gr
import torch
import whisperx
import pysubs2

# ---------------- helpers: language ----------------
LANG_MAP = {
    "auto": None, "auto-detect": None, "automatic": None,
    "english": "en", "spanish": "es", "hungarian": "hu",
    "chinese": "zh", "mandarin": "zh", "cantonese": "yue",
    "french": "fr", "german": "de", "italian": "it",
    "japanese": "ja", "korean": "ko", "portuguese": "pt",
    "russian": "ru", "arabic": "ar", "hindi": "hi", "bengali": "bn",
    "urdu": "ur",
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

# ---------------- model cache ----------------
_asr_model = None

def get_asr_model():
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"  # CPU-safe default

    try:
        _asr_model = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        # Fallback if int8 kernels aren’t available on this CPU
        if "compute type" in str(e).lower():
            fallback = "int16" if device == "cpu" else "float32"
            _asr_model = whisperx.load_model("small", device=device, compute_type=fallback)
        else:
            raise
    return _asr_model

# ---------------- transcription ----------------
def transcribe(audio_path, language_code):
    model = get_asr_model()
    # Ask for word timestamps if backend supports it; WhisperX passes through to faster-whisper
    result = model.transcribe(audio_path, language=language_code, word_timestamps=True)
    # Join plain text for preview
    text = " ".join(seg["text"].strip() for seg in result["segments"])
    return result["segments"], text

# ---------------- lyrics retiming (simple) ----------------
def split_lyrics(text: str):
    # split on new lines; if user pasted a paragraph, split on sentences
    raw_lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(raw_lines) >= 2:
        return raw_lines
    # fallback: sentence-ish split
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def retime_lyrics_to_segments(lyrics_lines, segments):
    """Map each lyric line to the available segment durations (greedy, linear).
       Keeps segment boundaries; multiple lines may share a segment time if fewer segments."""
    if not segments or not lyrics_lines:
        return segments

    # Collect segment timing
    seg_times = [(s["start"], s["end"]) for s in segments]
    seg_durs = [max(0.01, e - st) for st, e in seg_times]
    total_dur = sum(seg_durs)

    # Build a cumulative “timeline” of segment bins
    bins = []
    acc = 0.0
    for dur in seg_durs:
        bins.append((acc, acc+dur))
        acc += dur

    # Divide total duration into N lyric slices
    N = len(lyrics_lines)
    slice_edges = [i * (total_dur / N) for i in range(N+1)]

    def time_in_bins(t):
        # map absolute time in [0,total_dur] back to real [start,end] using bins
        # find which bin contains t
        for idx,(b0,b1) in enumerate(bins):
            if b0 <= t <= b1 or (idx==len(bins)-1 and t> b1):
                # fraction within this bin
                frac = 0.0 if b1==b0 else (t - b0) / (b1 - b0)
                real_start, real_end = seg_times[idx]
                return real_start + frac*(real_end - real_start)
        # fallback: end of last seg
        return seg_times[-1][1]

    # Build retimed “segments-like” dicts for lyrics
    out = []
    for i in range(N):
        t0 = time_in_bins(slice_edges[i])
        t1 = time_in_bins(slice_edges[i+1])
        out.append({
            "id": i,
            "start": float(min(t0,t1)),
            "end": float(max(t0,t1)),
            "text": lyrics_lines[i]
        })
    return out

# ---------------- export: SRT / ASS ----------------
def segments_to_srt(segments):
    def fmt_time(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int(round((t - int(t)) * 1000))
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    lines = []
    for i, s in enumerate(segments, start=1):
        start = max(0.0, float(s["start"]))
        end = max(start, float(s["end"]))
        text = s.get("text", "").strip()
        lines.append(str(i))
        lines.append(f"{fmt_time(start)} --> {fmt_time(end)}")
        lines.append(text)
        lines.append("")  # blank
    return "\n".join(lines)

def segments_to_ass(segments, font, size_px, text_color, outline_color, outline_w):
    # Convert HTML colors (#RRGGBB) to ASS BGR format (&HBBGGRR&)
    def hex_to_ass(c):
        c = c.lstrip("#")
        if len(c) == 3:
            c = "".join([ch*2 for ch in c])
        # ASS is BGR and &HAABBGGRR (we’ll keep alpha opaque AA=00)
        rr = c[0:2]; gg = c[2:4]; bb = c[4:6]
        return f"&H00{bb.upper()}{gg.upper()}{rr.upper()}"

    subs = pysubs2.SSAFile()
    subs.info["Collisions"] = "Normal"
    # Create a style
    style = pysubs2.SSAStyle(
        fontname=("Arial" if font == "Default" else font),
        fontsize=int(size_px),
        primarycolor=hex_to_ass(text_color),
        outlinecolor=hex_to_ass(outline_color),
        backcolor="&H00000000",
        bold=False, italic=False, underline=False, strikeout=False,
        scale_x=100, scale_y=100, spacing=0, angle=0,
        borderstyle=1,  # opaque box=3; outline=1
        outline=int(outline_w),
        shadow=0,
        alignment=2,  # bottom-center
        marginl=10, marginr=10, marginv=20,
        encoding=1
    )
    subs.styles["Default"] = style

    for s in segments:
        start_ms = int(max(0.0, float(s["start"])) * 1000)
        end_ms = int(max(float(s["start"]), float(s["end"])) * 1000)
        line = pysubs2.SSAEvent(
            start=start_ms,
            end=end_ms,
            text=s.get("text", "").strip(),
            style="Default"
        )
        subs.events.append(line)

    return subs.to_string("ass")

def export_files(segments, font, size_px, text_color, outline_color, outline_w):
    srt_str = segments_to_srt(segments)
    ass_str = segments_to_ass(segments, font, size_px, text_color, outline_color, outline_w)

    srt_path = tempfile.mkstemp(suffix=".srt")[1]
    ass_path = tempfile.mkstemp(suffix=".ass")[1]
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_str)
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_str)
    return srt_path, ass_path

# ---------------- UI callbacks ----------------
def run_pipeline(audio, language_ui, font_family, font_size, text_color, outline_color, outline_w, lyrics_text, use_lyrics):
    if audio is None:
        raise gr.Error("Please upload an audio or video file (mp3/wav/mp4).")

    lang_code = normalize_lang(language_ui)

    segments, text = transcribe(audio, lang_code)

    if use_lyrics and lyrics_text and lyrics_text.strip():
        lyric_lines = split_lyrics(lyrics_text)
        # replace segments with retimed lyric lines
        segments = retime_lyrics_to_segments(lyric_lines, segments)
        text = "\n".join(l for l in lyric_lines)

    # Build preview HTML
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
    srt_path, ass_path = export_files(segments, font_family, font_size, text_color, outline_color, outline_w)
    return styled, segments, srt_path, ass_path

# ---------------- UI ----------------
FONT_CHOICES = [
    "Default", "Arial", "Roboto", "Open Sans", "Lato", "Noto Sans", "Montserrat",
]
LANG_CHOICES = [
    ("Auto-detect", "auto"),
    ("Hungarian (hu)", "hu"),
    ("Spanish (es)", "es"),
    ("English (en)", "en"),
]

custom_css = """
/* Wider container and clear contrast */
.gradio-container { max-width: 1220px !important; }
body { background: #0e1116; }
#app-title { color: #f2f5f9 !important; }
.card { background: #141922; border-radius: 16px; border: 1px solid #1f2633; }
.card h3, .card label, .card p { color: #e7ecf3 !important; }
.preview-box { background: #0b0f15; border-radius: 12px; border: 1px solid #1b2330; padding: 12px; }
.json-box { background: #0b0f15; border-radius: 12px; border: 1px solid #1b2330; }
.settings-card { position: sticky; top: 12px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🎬 **Colorvideo Subs**", elem_id="app-title")
    gr.Markdown(
        "Transcribe + style subtitles. Export **SRT/ASS**. Optional: paste lyrics to retime.",
        elem_classes=["card"], padding=12
    )

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group(elem_classes=["card"]):
                audio = gr.Audio(label="Audio / Video (mp3, wav, mp4)", type="filepath")
                language = gr.Dropdown(choices=LANG_CHOICES, value="auto", label="Language")
                run = gr.Button("Run", variant="primary")

            with gr.Group(elem_classes=["card"]):
                gr.Markdown("### Preview")
                transcript_html = gr.HTML(label="Preview", elem_classes=["preview-box"])
                gr.Markdown("### Segments")
                segments_json = gr.JSON(label="Segments", elem_classes=["json-box"])

            with gr.Group(elem_classes=["card"]):
                gr.Markdown("### Export")
                srt_file = gr.File(label="Download SRT", interactive=False)
                ass_file = gr.File(label="Download ASS", interactive=False)

        with gr.Column(scale=1):
            with gr.Group(elem_classes=["card", "settings-card"]):
                gr.Markdown("### Subtitle Settings")
                font_family = gr.Dropdown(FONT_CHOICES, value="Default", label="Font")
                font_size   = gr.Slider(14, 72, value=34, step=1, label="Font size")
                text_color  = gr.ColorPicker(value="#FFFFFF", label="Text color")
                outline_color = gr.ColorPicker(value="#000000", label="Outline color")
                outline_w   = gr.Slider(0, 4, value=2, step=1, label="Outline width (px)")

            with gr.Group(elem_classes=["card"]):
                gr.Markdown("### Optional Lyrics (retime)")
                use_lyrics = gr.Checkbox(label="Use pasted lyrics to retime", value=False)
                lyrics_text = gr.Textbox(
                    label="Paste lyrics or lines here",
                    placeholder="One line per subtitle… or paste a paragraph and I’ll split into sentences.",
                    lines=8
                )

    run.click(
        run_pipeline,
        inputs=[audio, language, font_family, font_size, text_color, outline_color, outline_w, lyrics_text, use_lyrics],
        outputs=[transcript_html, segments_json, srt_file, ass_file]
    )

if __name__ == "__main__":
    demo.launch()
