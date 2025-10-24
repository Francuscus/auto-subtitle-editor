# app.py
import os, re, io, tempfile
import difflib
from typing import List, Tuple, Optional

import gradio as gr
import torch
from faster_whisper import WhisperModel


# ---------------------------
# Language helpers (Spanish + Hungarian + Auto by default; more can be added)
# ---------------------------
LANG_CHOICES = [
    ("Auto-detect", "auto"),
    ("Spanish (es)", "es"),
    ("Hungarian (hu)", "hu"),
    ("English (en)", "en"),
]
LANG_MAP = {v: v for _, v in LANG_CHOICES if v != "auto"}
NAME_TO_CODE = {
    "auto": None, "auto-detect": None, "automatic": None,
    "spanish": "es", "hungarian": "hu", "english": "en"
}
def normalize_lang(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = s.strip().lower()
    if t in NAME_TO_CODE:
        return NAME_TO_CODE[t]
    if t in LANG_MAP:
        return t
    m = re.search(r"\b([a-z]{2,3})\b", t)
    return m.group(1) if m else None


# ---------------------------
# ASR model (lazy, CPU-safe)
# ---------------------------
_asr_model: Optional[WhisperModel] = None
def get_model() -> WhisperModel:
    global _asr_model
    if _asr_model:
        return _asr_model

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute_type = "float16" if use_cuda else "int8"  # CPU -> int8

    # beam_size 1-5 is fine; vad_filter reduces silences
    _asr_model = WhisperModel("small", device=device, compute_type=compute_type)
    return _asr_model


# ---------------------------
# Transcription
# ---------------------------
def transcribe_with_words(audio_path: str, lang_code: Optional[str]):
    """
    Returns (segments, full_text).
    segments: list of dicts:
      {
        "start": float, "end": float, "text": str,
        "words": [{"start": float, "end": float, "word": str}, ...]  # may be empty if model returns none
      }
    """
    model = get_model()
    segments_iter, info = model.transcribe(
        audio_path,
        language=lang_code,                # None -> auto
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        word_timestamps=True,              # <-- ask for per-word times
    )

    segments = []
    all_text = []
    for seg in segments_iter:
        words = []
        if seg.words:
            for w in seg.words:
                # faster-whisper can include punctuation tokens; strip outer spaces
                txt = (w.word or "").strip()
                if not txt:
                    continue
                words.append({"start": float(w.start or seg.start),
                              "end": float(w.end or seg.end),
                              "word": txt})
        segments.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": (seg.text or "").strip(),
            "words": words
        })
        all_text.append((seg.text or "").strip())

    return segments, " ".join(all_text).strip()


# ---------------------------
# Forced alignment of user-pasted lyrics
# (Lightweight approach: map provided words to recognized words via sequence matching,
# then group into lines with timings.)
# ---------------------------
def tokenize_words(s: str) -> List[str]:
    # Keep letters/numbers and basic apostrophes, split on whitespace and punctuation
    return re.findall(r"[0-9A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű’']+|-+", s, flags=re.UNICODE)

def collect_recognized_words(segments) -> List[Tuple[str, float, float]]:
    """Flatten recognized words [(word, start, end), ...]."""
    flat = []
    for seg in segments:
        if seg.get("words"):
            for w in seg["words"]:
                flat.append((w["word"], w["start"], w["end"]))
        else:
            # If no word timing, fallback: split text evenly across seg duration
            words = tokenize_words(seg["text"])
            if not words:
                continue
            dur = max(0.0, seg["end"] - seg["start"])
            step = dur / max(1, len(words))
            for i, w in enumerate(words):
                s = seg["start"] + i * step
                e = seg["start"] + (i + 1) * step
                flat.append((w, s, e))
    return flat

def align_lyrics_to_words(lyrics_text: str, rec_words: List[Tuple[str, float, float]]):
    """
    Map lyrics words -> recognized words by content (case-insensitive),
    then produce time-coded lyric lines. Returns list of segments:
      [{"start":..., "end":..., "text":"line text", "words":[{word,start,end},...]}]
    """
    # Build target tokens
    lyrics_lines = [ln.strip() for ln in lyrics_text.strip().splitlines() if ln.strip()]
    lyrics_tokens = [tokenize_words(ln) for ln in lyrics_lines]

    # Flatten lyrics tokens with (line_idx, word_idx_in_line, token)
    L = []
    for li, toks in enumerate(lyrics_tokens):
        for wi, t in enumerate(toks):
            L.append((li, wi, t))

    # Normalize tokens for matching
    def norm(t): return re.sub(r"\W+", "", t, flags=re.UNICODE).lower()

    rec_norm = [norm(w) for (w, _, _) in rec_words]
    lyr_norm = [norm(t) for (_, _, t) in L]

    # Use difflib to map lyric tokens to recognized tokens (greedy-ish)
    sm = difflib.SequenceMatcher(a=rec_norm, b=lyr_norm, autojunk=False)
    mapping = {}  # b_index (lyrics idx) -> a_index (recognized idx)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "replace"):
            # pair aligned spans positionally
            span = min(i2 - i1, j2 - j1)
            for k in range(span):
                mapping[j1 + k] = i1 + k
        # "insert"/"delete" we skip; unmatched lyrics words will be time-interpolated below

    # Build timed lines by walking the lyrics again
    out_segments = []
    cursor = 0
    for li, toks in enumerate(lyrics_tokens):
        words = []
        start_t, end_t = None, None
        for wi, tok in enumerate(toks):
            lyr_flat_idx = cursor + wi
            if lyr_flat_idx in mapping:
                ri = mapping[lyr_flat_idx]
                # clamp index
                if 0 <= ri < len(rec_words):
                    _, s, e = rec_words[ri]
                    words.append({"word": tok, "start": s, "end": e})
                    start_t = s if start_t is None else min(start_t, s)
                    end_t = e if end_t is None else max(end_t, e)
            else:
                # Unmatched: we’ll estimate timing from neighbors later
                words.append({"word": tok, "start": None, "end": None})

        # Fill missing times by interpolating between nearest known neighbors
        # 1) left-to-right fill starts
        last_t = start_t
        for w in words:
            if w["start"] is None and last_t is not None:
                w["start"] = last_t
            if w["end"] is not None:
                last_t = w["end"]
        # 2) right-to-left fill ends
        next_t = end_t
        for w in reversed(words):
            if w["end"] is None and next_t is not None:
                w["end"] = next_t
            if w["start"] is not None:
                next_t = w["start"]
        # 3) still None? use previous/next line bounds (fallback later)
        if start_t is None or end_t is None:
            # fallback to neighbor recognized words window if any
            # find closest mapped rec index around this line range
            mapped_idxs = [mapping.get(cursor + k) for k in range(len(toks)) if (cursor + k) in mapping]
            mapped_idxs = [x for x in mapped_idxs if x is not None]
            if mapped_idxs:
                first = min(mapped_idxs)
                last = max(mapped_idxs)
                start_t = rec_words[first][1]
                end_t = rec_words[last][2]
                span = max(0.01, (end_t - start_t) / max(1, len(words)))
                for i, w in enumerate(words):
                    if w["start"] is None:
                        w["start"] = start_t + i * span
                    if w["end"] is None:
                        w["end"] = start_t + (i + 1) * span
            else:
                # absolute fallback: make a 2s line placed at 0
                start_t = 0.0
                end_t = 2.0
                span = (end_t - start_t) / max(1, len(words))
                for i, w in enumerate(words):
                    w["start"] = start_t + i * span
                    w["end"] = start_t + (i + 1) * span

        out_segments.append({
            "start": float(start_t),
            "end": float(end_t),
            "text": " ".join(toks),
            "words": words
        })
        cursor += len(toks)

    return out_segments


# ---------------------------
# Subtitle exporters
# ---------------------------
def to_srt(segments) -> str:
    def fmt(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{fmt(seg['start'])} --> {fmt(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")  # blank line
    return "\n".join(lines).strip() + "\n"

def ass_color_from_hex(hex_color: str, alpha: int = 0x00) -> str:
    """
    ASS uses &HAABBGGRR (hex) with little-endian RGB and AA alpha (00 opaque, FF transparent).
    Input: '#RRGGBB'
    """
    hex_color = hex_color.strip()
    if hex_color.startswith("#"):
        hex_color = hex_color[1:]
    if len(hex_color) != 6:
        # default white
        hex_color = "FFFFFF"
    rr = int(hex_color[0:2], 16)
    gg = int(hex_color[2:4], 16)
    bb = int(hex_color[4:6], 16)
    return f"&H{alpha:02X}{bb:02X}{gg:02X}{rr:02X}"

def to_ass(segments, font_family, font_size, text_color, outline_color, outline_w, bg_color=None, karaoke=False) -> str:
    primary = ass_color_from_hex(text_color, alpha=0x00)
    outline = ass_color_from_hex(outline_color, alpha=0x00)
    backcol = ass_color_from_hex(bg_color or "#000000", alpha=0x40)  # semi-transparent

    border_style = 3 if bg_color else 1  # 3 draws an opaque box; we’ll use semi alpha
    shadow = 0

    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding",
        f"Style: Default,{font_family},{int(font_size)},{primary},&H000000FF,{outline},{backcol},"
        f"0,0,0,0,100,100,0,0,{border_style},{int(outline_w)},{shadow},2,30,30,30,1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]

    def fmt(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        cs = int((t - int(t)) * 100)  # centiseconds
        return f"{h:01d}:{m:02d}:{s:02d}.{cs:02d}"

    body = []
    for seg in segments:
        start = fmt(seg["start"])
        end = fmt(seg["end"])
        text = seg["text"].replace("\n", " ").strip()

        if karaoke and seg.get("words"):
            # Build {\k<centi>} karaoke timing tags
            ks = []
            for w in seg["words"]:
                dur_cs = max(1, int(round((w["end"] - w["start"]) * 100)))
                token = re.sub(r"\s+", " ", w["word"]).strip()
                if token:
                    ks.append(rf"{{\k{dur_cs}}}{token}")
            if ks:
                text = "".join(ks)

        body.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    return "\n".join(header + body) + "\n"


# ---------------------------
# Gradio UI
# ---------------------------
FONT_CHOICES = [
    "Arial", "Roboto", "Open Sans", "Lato", "Noto Sans", "Montserrat", "Inter"
]

custom_css = """
:root {
  --bg: #0e1320;
  --panel: #141a2a;
  --panel-2: #1b2336;
  --text: #e9eefc;
  --muted: #9fb0d1;
  --accent: #7aa2ff;
}
.gradio-container { max-width: 1200px !important; }
body { background: var(--bg); }
.dark .gr-block, .dark .gr-panel, .gr-group, .gr-form { background: var(--panel) !important; }
h1, h2, h3, .gr-markdown { color: var(--text) !important; }
label, .gr-input, .gr-button { color: var(--text) !important; }
span, p { color: var(--muted) !important; }
.settings-card { position: sticky; top: 12px; border-radius: 16px; background: var(--panel-2)!important; }
"""

def run_pipeline(audio_path, language_ui, lyrics_text,
                 font_family, font_size, text_color, outline_color, outline_w,
                 bg_color, use_karaoke, use_lyrics):
    if audio_path is None:
        raise gr.Error("Please upload an audio or video file.")

    lang_code = normalize_lang(language_ui)
    segs, full_text = transcribe_with_words(audio_path, lang_code)

    # If user provided lyrics, align those instead of raw ASR segments
    if use_lyrics and lyrics_text and lyrics_text.strip():
        flat_words = collect_recognized_words(segs)
        segs_aligned = align_lyrics_to_words(lyrics_text.strip(), flat_words)
        segs = segs_aligned

    # Build styled HTML preview (quick on-screen preview)
    preview_text = " ".join(s["text"] for s in segs).strip()
    styled_html = f"""
<div style="
  font-family:{font_family}, sans-serif;
  font-size:{int(font_size)}px;
  line-height:1.35;
  color:{text_color};
  background:{bg_color};
  padding:12px;
  border-radius:12px;
  text-shadow:
    -{outline_w}px 0 {outline_color},
     {outline_w}px 0 {outline_color},
     0 -{outline_w}px {outline_color},
     0  {outline_w}px {outline_color};
">{preview_text}</div>
"""

    # Create SRT and ASS files
    srt_text = to_srt(segs)
    ass_text = to_ass(
        segs,
        font_family=font_family,
        font_size=font_size,
        text_color=text_color,
        outline_color=outline_color,
        outline_w=outline_w,
        bg_color=bg_color,
        karaoke=bool(use_karaoke),
    )

    # Save temp files so Gradio can offer downloads
    srt_path = tempfile.NamedTemporaryFile(delete=False, suffix=".srt").name
    ass_path = tempfile.NamedTemporaryFile(delete=False, suffix=".ass").name
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_text)
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_text)

    return styled_html, segs, srt_path, ass_path


with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🎨 Colorvideo Subs — Transcribe, Style, and Export (SRT/ASS)")
    gr.Markdown(
        "Upload **MP3/WAV/MP4**, choose **Spanish/Hungarian/Auto**, optionally paste **lyrics** to force-align, "
        "then **download SRT/ASS**. ASS carries your styling & karaoke."
    )

    with gr.Row():
        with gr.Column(scale=3):
            audio = gr.Audio(label="Audio / Video", type="filepath")
            language = gr.Dropdown(choices=LANG_CHOICES, value="auto", label="Language")
            lyrics = gr.Textbox(lines=6, label="Optional lyrics (one or multiple lines)")
            use_lyrics = gr.Checkbox(value=False, label="Use pasted lyrics for timing (forced alignment)")
            run = gr.Button("Transcribe / Align", variant="primary")

            gr.Markdown("### Preview")
            transcript_html = gr.HTML()
            gr.Markdown("### Segments (debug/export)")
            segments_json = gr.JSON()

            gr.Markdown("### Downloads")
            srt_file = gr.File(label="SRT file", file_count="single")
            ass_file = gr.File(label="ASS file (styled/karaoke)", file_count="single")

        with gr.Column(scale=1):
            with gr.Group(elem_classes=["settings-card"]):
                gr.Markdown("### Subtitle Style")
                font_family = gr.Dropdown(FONT_CHOICES, value="Inter", label="Font")
                font_size   = gr.Slider(14, 72, value=36, step=1, label="Font size")
                text_color  = gr.ColorPicker(value="#FFFFFF", label="Text color")
                bg_color    = gr.ColorPicker(value="#000000", label="Background color (ASS box)")
                outline_color = gr.ColorPicker(value="#000000", label="Outline color")
                outline_w   = gr.Slider(0, 4, value=2, step=1, label="Outline width (px)")
                use_karaoke = gr.Checkbox(value=True, label="Karaoke (per-word) in ASS")

    run.click(
        run_pipeline,
        inputs=[audio, language, lyrics,
                font_family, font_size, text_color, outline_color, outline_w,
                bg_color, use_karaoke, use_lyrics],
        outputs=[transcript_html, segments_json, srt_file, ass_file]
    )

if __name__ == "__main__":
    demo.launch()
