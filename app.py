# Language Learning Subtitle Editor
# Version 2.1 — External Editor Workflow (Gradio 4.x safe)

import os
import re
import tempfile
from typing import List, Tuple, Optional

import gradio as gr
import torch
import whisperx

# -------------------------- Config --------------------------

VERSION = "2.1"
BANNER_COLOR = "#00BCD4"  # Cyan banner
DEFAULT_SAMPLE_TEXT_COLOR = "#1e88e5"  # Blue so it isn't white-on-white

LANG_MAP = {
    "auto": None, "auto-detect": None, "automatic": None,
    "hungarian": "hu", "magyar": "hu", "hun": "hu", "hu": "hu",
    "spanish": "es", "español": "es", "esp": "es", "es": "es",
    "english": "en", "eng": "en", "en": "en",
}

# -------------------------- Utilities --------------------------

def normalize_lang(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = s.strip().lower()
    if t in LANG_MAP:
        return LANG_MAP[t]
    m = re.search(r"\b([a-z]{2,3})\b", t)
    return m.group(1) if m else None

def seconds_to_timestamp_srt(t: float) -> str:
    """Convert seconds to SRT timestamp (HH:MM:SS,mmm)."""
    t = max(t, 0.0)
    h = int(t // 3600)
    t -= h * 3600
    m = int(t // 60)
    t -= m * 60
    s = int(t)
    ms = int(round((t - s) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def seconds_to_timestamp_ass(t: float) -> str:
    """Convert seconds to ASS timestamp (H:MM:SS.cc)."""
    t = max(t, 0.0)
    h = int(t // 3600)
    t -= h * 3600
    m = int(t // 60)
    t -= m * 60
    s = int(t)
    cs = int(round((t - s) * 100))  # centiseconds
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

# -------------------------- ASR Model --------------------------

_asr_model = None

def get_asr_model():
    """Lazy-load WhisperX with safe compute types for CPU/GPU."""
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"  # CPU int8; safe + fast

    print(f"[v{VERSION}] Loading WhisperX on {device} with compute_type={compute}")
    try:
        _asr_model = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        # Fallback if int8 not supported on this CPU
        if "compute type" in str(e).lower():
            fallback = "int16" if device == "cpu" else "float32"
            print(f"[v{VERSION}] Falling back to compute_type={fallback}")
            _asr_model = whisperx.load_model("small", device=device, compute_type=fallback)
        else:
            raise
    return _asr_model

# -------------------------- Transcription --------------------------

def transcribe_with_words(audio_path: str, language_code: Optional[str]) -> List[dict]:
    """
    Transcribe audio and return a list of word dicts:
    { start: float, end: float, text: str, color: str? }
    """
    model = get_asr_model()
    print("[Transcribe] Starting transcription...")
    # Do NOT pass unsupported args (e.g., word_timestamps). WhisperX will include words if available.
    result = model.transcribe(audio_path, language=language_code)
    segments = result.get("segments", [])

    words: List[dict] = []
    for seg in segments:
        s_start = float(seg.get("start", 0.0))
        s_end = float(seg.get("end", max(s_start + 0.2, s_start)))
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                # Some models store "word", others "text"
                w_text = (w.get("word") or w.get("text") or "").strip()
                if not w_text:
                    continue
                w_start = float(w.get("start", s_start))
                w_end = float(w.get("end", s_end))
                words.append({"start": w_start, "end": w_end, "text": w_text})
        else:
            # Fallback split by spaces into equal-duration chunks
            text = (seg.get("text") or "").strip()
            toks = [t for t in text.split() if t]
            if not toks:
                continue
            dur = max(s_end - s_start, 0.2)
            step = dur / len(toks)
            for i, tok in enumerate(toks):
                w_start = s_start + i * step
                w_end = min(s_start + (i + 1) * step, s_end)
                words.append({"start": round(w_start, 3), "end": round(w_end, 3), "text": tok})
    print(f"[Transcribe] Got {len(words)} words.")
    return words

# -------------------------- HTML Export / Import --------------------------

HTML_TEMPLATE_HEAD = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8" />
<title>Edit Your Subtitles</title>
<style>
  body {{ font-family: Arial, sans-serif; padding: 32px; max-width: 900px; margin: 0 auto; }}
  h1 {{ color: {BANNER_COLOR}; margin-bottom: 0.25rem; }}
  .note {{ color: #555; margin-bottom: 1rem; }}
  .panel {{ background: #f7f7f9; border: 1px solid #e2e2e6; border-radius: 10px; padding: 16px; margin: 16px 0; }}
  .text-area {{ border: 2px solid {BANNER_COLOR}; border-radius: 10px; padding: 16px; line-height: 2; font-size: 18px; }}
  .word {{ display: inline-block; padding: 2px 4px; margin: 2px 2px; cursor: text; color: {DEFAULT_SAMPLE_TEXT_COLOR}; }}
  .time {{ color: #999; font-size: 12px; margin-right: 6px; }}
</style>
</head>
<body>
<h1>Edit Your Subtitles</h1>
<p class="note">Tip: Edit text freely. Use your editor’s <b>Text Color</b> or <b>Highlighter</b> on any words. Save as <code>.html</code> and upload back.</p>
<div class="panel">
  <b>Recommended Color Guide (optional):</b>
  <ul>
    <li><span style="background:#FFFF00">Yellow</span> verbs</li>
    <li><span style="color:#FF0000">Red</span> important</li>
    <li><span style="color:#00FFFF">Cyan</span> nouns</li>
    <li><span style="color:#00FF00">Green</span> adjectives</li>
    <li><span style="color:#AA96DA">Purple</span> endings/conjugations</li>
  </ul>
</div>
<div class="text-area" contenteditable="true">
"""

HTML_TEMPLATE_TAIL = """
</div>
<p class="note">When done, save this page as <b>HTML</b> and upload it back to the app.</p>
</body>
</html>
"""

def export_to_html_for_editing(word_segments: List[dict]) -> str:
    """Create an editable HTML with words wrapped in spans."""
    html = [HTML_TEMPLATE_HEAD]
    for w in word_segments:
        # Timestamps included as data attributes (optional for future)
        html.append(
            f'<span class="word" data-start="{w["start"]:.3f}" data-end="{w["end"]:.3f}">{w["text"]}</span> '
        )
    html.append(HTML_TEMPLATE_TAIL)

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "subtitles_for_editing.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(html))
    return path

def _css_color_to_hex(style: str) -> Optional[str]:
    """
    Extract #RRGGBB or rgb(...) from inline style and normalize to #RRGGBB.
    Returns None if no color found.
    """
    if not style:
        return None
    # First prefer explicit color property if present
    # Try hex
    m = re.search(r"#([0-9A-Fa-f]{6})", style)
    if m:
        return "#" + m.group(1).upper()
    # Try rgb(...)
    m = re.search(r"rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", style)
    if m:
        r, g, b = map(int, m.groups())
        return f"#{r:02X}{g:02X}{b:02X}"
    # Try named colors minimally (common ones)
    named = {
        "red": "#FF0000", "yellow": "#FFFF00", "cyan": "#00FFFF", "aqua": "#00FFFF",
        "green": "#00FF00", "blue": "#0000FF", "magenta": "#FF00FF", "fuchsia": "#FF00FF",
        "black": "#000000", "white": "#FFFFFF", "purple": "#800080",
    }
    m = re.search(r"color\s*:\s*([a-zA-Z]+)", style)
    if m and m.group(1).lower() in named:
        return named[m.group(1).lower()]
    # Also check background
    m = re.search(r"background(?:-color)?\s*:\s*#([0-9A-Fa-f]{6})", style)
    if m:
        return "#" + m.group(1).upper()
    m = re.search(r"background(?:-color)?\s*:\s*rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", style)
    if m:
        r, g, b = map(int, m.groups())
        return f"#{r:02X}{g:02X}{b:02X}"
    return None

def import_from_html(html_path: str, original_words: List[dict]) -> List[dict]:
    """
    Parse edited HTML, extract words + colors, align back onto original timestamps order.
    We read visible text in the editable area left-to-right; colors from inline styles if present.
    """
    from html.parser import HTMLParser

    class Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_text_area = False
            self.stack_styles: List[str] = []
            self.words: List[Tuple[str, Optional[str]]] = []  # (text, hex_color)

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "div" and attrs.get("class", "") == "text-area":
                self.in_text_area = True
            if self.in_text_area and tag in ("span", "font", "mark"):
                style = attrs.get("style", "")
                color = _css_color_to_hex(style)
                # Some editors put color in <font color="">
                if not color and "color" in attrs:
                    color = _css_color_to_hex(f"color:{attrs['color']}")
                # Some highlights are "background: ..."
                if not color and "background" in attrs:
                    color = _css_color_to_hex(f"background:{attrs['background']}")
                self.stack_styles.append(color or "")

        def handle_endtag(self, tag):
            if tag == "div" and self.in_text_area:
                self.in_text_area = False
            if self.in_text_area and tag in ("span", "font", "mark"):
                if self.stack_styles:
                    self.stack_styles.pop()

        def handle_data(self, data):
            if not self.in_text_area:
                return
            # split on whitespace, keep only non-empty tokens
            for tok in re.split(r"\s+", data):
                tok = tok.strip()
                if not tok:
                    continue
                # pick last non-empty color in the stack
                color = None
                for c in reversed(self.stack_styles):
                    if c:
                        color = c
                        break
                if not color:
                    color = DEFAULT_SAMPLE_TEXT_COLOR  # default blue
                self.words.append((tok, color))

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    p = Parser()
    p.feed(html)

    edited = []
    for i, ow in enumerate(original_words):
        if i < len(p.words):
            txt, col = p.words[i]
            edited.append({"start": ow["start"], "end": ow["end"], "text": txt, "color": col})
        else:
            edited.append({"start": ow["start"], "end": ow["end"], "text": ow["text"], "color": DEFAULT_SAMPLE_TEXT_COLOR})
    return edited

# -------------------------- SubRip (SRT) / ASS --------------------------

def export_to_srt(words: List[dict], words_per_line: int = 5) -> str:
    """Plain SRT (no color)."""
    i = 0
    n = 1
    out = []
    while i < len(words):
        chunk = words[i:i + words_per_line]
        start = seconds_to_timestamp_srt(chunk[0]["start"])
        end = seconds_to_timestamp_srt(chunk[-1]["end"])
        text = " ".join(w["text"] for w in chunk)
        out.append(f"{n}\n{start} --> {end}\n{text}\n")
        i += words_per_line
        n += 1
    return "\n".join(out)

def _hex_to_ass_bgr(hex_color: str) -> str:
    """#RRGGBB -> &H00BBGGRR for ASS."""
    if not hex_color or not hex_color.startswith("#") or len(hex_color) != 7:
        hex_color = "#FFFFFF"
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"&H00{b:02X}{g:02X}{r:02X}"

def export_to_ass(words: List[dict], words_per_line: int = 5, font="Arial", size=36) -> str:
    """ASS with per-word colors."""
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,30,30,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    i = 0
    lines = [header]
    while i < len(words):
        chunk = words[i:i + words_per_line]
        start = seconds_to_timestamp_ass(chunk[0]["start"])
        end = seconds_to_timestamp_ass(chunk[-1]["end"])
        parts = []
        for w in chunk:
            col = _hex_to_ass_bgr(w.get("color", DEFAULT_SAMPLE_TEXT_COLOR))
            parts.append(f"{{\\c{col}}}{w['text']}")
        text = " ".join(parts)
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
        i += words_per_line
    return "\n".join(lines) + "\n"

def _save_temp(content: str, ext: str) -> str:
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, f"subtitles{ext}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

# -------------------------- Gradio App --------------------------

def create_app():
    with gr.Blocks(theme=gr.themes.Soft(), title=f"Language Subtitle Editor v{VERSION}") as demo:

        gr.HTML(
            f"""
            <div style="background:{BANNER_COLOR};color:white;padding:18px;border-radius:12px;margin-bottom:16px;text-align:center">
              <div style="font-size:22px;font-weight:700;">Language Learning Subtitle Editor</div>
              <div style="opacity:0.9;">Version {VERSION} — Edit in Word/Google Docs/Your Browser</div>
            </div>
            """
        )

        # States
        word_segments_state = gr.State([])     # original words (timestamps)
        edited_words_state = gr.State([])      # edited (with colors)
        status_box = gr.Textbox(label="Status", value="Ready.", interactive=False, lines=3)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1) Transcribe Audio")
                audio_input = gr.Audio(label="Upload Audio/Video", type="filepath")
                language_dropdown = gr.Dropdown(
                    choices=[("Auto-detect", "auto"), ("Spanish", "es"), ("Hungarian", "hu"), ("English", "en")],
                    value="auto",
                    label="Language"
                )
                transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")
                transcript_preview = gr.Textbox(label="Transcript Preview", lines=8, interactive=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 2) Download Editable HTML")
                download_html_btn = gr.Button("Create & Download HTML", size="lg")
                html_file = gr.File(label="Download the generated HTML, edit it, then upload it below")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 3) Upload Edited HTML")
                upload_html = gr.File(label="Upload your edited HTML", file_types=[".html", ".htm"])
                import_btn = gr.Button("Import Edited HTML", variant="primary", size="lg")
                import_status = gr.Textbox(label="Import Status", interactive=False, lines=3)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 4) Export Subtitles")
                with gr.Row():
                    words_per = gr.Slider(minimum=2, maximum=10, value=5, step=1, label="Words per subtitle line")
                    font_family = gr.Dropdown(
                        choices=["Arial", "Times New Roman", "Courier New", "Georgia", "Verdana"],
                        value="Arial", label="Font"
                    )
                    font_size = gr.Slider(minimum=20, maximum=72, value=36, step=2, label="Size")
                with gr.Row():
                    export_srt_btn = gr.Button("Export SRT (no colors)")
                    export_ass_btn = gr.Button("Export ASS (with colors)", variant="primary")
                srt_file = gr.File(label="SRT File")
                ass_file = gr.File(label="ASS File")

        # ---------- Handlers ----------

        def do_transcribe(audio_path, lang_sel):
            if not audio_path:
                return "❌ Error: no audio file provided.", [], ""
            try:
                lang_code = normalize_lang(lang_sel)
                msg = f"Loading model…\nLanguage: {lang_sel}"
                # step 1 message
                yield msg, [], ""
                words = transcribe_with_words(audio_path, lang_code)
                preview = " ".join(w["text"] for w in words[:120])
                if len(words) > 120:
                    preview += " …"
                yield f"✅ Transcribed {len(words)} words.", words, preview
            except Exception as e:
                yield f"❌ Error during transcription: {e}", [], ""

        transcribe_btn.click(
            fn=do_transcribe,
            inputs=[audio_input, language_dropdown],
            outputs=[status_box, word_segments_state, transcript_preview]
        )

        def handle_download_html(words):
            if not words:
                gr.Warning("Transcribe first.")
                return None
            try:
                path = export_to_html_for_editing(words)
                return path
            except Exception as e:
                gr.Warning(f"Error creating HTML: {e}")
                return None

        download_html_btn.click(
            fn=handle_download_html,
            inputs=[word_segments_state],
            outputs=[html_file]
        )

        def handle_import_html(file, original_words):
            if not file:
                return "❌ No file uploaded.", []
            if not original_words:
                return "❌ Transcribe first.", []
            try:
                edited = import_from_html(file.name, original_words)
                return f"✅ Imported {len(edited)} words with colors.", edited
            except Exception as e:
                return f"❌ Error importing HTML: {e}", []

        import_btn.click(
            fn=handle_import_html,
            inputs=[upload_html, word_segments_state],
            outputs=[import_status, edited_words_state]
        )

        def handle_export_srt(edited_words, n_words):
            if not edited_words:
                gr.Warning("Import your edited HTML first.")
                return None
            srt = export_to_srt(edited_words, int(n_words))
            return _save_temp(srt, ".srt")

        export_srt_btn.click(
            fn=handle_export_srt,
            inputs=[edited_words_state, words_per],
            outputs=[srt_file]
        )

        def handle_export_ass(edited_words, n_words, font, size):
            if not edited_words:
                gr.Warning("Import your edited HTML first.")
                return None
            ass = export_to_ass(edited_words, int(n_words), font, int(size))
            return _save_temp(ass, ".ass")

        export_ass_btn.click(
            fn=handle_export_ass,
            inputs=[edited_words_state, words_per, font_family, font_size],
            outputs=[ass_file]
        )

        return demo

# -------------------------- Main --------------------------

if __name__ == "__main__":
    demo = create_app()
    demo.launch()
