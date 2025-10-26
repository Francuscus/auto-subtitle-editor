# Language Learning Subtitle Editor
# Version 2.1  (same as previous workflow; adds ZIP import support only)

import os
import re
import tempfile
from typing import List
from html import unescape
import zipfile

import gradio as gr
import torch
import whisperx


# -------------------------- Config --------------------------

VERSION = "2.1"

LANG_MAP = {
    "auto": None, "auto-detect": None, "automatic": None,
    "hungarian": "hu", "magyar": "hu", "hun": "hu",
    "spanish": "es", "español": "es", "esp": "es",
    "english": "en", "eng": "en",
}


# -------------------------- Utilities --------------------------

def normalize_lang(s: str | None):
    if not s:
        return None
    t = s.strip().lower()
    if t in LANG_MAP:
        return LANG_MAP[t]
    m = re.search(r"\b([a-z]{2,3})\b", t)
    return m.group(1) if m else None


def seconds_to_timestamp(t: float) -> str:
    """Convert seconds to SRT/ASS timestamp format"""
    t = max(t, 0.0)
    h = int(t // 3600)
    t -= h * 3600
    m = int(t // 60)
    t -= m * 60
    s = int(t)
    ms = int(round((t - s) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# -------------------------- ASR Model --------------------------

_asr_model = None

def get_asr_model():
    """
    Same behavior as before:
      - WhisperX "small"
      - CUDA if available (float16), else CPU (int8)
      - Falls back to int16/float32 if needed
    """
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"

    try:
        _asr_model = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        if "compute type" in str(e).lower():
            fallback = "int16" if device == "cpu" else "float32"
            _asr_model = whisperx.load_model("small", device=device, compute_type=fallback)
        else:
            raise

    return _asr_model


# -------------------------- Transcription --------------------------

def transcribe_with_words(audio_path: str, language_code: str | None) -> List[dict]:
    """
    Transcribe and return *word-level* timestamps if present.
    (No API signature changes; no word_timestamps arg.)
    """
    model = get_asr_model()
    result = model.transcribe(audio_path, language=language_code)
    segments = result["segments"]

    word_segments = []
    for seg in segments:
        # Prefer native word timing if provided by backend
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                word_segments.append({
                    "start": float(w.get("start", seg["start"])),
                    "end": float(w.get("end", seg["end"])),
                    "text": (w.get("word") or w.get("text") or "").strip(),
                })
        else:
            # Fallback: uniform split of the segment text
            text = seg.get("text", "").strip()
            words = text.split()
            if not words:
                continue
            seg_start = float(seg["start"])
            seg_end = float(seg["end"])
            seg_dur = max(seg_end - seg_start, 0.001)
            word_dur = seg_dur / len(words)
            for i, word in enumerate(words):
                w_start = seg_start + i * word_dur
                w_end = min(seg_end, w_start + word_dur)
                word_segments.append({
                    "start": round(w_start, 3),
                    "end": round(w_end, 3),
                    "text": word
                })
    return word_segments


# -------------------------- HTML Export for External Editing --------------------------

def export_to_html_for_editing(word_segments: List[dict]) -> str:
    """
    Creates a simple, editable HTML file (contenteditable) with words laid out.
    Users can fix text and apply colors (font color or highlight) in Word/Docs/Browser.
    """
    html_content = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Edit Your Subtitles</title>
<style>
  body { font-family: Arial, sans-serif; padding: 40px; max-width: 900px; margin: 0 auto; }
  h1 { color: #00BCD4; }
  .instructions { background: #f7f7f7; padding: 16px 20px; border-radius: 8px; margin-bottom: 18px; }
  .text-area { font-size: 18px; line-height: 2; padding: 20px; border: 2px solid #00BCD4; border-radius: 8px; }
  .word { display: inline-block; padding: 2px 4px; margin: 2px 1px; cursor: text; }
</style>
</head>
<body>
<h1>Edit Your Subtitles</h1>
<div class="instructions">
  <ol>
    <li>Edit the text below (fix typos, change words)</li>
    <li>Use your editor’s <b>Text Color</b> or <b>Highlight</b> tools on any words</li>
    <li>Save as <b>HTML</b> (or in Google Docs: <b>File → Download → Web page (.zip)</b>)</li>
    <li>Upload the HTML or ZIP back into the app</li>
  </ol>
</div>
<div class="text-area" contenteditable="true">
"""

    for w in word_segments:
        txt = (w["text"] or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html_content += f'<span class="word">{txt}</span> '

    html_content += """
</div>
</body>
</html>
"""

    temp_dir = tempfile.mkdtemp()
    html_path = os.path.join(temp_dir, "subtitles_for_editing.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return html_path


# -------------------------- Import HTML (now supports .zip from Google Docs) --------------------------

def _extract_html_from_zip(zip_path: str) -> str | None:
    """
    Extracts the first .html/.htm file from a Google Docs "Web page (.zip)" export.
    Returns the path to the found HTML, or None if none present.
    """
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, "r") as zf:
        # extract all
        zf.extractall(temp_dir)
    # walk to find first .html/.htm
    for root, _, files in os.walk(temp_dir):
        for name in files:
            lower = name.lower()
            if lower.endswith(".html") or lower.endswith(".htm"):
                return os.path.join(root, name)
    return None


def import_from_html_or_zip(upload_path: str, original_words: List[dict]) -> List[dict]:
    """
    Accepts either a .html/.htm file OR a .zip (Google Docs "Web page").
    Extracts words and any inline colors (font color or background).
    """
    # 1) Resolve to an HTML file
    path_lower = upload_path.lower()
    if path_lower.endswith(".zip"):
        html_path = _extract_html_from_zip(upload_path)
        if not html_path:
            raise ValueError("Could not find an HTML file inside the ZIP.")
    elif path_lower.endswith(".html") or path_lower.endswith(".htm"):
        html_path = upload_path
    else:
        raise ValueError("Please upload a .html/.htm file or a Google Docs .zip export.")

    # 2) Parse HTML
    from html.parser import HTMLParser

    class ColorHTMLParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.words = []
            self.current_color = "#FFFFFF"
            self.in_text_area = False

        def handle_starttag(self, tag, attrs):
            attrs_dict = dict(attrs)
            # detect the editable region (by class)
            if tag == "div" and "class" in attrs_dict and "text-area" in attrs_dict["class"]:
                self.in_text_area = True
            if self.in_text_area and tag in ("span", "font"):
                # gather color from style, or <font color="">
                style = attrs_dict.get("style", "")
                font_color = attrs_dict.get("color", "")
                found = None

                # Try hex in style/background-color/color
                m = re.search(r"#([0-9A-Fa-f]{6})", style)
                if m:
                    found = "#" + m.group(1)
                # Try rgb()
                if not found:
                    m = re.search(r"rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", style)
                    if m:
                        r, g, b = map(int, m.groups())
                        found = f"#{r:02X}{g:02X}{b:02X}"
                # Try <font color="">
                if not found and font_color:
                    # #RRGGBB or named colors aren't common here; handle hex first
                    m = re.search(r"#([0-9A-Fa-f]{6})", font_color)
                    if m:
                        found = "#" + m.group(1)

                if found:
                    self.current_color = found

        def handle_endtag(self, tag):
            if tag == "div" and self.in_text_area:
                self.in_text_area = False
            # reset color when leaving a colored span/font
            if self.in_text_area and tag in ("span", "font"):
                self.current_color = "#FFFFFF"

        def handle_data(self, data):
            if not self.in_text_area:
                return
            for raw in data.split():
                word = raw.strip()
                if not word:
                    continue
                self.words.append({"text": unescape(word), "color": self.current_color})

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    parser = ColorHTMLParser()
    parser.feed(html)
    edited = parser.words

    # 3) Map edited word texts/colors back onto original timings (by index)
    out = []
    for i, orig in enumerate(original_words):
        if i < len(edited):
            out.append({
                "start": orig["start"],
                "end": orig["end"],
                "text": edited[i]["text"],
                "color": edited[i]["color"] or "#FFFFFF"
            })
        else:
            out.append({
                "start": orig["start"],
                "end": orig["end"],
                "text": orig["text"],
                "color": "#FFFFFF"
            })
    return out


# -------------------------- Subtitle Exports --------------------------

def export_to_ass(words: List[dict], words_per_line: int = 5,
                  font: str = "Arial", size: int = 36) -> str:
    """Export ASS with per-word color (like before)."""
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

    events = []
    i = 0
    while i < len(words):
        chunk = words[i:i + words_per_line]
        if not chunk:
            break

        start = seconds_to_timestamp(chunk[0]["start"]).replace(",", ".")
        end = seconds_to_timestamp(chunk[-1]["end"]).replace(",", ".")

        parts = []
        for w in chunk:
            color = (w.get("color") or "#FFFFFF").lstrip("#")
            # convert to BGR for ASS
            try:
                r = int(color[0:2], 16)
                g = int(color[2:4], 16)
                b = int(color[4:6], 16)
            except Exception:
                r, g, b = (255, 255, 255)
            ass_color = f"&H00{b:02X}{g:02X}{r:02X}"
            safe_text = (w["text"] or "").replace("{", "(").replace("}", ")")
            parts.append(f"{{\\c{ass_color}}}{safe_text}")

        line = " ".join(parts)
        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{line}")
        i += words_per_line

    return header + "\n".join(events) + "\n"


def export_to_srt(words: List[dict], words_per_line: int = 5) -> str:
    """Plain SRT (no colors)."""
    lines = []
    i = 0
    n = 1
    while i < len(words):
        chunk = words[i:i + words_per_line]
        if not chunk:
            break
        start = seconds_to_timestamp(chunk[0]["start"])
        end = seconds_to_timestamp(chunk[-1]["end"])
        text = " ".join((w["text"] or "") for w in chunk)
        lines.append(f"{n}\n{start} --> {end}\n{text}\n")
        n += 1
        i += words_per_line
    return "\n".join(lines)


def save_file(content: str, extension: str) -> str:
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, f"subtitles{extension}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# -------------------------- Gradio App (same layout/flow) --------------------------

def create_app():
    with gr.Blocks(theme=gr.themes.Soft(), title=f"LL Subtitle Editor v{VERSION}") as demo:
        gr.HTML(f"""
        <div style="background:#00BCD4;color:white;padding:16px;text-align:center;border-radius:8px;margin-bottom:12px;">
          <h2 style="margin:0;">Language Learning Subtitle Editor</h2>
          <div>Version {VERSION} — Edit in Word/Docs and upload HTML or Google Docs ZIP</div>
        </div>
        """)

        word_segments_state = gr.State([])
        edited_words_state = gr.State([])

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 1 — Transcribe")
                audio_input = gr.Audio(label="Upload Audio/Video", type="filepath")
                language_dropdown = gr.Dropdown(
                    choices=[("Auto-detect", "auto"), ("Spanish", "es"), ("Hungarian", "hu"), ("English", "en")],
                    value="auto",
                    label="Language"
                )
                transcribe_btn = gr.Button("Transcribe", variant="primary")
                status_text = gr.Textbox(label="Status", value="Ready…", interactive=False, lines=3)
                transcript_preview = gr.Textbox(label="Transcript Preview", lines=8, interactive=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 2 — Download HTML for Editing")
                download_html_btn = gr.Button("Download HTML")
                html_file = gr.File(label="Click to download, then edit in Word/Docs/Browser")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 3 — Upload Edited File (HTML or Google Docs ZIP)")
                upload_html = gr.File(label="Upload .html/.htm or .zip", file_types=[".html", ".htm", ".zip"])
                import_btn = gr.Button("Import Edited File", variant="primary")
                import_status = gr.Textbox(label="Import Status", value="", interactive=False, lines=3)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 4 — Export Subtitles")
                with gr.Row():
                    words_per = gr.Slider(minimum=2, maximum=10, value=5, step=1, label="Words per subtitle line")
                    font_family = gr.Dropdown(choices=["Arial", "Times New Roman", "Courier New", "Georgia", "Verdana"],
                                              value="Arial", label="ASS Font")
                    font_size = gr.Slider(minimum=20, maximum=72, value=36, step=2, label="ASS Size")
                with gr.Row():
                    export_srt_btn = gr.Button("Export SRT")
                    export_ass_btn = gr.Button("Export ASS (with colors)", variant="primary")
                srt_file = gr.File(label="SRT File")
                ass_file = gr.File(label="ASS File")

        # ---- Actions ----

        def do_transcribe(audio_path, language):
            if not audio_path:
                return "❌ No audio file.", [], ""
            try:
                yield "Loading model…", [], ""
                lang_code = normalize_lang(language)
                yield "Transcribing…", [], ""
                words = transcribe_with_words(audio_path, lang_code)
                preview = " ".join(w["text"] for w in words[:120])
                if len(words) > 120:
                    preview += " …"
                return f"✅ Done. {len(words)} words.", words, preview
            except Exception as e:
                return f"❌ Error: {e}", [], ""

        def do_download_html(word_segments):
            if not word_segments:
                gr.Warning("Transcribe first.")
                return None
            try:
                return export_to_html_for_editing(word_segments)
            except Exception as e:
                gr.Warning(f"Error creating HTML: {e}")
                return None

        def do_import_html(uploaded, original_words):
            if not uploaded:
                return [], "❌ No file uploaded."
            if not original_words:
                return [], "❌ Transcribe first."

            try:
                path = uploaded.name
                edited = import_from_html_or_zip(path, original_words)
                return edited, f"✅ Imported {len(edited)} words with colors (if any)."
            except Exception as e:
                return [], f"❌ Error importing: {e}"

        def do_export_srt(words, words_per_line):
            if not words:
                gr.Warning("Import edited file first.")
                return None
            return save_file(export_to_srt(words, int(words_per_line)), ".srt")

        def do_export_ass(words, words_per_line, font, size):
            if not words:
                gr.Warning("Import edited file first.")
                return None
            return save_file(export_to_ass(words, int(words_per_line), font, int(size)), ".ass")

        transcribe_btn.click(
            fn=do_transcribe,
            inputs=[audio_input, language_dropdown],
            outputs=[status_text, word_segments_state, transcript_preview]
        )

        download_html_btn.click(
            fn=do_download_html,
            inputs=[word_segments_state],
            outputs=[html_file]
        )

        import_btn.click(
            fn=do_import_html,
            inputs=[upload_html, word_segments_state],
            outputs=[edited_words_state, import_status]
        )

        export_srt_btn.click(
            fn=do_export_srt,
            inputs=[edited_words_state, words_per],
            outputs=[srt_file]
        )

        export_ass_btn.click(
            fn=do_export_ass,
            inputs=[edited_words_state, words_per, font_family, font_size],
            outputs=[ass_file]
        )

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch()
