# Language Learning Subtitle Editor
# Version 2.1 - External Editor Workflow (HTML / Google Docs ZIP supported)
# Banner Color: #00BCD4 (Cyan)

import os
import re
import tempfile
from typing import List, Tuple
import zipfile

import gradio as gr
import torch
import whisperx


# -------------------------- Config --------------------------

VERSION = "2.1"
BANNER_COLOR = "#00BCD4"  # Cyan banner

LANG_MAP = {
    "auto": None, "auto-detect": None, "automatic": None,
    "hungarian": "hu", "magyar": "hu", "hun": "hu", "hu": "hu",
    "spanish": "es", "español": "es", "esp": "es", "es": "es",
    "english": "en", "eng": "en", "en": "en",
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


def seconds_to_srt_ts(t: float) -> str:
    """Seconds -> SRT timestamp 00:00:00,000"""
    t = max(t, 0.0)
    h = int(t // 3600)
    t -= h * 3600
    m = int(t // 60)
    t -= m * 60
    s = int(t)
    ms = int(round((t - s) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def seconds_to_ass_ts(t: float) -> str:
    """Seconds -> ASS timestamp h:MM:SS.cc (centiseconds)"""
    t = max(t, 0.0)
    h = int(t // 3600)
    t -= h * 3600
    m = int(t // 60)
    t -= m * 60
    s = int(t)
    cs = int(round((t - s) * 100))  # centiseconds
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def hex_to_ass_bgr(hex_color: str) -> str:
    """
    #RRGGBB -> &HBBGGRR (ASS BGR with leading &H00)
    returns like &H00BBGGRR
    """
    if not hex_color:
        return "&H00FFFFFF"
    hx = hex_color.strip()
    if hx.startswith("#"):
        hx = hx[1:]
    # pad if short
    if len(hx) < 6:
        hx = (hx + "FFFFFF")[:6]
    r = int(hx[0:2], 16)
    g = int(hx[2:4], 16)
    b = int(hx[4:6], 16)
    return f"&H00{b:02X}{g:02X}{r:02X}"


# -------------------------- ASR Model --------------------------

_asr_model = None


def get_asr_model():
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"  # CPU -> int8 for Spaces

    print(f"[v{VERSION}] Loading WhisperX on {device} with {compute}")
    try:
        _asr_model = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        # Fallback if int8 not supported in this CPU
        if "compute type" in str(e).lower():
            fallback = "int16" if device == "cpu" else "float32"
            print(f"[v{VERSION}] Falling back to compute_type={fallback}")
            _asr_model = whisperx.load_model("small", device=device, compute_type=fallback)
        else:
            raise
    return _asr_model


# -------------------------- Transcription --------------------------

def transcribe_with_words(audio_path: str, language_code: str | None) -> List[dict]:
    """
    Returns a list of words:
    [{ "start": float, "end": float, "text": str, "color": "#FFFFFF" (default) }, ...]
    """
    model = get_asr_model()
    print("[ASR] Transcribing...")
    result = model.transcribe(audio_path, language=language_code)  # do NOT pass word_timestamps (whisperx sets per model)

    words: List[dict] = []
    segments = result.get("segments", [])
    for seg in segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        seg_text = seg.get("text", "").strip()

        # Prefer word-level if present
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                w_text = (w.get("word") or w.get("text") or "").strip()
                if not w_text:
                    continue
                w_start = float(w.get("start", seg_start))
                w_end = float(w.get("end", seg_end))
                words.append({"start": w_start, "end": w_end, "text": w_text, "color": "#FFFFFF"})
        else:
            # Fallback: split evenly by word count
            tokens = seg_text.split()
            if not tokens:
                continue
            dur = max(seg_end - seg_start, 0.01)
            step = dur / len(tokens)
            for i, tok in enumerate(tokens):
                w_start = seg_start + i * step
                w_end = min(seg_start + (i + 1) * step, seg_end)
                words.append({"start": round(w_start, 3), "end": round(w_end, 3), "text": tok, "color": "#FFFFFF"})

    print(f"[ASR] {len(words)} words.")
    return words


# -------------------------- HTML Export / Import --------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Edit Your Subtitles</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 40px; max-width: 900px; margin: 0 auto; }}
    h1 {{ color: {banner}; }}
    .instructions {{ background: #f7f7f7; padding: 16px 20px; border-radius: 10px; margin-bottom: 22px; }}
    .text-area {{ font-size: 20px; line-height: 2; padding: 16px; border: 2px solid {banner}; border-radius: 10px; }}
    .word {{ display: inline-block; padding: 2px 4px; margin: 2px 2px; }}
  </style>
</head>
<body>
  <h1>Edit Your Subtitles</h1>
  <div class="instructions">
    <ol>
      <li>Edit text freely (fix typos, reorder, etc.).</li>
      <li>Use your editor's <b>Highlight</b> or <b>Text Color</b> tools to color words.</li>
      <li>Save as <b>web page (.html)</b>. In Google Docs, it downloads a <b>.zip</b>—upload that zip back.</li>
    </ol>
  </div>

  <div class="text-area" contenteditable="true">
    {words_html}
  </div>

  <p style="color:#666;margin-top:18px;">Tip: Colors you apply will be preserved into the .ASS subtitle export.</p>
</body>
</html>
"""


def export_to_html_for_editing(word_segments: List[dict]) -> str:
    # Build words
    parts = []
    for w in word_segments:
        safe = (w["text"] or "").replace("<", "&lt;").replace(">", "&gt;")
        parts.append(f'<span class="word">{safe}</span>')
    html = HTML_TEMPLATE.format(banner=BANNER_COLOR, words_html=" ".join(parts))

    temp_dir = tempfile.mkdtemp()
    html_path = os.path.join(temp_dir, "subtitles_for_editing.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def _extract_color_from_style(style: str) -> str | None:
    if not style:
        return None
    # #RRGGBB
    m = re.search(r"#([0-9a-fA-F]{6})", style)
    if m:
        return f"#{m.group(1).upper()}"
    # rgb(r,g,b)
    m = re.search(r"rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", style)
    if m:
        r, g, b = map(int, m.groups())
        return f"#{r:02X}{g:02X}{b:02X}"
    return None


def import_from_html(html_path: str, original_words: List[dict]) -> List[dict]:
    """
    Parse edited HTML:
    - Extract words in order
    - Pick inline color (background-color or color) if present
    - Map back onto original timestamps (1:1 by index)
    """
    from bs4 import BeautifulSoup  # lightweight parser; included in requirements via gradio deps

    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    text_area = soup.select_one(".text-area") or soup.body
    collected: List[Tuple[str, str | None]] = []

    if text_area:
        # prefer span.word, but also capture raw text nodes split by spaces
        spans = text_area.find_all("span")
        if spans:
            for sp in spans:
                txt = (sp.get_text() or "").strip()
                if not txt:
                    continue
                color = None
                style = sp.get("style", "")
                color = _extract_color_from_style(style) or color
                # Also check inline 'bgcolor' or highlight spans (Google Docs may wrap with <span style="background-color: ...">)
                color = color or _extract_color_from_style(sp.get("style", ""))
                collected.append((txt, color))
        else:
            # fall back: split plain text
            tokens = (text_area.get_text() or "").split()
            for tok in tokens:
                collected.append((tok.strip(), None))
    else:
        # whole doc fallback
        tokens = (soup.get_text() or "").split()
        for tok in tokens:
            collected.append((tok.strip(), None))

    # Map words back to original timing (1-to-1 by index)
    edited: List[dict] = []
    for i, orig in enumerate(original_words):
        if i < len(collected):
            text, color = collected[i]
            edited.append({
                "start": orig["start"],
                "end": orig["end"],
                "text": text,
                "color": color or "#FFFFFF",
            })
        else:
            edited.append({
                "start": orig["start"],
                "end": orig["end"],
                "text": orig["text"],
                "color": orig.get("color", "#FFFFFF"),
            })
    return edited


# -------------------------- Subtitle Exporters --------------------------

def export_to_ass(words: List[dict], words_per_line: int = 5, font: str = "Arial", size: int = 36) -> str:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,30,30,48,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    i = 0
    n = len(words)
    while i < n:
        chunk = words[i:i + words_per_line]
        if not chunk:
            break
        start = seconds_to_ass_ts(chunk[0]["start"])
        end = seconds_to_ass_ts(chunk[-1]["end"])

        text_parts = []
        for w in chunk:
            c = hex_to_ass_bgr(w.get("color", "#FFFFFF"))
            # \c&HBBGGRR sets primary color; use per word
            safe = (w["text"] or "").replace("{", "").replace("}", "")
            text_parts.append(f"{{\\c{c}}}{safe}")
        line_text = " ".join(text_parts)
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{line_text}")
        i += words_per_line

    return header + "\n".join(lines) + "\n"


def export_to_srt(words: List[dict], words_per_line: int = 5) -> str:
    out = []
    i = 0
    idx = 1
    n = len(words)
    while i < n:
        chunk = words[i:i + words_per_line]
        if not chunk:
            break
        start = seconds_to_srt_ts(chunk[0]["start"])
        end = seconds_to_srt_ts(chunk[-1]["end"])
        txt = " ".join(w["text"] for w in chunk)
        out.append(f"{idx}\n{start} --> {end}\n{txt}\n")
        idx += 1
        i += words_per_line
    return "\n".join(out)


def save_text_file(content: str, ext: str) -> str:
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, f"subtitles{ext}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# -------------------------- Gradio App --------------------------

def create_app():
    with gr.Blocks(theme=gr.themes.Soft(), title="Language Learning Subtitle Editor") as demo:
        gr.HTML(f"""
        <div style="background:{BANNER_COLOR};color:white;padding:20px;text-align:center;border-radius:8px;margin-bottom:16px;">
          <h1 style="margin:0;">Language Learning Subtitle Editor</h1>
          <p style="margin:6px 0 0 0;">Version {VERSION} &middot; Edit in Word / Google Docs, then re-import</p>
        </div>
        """)

        gr.Markdown("""
**Workflow**
1) Upload audio/video and Transcribe  
2) Download **HTML** (edit anywhere, add colors)  
3) Re-upload edited **.html** or Google Docs **.zip**  
4) Export **SRT** (no colors) or **ASS** (with colors)
""")

        # States
        word_segments_state = gr.State([])
        edited_words_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Step 1 — Transcribe")
                audio_in = gr.Audio(label="Audio/Video", type="filepath")
                lang = gr.Dropdown(
                    choices=[("Auto-detect", "auto"), ("Spanish", "es"), ("Hungarian", "hu"), ("English", "en")],
                    value="auto", label="Language"
                )
                btn_transcribe = gr.Button("Transcribe", variant="primary")
                status = gr.Textbox(label="Status", value="Ready.", interactive=False, lines=3)
                preview = gr.Textbox(label="Transcript Preview", lines=8, interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("### Step 2 — Edit Externally")
                btn_dl_html = gr.Button("Download HTML for Editing")
                html_file = gr.File(label="Download this file, edit, then upload back", interactive=False)
                gr.Markdown("""
**Google Docs tip:** Use **File → Download → Web page (.html, zipped)**.  
Then upload the **.zip** directly in Step 3.
""")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Step 3 — Import Edited HTML / ZIP")
                upload_html = gr.File(
                    label="Upload edited HTML or Google Docs ZIP",
                    file_types=[".html", ".htm", ".zip"]
                )
                btn_import = gr.Button("Import Edited File", variant="primary")
                import_status = gr.Textbox(label="Import Status", value="", interactive=False, lines=2)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Step 4 — Export Subtitles")
                with gr.Row():
                    words_per = gr.Slider(minimum=2, maximum=10, value=5, step=1, label="Words per subtitle line")
                    font = gr.Dropdown(choices=["Arial", "Times New Roman", "Verdana", "Georgia", "Courier New"],
                                       value="Arial", label="ASS Font")
                    size = gr.Slider(minimum=20, maximum=72, value=36, step=2, label="ASS Size")

                with gr.Row():
                    btn_srt = gr.Button("Export SRT (no colors)")
                    btn_ass = gr.Button("Export ASS (keeps colors)", variant="primary")

                srt_out = gr.File(label="SRT File")
                ass_out = gr.File(label="ASS File")

        # ---------- Handlers ----------

        def do_transcribe(audio_path, language):
            if not audio_path:
                return "❌ Please upload a file first.", [], ""
            try:
                yield "Loading model...", [], ""
                lang_code = normalize_lang(language)
                yield "Transcribing…", [], ""
                words = transcribe_with_words(audio_path, lang_code)
                prev = " ".join(w["text"] for w in words[:120])
                if len(words) > 120:
                    prev += " …"
                return f"✅ Done. {len(words)} words.", words, prev
            except Exception as e:
                return f"❌ Error: {e}", [], ""

        def do_download_html(words):
            if not words:
                gr.Warning("Transcribe first.")
                return None
            try:
                return export_to_html_for_editing(words)
            except Exception as e:
                gr.Warning(f"Error creating HTML: {e}")
                return None

        def _resolve_uploaded_html(path: str) -> str | None:
            """Accept .html/.htm or Google Docs .zip (pick index.html or largest html)."""
            if path.lower().endswith((".html", ".htm")):
                return path
            if path.lower().endswith(".zip"):
                with zipfile.ZipFile(path, "r") as z:
                    htmls = [n for n in z.namelist() if n.lower().endswith((".html", ".htm"))]
                    if not htmls:
                        return None
                    pick = None
                    for n in htmls:
                        base = os.path.basename(n).lower()
                        if base in ("index.html", "index.htm"):
                            pick = n
                            break
                    if pick is None:
                        pick = max(htmls, key=lambda n: z.getinfo(n).file_size)
                    tmpdir = tempfile.mkdtemp()
                    return z.extract(pick, path=tmpdir)
            return None

        def do_import_html(file, original_words):
            if not file:
                return "❌ No file uploaded.", []
            if not original_words:
                return "❌ Transcribe first.", []
            try:
                path = _resolve_uploaded_html(file.name)
                if not path:
                    return "❌ Could not find an HTML file in the ZIP.", []
                edited = import_from_html(path, original_words)
                return f"✅ Imported {len(edited)} words with colors.", edited
            except Exception as e:
                return f"❌ Error importing HTML: {e}", []

        def do_export_srt(words, wpl):
            if not words:
                gr.Warning("Import edited file first.")
                return None
            srt = export_to_srt(words, int(wpl))
            return save_text_file(srt, ".srt")

        def do_export_ass(words, wpl, fnt, sz):
            if not words:
                gr.Warning("Import edited file first.")
                return None
            ass = export_to_ass(words, int(wpl), fnt, int(sz))
            return save_text_file(ass, ".ass")

        # Wire up
        btn_transcribe.click(
            fn=do_transcribe,
            inputs=[audio_in, lang],
            outputs=[status, word_segments_state, preview]
        )

        btn_dl_html.click(
            fn=do_download_html,
            inputs=[word_segments_state],
            outputs=[html_file]
        )

        btn_import.click(
            fn=do_import_html,
            inputs=[upload_html, word_segments_state],
            outputs=[import_status, edited_words_state]
        )

        btn_srt.click(
            fn=do_export_srt,
            inputs=[edited_words_state, words_per],
            outputs=[srt_out]
        )

        btn_ass.click(
            fn=do_export_ass,
            inputs=[edited_words_state, words_per, font, size],
            outputs=[ass_out]
        )

    return demo


if __name__ == "__main__":
    demo = create_app()
    # SSR can be left on; use share=True locally if you want a public link
    demo.launch()
