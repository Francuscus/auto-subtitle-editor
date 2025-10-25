# Language Learning Subtitle Editor (HF Space)
# Version 2.1  — stable transcription + external HTML editor workflow
# Default text color: blue (#1E88E5)

from __future__ import annotations
import os, re, json, tempfile, subprocess, shlex
from typing import List
from html import unescape

import gradio as gr
import torch
import whisperx


# -------------------------- App Config --------------------------

VERSION = "2.1"
TITLE = "Language Learning Subtitle Editor"
BANNER_COLOR = "#00BCD4"          # cyan
DEFAULT_TEXT_COLOR = "#1E88E5"    # blue

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


def seconds_to_srt_timestamp(t: float) -> str:
    t = max(float(t), 0.0)
    h = int(t // 3600); t -= h * 3600
    m = int(t // 60);   t -= m * 60
    s = int(t)
    ms = int(round((t - s) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def seconds_to_ass_timestamp(t: float) -> str:
    t = max(float(t), 0.0)
    h = int(t // 3600); t -= h * 3600
    m = int(t // 60);   t -= m * 60
    s = int(t)
    cs = int(round((t - s) * 100))  # centiseconds
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def save_text(content: str, suffix: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, f"subtitles{suffix}")
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return p


def save_json(obj, basename: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, basename)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return p


def looks_like_video(path: str) -> bool:
    ext = (os.path.splitext(path)[1] or "").lower()
    return ext in {".mp4", ".mov", ".mkv", ".webm", ".avi"}


# -------------------------- WhisperX (robust) --------------------------

_asr_model = None

def get_asr_model():
    """
    Load WhisperX with a safe compute_type.
    CPU -> int8 (fallback int16)
    GPU -> float16 (fallback float32)
    """
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"
    print(f"[v{VERSION}] WhisperX device={device}, compute_type={compute}")

    try:
        _asr_model = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        if "compute type" in str(e).lower():
            fb = "float32" if use_cuda else "int16"
            print(f"[v{VERSION}] Falling back to compute_type={fb}")
            _asr_model = whisperx.load_model("small", device=device, compute_type=fb)
        else:
            raise
    return _asr_model


def transcribe_with_words(audio_path: str, language_code: str | None) -> List[dict]:
    """
    Returns list of dicts: {start, end, text}
    If word timing is absent, approximates words within each segment.
    """
    model = get_asr_model()
    print("[ASR] Transcribing…")
    result = model.transcribe(audio_path, language=language_code)  # no word_timestamps kw
    segments = result.get("segments", []) or []

    words: List[dict] = []
    for seg in segments:
        s0 = float(seg.get("start", 0.0))
        s1 = float(seg.get("end", s0))
        if seg.get("words"):
            for w in seg["words"]:
                words.append({
                    "start": float(w.get("start", s0)),
                    "end": float(w.get("end", s1)),
                    "text": (w.get("word") or w.get("text") or "").strip()
                })
        else:
            text = (seg.get("text") or "").strip()
            tokens = [t for t in text.split() if t]
            if not tokens:
                continue
            dur = max(s1 - s0, 0.001)
            step = dur / len(tokens)
            for i, tkn in enumerate(tokens):
                t_start = s0 + i * step
                t_end = t_start + step
                words.append({"start": round(t_start, 3), "end": round(t_end, 3), "text": tkn})
    print(f"[ASR] {len(words)} words")
    return words


# -------------------------- HTML import (colors + text) --------------------------

def import_from_html(html_path: str, original_words: List[dict]) -> List[dict]:
    """
    Reads an edited HTML file (from editor.html or any WYSIWYG) and extracts
    words with their colors. Aligns by index to original timing.
    """
    from html.parser import HTMLParser

    class Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.words: List[dict] = []
            self.in_text_area = False
            self.cur_color = DEFAULT_TEXT_COLOR

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "div" and ("class" in attrs and "text-area" in attrs["class"]):
                self.in_text_area = True
            if tag == "span" and self.in_text_area:
                style = attrs.get("style", "")
                # Prefer explicit color; fall back to highlight as color if only that exists
                hex_match = re.search(r'#([0-9A-Fa-f]{6})', style)
                if hex_match:
                    self.cur_color = "#" + hex_match.group(1)
                elif "rgb" in style:
                    m = re.search(r'rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', style)
                    if m:
                        r, g, b = map(int, m.groups())
                        self.cur_color = f"#{r:02X}{g:02X}{b:02X}"

        def handle_endtag(self, tag):
            if tag == "div" and self.in_text_area:
                self.in_text_area = False

        def handle_data(self, data):
            if not self.in_text_area: return
            for raw in data.split():
                w = raw.strip()
                if w:
                    self.words.append({"text": w, "color": self.cur_color})

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    p = Parser()
    p.feed(html)

    # map color+text onto original timings by index
    out: List[dict] = []
    for i, ow in enumerate(original_words):
        if i < len(p.words):
            out.append({
                "start": ow["start"],
                "end": ow["end"],
                "text": p.words[i]["text"],
                "color": p.words[i]["color"] or DEFAULT_TEXT_COLOR
            })
        else:
            out.append({
                "start": ow["start"], "end": ow["end"],
                "text": ow["text"], "color": DEFAULT_TEXT_COLOR
            })
    return out


# -------------------------- SubRip / ASS exporters --------------------------

def export_to_srt(words: List[dict], words_per_line: int = 5) -> str:
    lines = []
    i, n = 0, 1
    while i < len(words):
        chunk = words[i:i+words_per_line]
        if not chunk: break
        start = seconds_to_srt_timestamp(chunk[0]["start"])
        end   = seconds_to_srt_timestamp(chunk[-1]["end"])
        text  = " ".join(w["text"] for w in chunk)
        lines.append(f"{n}\n{start} --> {end}\n{text}\n")
        i += words_per_line; n += 1
    return "\n".join(lines)


def _hex_to_ass_bgr(hex_color: str) -> str:
    # ASS color is &H AABBGGRR; we use full opacity (00 alpha)
    hx = (hex_color or DEFAULT_TEXT_COLOR).lstrip("#")
    if len(hx) != 6:
        hx = DEFAULT_TEXT_COLOR.lstrip("#")
    r = int(hx[0:2], 16); g = int(hx[2:4], 16); b = int(hx[4:6], 16)
    return f"&H00{b:02X}{g:02X}{r:02X}"


def export_to_ass(words: List[dict], words_per_line: int = 5,
                  font: str = "Arial", size: int = 36) -> str:
    header = f"""[Script Info]
; {TITLE} v{VERSION}
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,30,30,36,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    out_lines = [header]
    i = 0
    while i < len(words):
        chunk = words[i:i+words_per_line]
        if not chunk: break
        t0 = seconds_to_ass_timestamp(chunk[0]["start"])
        t1 = seconds_to_ass_timestamp(chunk[-1]["end"])

        parts = []
        for w in chunk:
            c = _hex_to_ass_bgr(w.get("color") or DEFAULT_TEXT_COLOR)
            txt = (w.get("text") or "").replace("{", "").replace("}", "")
            parts.append(f"{{\\c{c}}}{txt}")

        out_lines.append(f"Dialogue: 0,{t0},{t1},Default,,0,0,0,,{' '.join(parts)}")
        i += words_per_line
    return "\n".join(out_lines) + "\n"


# -------------------------- Optional: burn-in render (ffmpeg) --------------------------

def burn_in_ass(video_path: str, ass_path: str) -> str:
    """
    Burns ASS into a video using ffmpeg. Requires a *video* input.
    Returns path to MP4.
    """
    out_dir = tempfile.mkdtemp()
    out_path = os.path.join(out_dir, "subtitled.mp4")
    cmd = f'ffmpeg -y -i {shlex.quote(video_path)} -vf subtitles={shlex.quote(ass_path)} -c:a copy -c:v libx264 -pix_fmt yuv420p {shlex.quote(out_path)}'
    print("[ffmpeg]", cmd)
    subprocess.run(cmd, shell=True, check=True)
    return out_path


# -------------------------- Gradio UI --------------------------

def create_app():
    with gr.Blocks(theme=gr.themes.Soft(), title=TITLE) as app:

        gr.HTML(f"""
        <div style="background:{BANNER_COLOR};color:white;padding:16px;border-radius:10px;margin-bottom:10px;text-align:center">
          <h2 style="margin:2px 0">{TITLE}</h2>
          <div>Version {VERSION} — edit in a rich HTML page, then import and export SRT/ASS</div>
        </div>
        """)

        # State
        word_segments_state = gr.State([])   # raw words from ASR
        edited_words_state  = gr.State([])   # words after importing edited HTML

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Step 1 — Transcribe")
                audio_in = gr.Audio(label="Upload Audio/Video", type="filepath")
                language = gr.Dropdown(
                    choices=[("Auto-detect","auto"), ("Spanish","es"), ("Hungarian","hu"), ("English","en")],
                    value="auto", label="Language"
                )
                btn_transcribe = gr.Button("Transcribe", variant="primary")
                status = gr.Textbox(label="Status", value="Ready.", lines=3, interactive=False)
                preview = gr.Textbox(label="Transcript preview", lines=6, interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("### Step 2 — Edit in the built-in HTML page")
                btn_words_json = gr.Button("Download words.json")
                file_words_json = gr.File(label="words.json")
                gr.HTML(
                    '<a href="file=editor.html" target="_blank" '
                    'style="display:inline-block;margin-top:8px;background:#00BCD4;color:#fff;'
                    'padding:10px 14px;border-radius:10px;text-decoration:none">Open built-in editor ↗</a>'
                )
                gr.Markdown(
                    "- In the editor: **Load words.json**, style text (font, size, bold/italic/underline, text color, highlight), then **Export HTML**.\n"
                    "- Default text color starts as **blue**."
                )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Step 3 — Import your edited HTML")
                file_html = gr.File(label="Upload edited HTML", file_types=[".html", ".htm"])
                btn_import = gr.Button("Import", variant="primary")
                import_status = gr.Textbox(label="Import status", lines=2, interactive=False)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Step 4 — Export subtitles")
                words_per = gr.Slider( min=2, max=10, value=5, step=1, label="Words per subtitle line" )
                font = gr.Dropdown( choices=["Arial","Times New Roman","Georgia","Verdana","Courier New"], value="Arial", label="Font" )
                size = gr.Slider( min=20, max=72, value=36, step=2, label="Font size")
                with gr.Row():
                    btn_srt = gr.Button("Export SRT")
                    btn_ass = gr.Button("Export ASS (keeps colors)", variant="primary")
                file_srt = gr.File(label="SRT file")
                file_ass = gr.File(label="ASS file")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Optional — Burn subtitles into a video (FFmpeg)")
                gr.Markdown("> Works only if your uploaded file is a **video** (e.g., .mp4).")
                btn_render = gr.Button("Render MP4 with subtitles")
                file_mp4 = gr.File(label="Subtitled video (MP4)")

        # ---------------- Events ----------------

        def do_transcribe(path, lang):
            if not path:
                return "❌ Please upload audio or video.", [], ""
            try:
                yield "Loading model…", [], ""
                code = normalize_lang(lang)
                yield "Transcribing…", [], ""
                words = transcribe_with_words(path, code)
                prev = " ".join(w["text"] for w in words[:80])
                if len(words) > 80: prev += " …"
                msg = f"✅ Done. {len(words)} words."
                return msg, words, prev
            except Exception as e:
                return f"❌ Error: {e}", [], ""

        btn_transcribe.click(
            fn=do_transcribe,
            inputs=[audio_in, language],
            outputs=[status, word_segments_state, preview]
        )

        def do_download_words_json(words):
            if not words:
                gr.Warning("Transcribe first.")
                return None
            return save_json(words, "words.json")

        btn_words_json.click(
            fn=do_download_words_json,
            inputs=[word_segments_state],
            outputs=[file_words_json]
        )

        def do_import_html(f, orig):
            if not f: return [], "❌ No file uploaded."
            if not orig: return [], "❌ Transcribe first."
            try:
                edited = import_from_html(f.name, orig)
                return edited, f"✅ Imported {len(edited)} words with colors."
            except Exception as e:
                return [], f"❌ Error: {e}"

        btn_import.click(
            fn=do_import_html,
            inputs=[file_html, word_segments_state],
            outputs=[edited_words_state, import_status]
        )

        def do_export_srt(edited, n):
            if not edited:
                gr.Warning("Import an edited HTML first.")
                return None
            return save_text(export_to_srt(edited, int(n)), ".srt")

        def do_export_ass(edited, n, fnt, sz):
            if not edited:
                gr.Warning("Import an edited HTML first.")
                return None
            return save_text(export_to_ass(edited, int(n), fnt, int(sz)), ".ass")

        btn_srt.click(do_export_srt, [edited_words_state, words_per], [file_srt])
        btn_ass.click(do_export_ass, [edited_words_state, words_per, font, size], [file_ass])

        def do_render(video_path, edited, n, fnt, sz):
            if not video_path or not looks_like_video(video_path):
                gr.Warning("Please upload a *video* (e.g. MP4) in Step 1.")
                return None
            if not edited:
                gr.Warning("Import an edited HTML first.")
                return None
            # make a temp ASS, burn it in
            ass_path = save_text(export_to_ass(edited, int(n), fnt, int(sz)), ".ass")
            try:
                out = burn_in_ass(video_path, ass_path)
                return out
            except Exception as e:
                gr.Warning(f"FFmpeg error: {e}")
                return None

        btn_render.click(
            fn=do_render,
            inputs=[audio_in, edited_words_state, words_per, font, size],
            outputs=[file_mp4]
        )

    return app


if __name__ == "__main__":
    demo = create_app()
    demo.launch()
