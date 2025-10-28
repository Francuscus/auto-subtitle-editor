# Language Learning Subtitle Editor
# Version 2.3 — HTML/ZIP import + ASS/SRT + Burn MP4 from Audio
# Banner Color: #00BCD4 (Cyan)

import os
import re
import tempfile
from typing import List, Tuple, Optional
import zipfile
import pathlib
import subprocess
import shutil

import gradio as gr
import torch
import whisperx

# -------------------------- Config --------------------------

VERSION = "2.3"
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
    t = max(t, 0.0)
    h = int(t // 3600); t -= h * 3600
    m = int(t // 60);   t -= m * 60
    s = int(t);         ms = int(round((t - s) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def seconds_to_timestamp_ass(t: float) -> str:
    t = max(t, 0.0)
    h = int(t // 3600); t -= h * 3600
    m = int(t // 60);   t -= m * 60
    s = int(t);         cs = int(round((t - s) * 100))
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

# -------------------------- ASR Model --------------------------

_asr_model = None

def get_asr_model():
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
        if "compute type" in str(e).lower():
            fallback = "int16" if device == "cpu" else "float32"
            print(f"[v{VERSION}] Falling back to compute_type={fallback}")
            _asr_model = whisperx.load_model("small", device=device, compute_type=fallback)
        else:
            raise
    return _asr_model

# -------------------------- Transcription --------------------------

def transcribe_with_words(audio_path: str, language_code: Optional[str]) -> List[dict]:
    model = get_asr_model()
    print("[Transcribe] Starting transcription...")
    result = model.transcribe(audio_path, language=language_code)
    segments = result.get("segments", [])

    words: List[dict] = []
    for seg in segments:
        s_start = float(seg.get("start", 0.0))
        s_end = float(seg.get("end", max(s_start + 0.2, s_start)))
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                w_text = (w.get("word") or w.get("text") or "").strip()
                if not w_text:
                    continue
                w_start = float(w.get("start", s_start))
                w_end = float(w.get("end", s_end))
                words.append({"start": w_start, "end": w_end, "text": w_text})
        else:
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
    html = [HTML_TEMPLATE_HEAD]
    for w in word_segments:
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
    if not style:
        return None
    m = re.search(r"#([0-9A-Fa-f]{6})", style)
    if m:
        return "#" + m.group(1).upper()
    m = re.search(r"rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", style)
    if m:
        r, g, b = map(int, m.groups())
        return f"#{r:02X}{g:02X}{b:02X}"
    named = {
        "red": "#FF0000", "yellow": "#FFFF00", "cyan": "#00FFFF", "aqua": "#00FFFF",
        "green": "#00FF00", "blue": "#0000FF", "magenta": "#FF00FF", "fuchsia": "#FF00FF",
        "black": "#000000", "white": "#FFFFFF", "purple": "#800080",
    }
    m = re.search(r"color\s*:\s*([a-zA-Z]+)", style)
    if m and m.group(1).lower() in named:
        return named[m.group(1).lower()]
    m = re.search(r"background(?:-color)?\s*:\s*#([0-9A-Fa-f]{6})", style)
    if m:
        return "#" + m.group(1).upper()
    m = re.search(r"background(?:-color)?\s*:\s*rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", style)
    if m:
        r, g, b = map(int, m.groups())
        return f"#{r:02X}{g:02X}{b:02X}"
    return None

def _parse_html_words(html_text: str) -> List[Tuple[str, Optional[str]]]:
    from html.parser import HTMLParser

    class Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_text_area = False
            self.stack_styles: List[str] = []
            self.words: List[Tuple[str, Optional[str]]] = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "div" and attrs.get("class", "") == "text-area":
                self.in_text_area = True
            if self.in_text_area and tag in ("span", "font", "mark"):
                style = attrs.get("style", "")
                color = _css_color_to_hex(style)
                if not color and "color" in attrs:
                    color = _css_color_to_hex(f"color:{attrs['color']}")
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
            for tok in re.split(r"\s+", data):
                tok = tok.strip()
                if not tok:
                    continue
                color = None
                for c in reversed(self.stack_styles):
                    if c:
                        color = c
                        break
                if not color:
                    color = DEFAULT_SAMPLE_TEXT_COLOR
                self.words.append((tok, color))

    p = Parser()
    p.feed(html_text)
    return p.words

def import_from_html(html_path: str, original_words: List[dict]) -> List[dict]:
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    parsed = _parse_html_words(html)

    edited = []
    for i, ow in enumerate(original_words):
        if i < len(parsed):
            txt, col = parsed[i]
            edited.append({"start": ow["start"], "end": ow["end"], "text": txt, "color": col})
        else:
            edited.append({"start": ow["start"], "end": ow["end"], "text": ow["text"], "color": DEFAULT_SAMPLE_TEXT_COLOR})
    return edited

def import_from_zip(zip_path: str, original_words: List[dict]) -> List[dict]:
    # Find first .html/.htm inside and parse it
    with zipfile.ZipFile(zip_path, "r") as z:
        html_names = [n for n in z.namelist() if n.lower().endswith((".html", ".htm"))]
        if not html_names:
            raise ValueError("No .html file found inside the ZIP.")
        # Prefer top-level or the first one
        name = sorted(html_names, key=lambda x: (x.count("/"), len(x)))[0]
        with z.open(name) as f:
            html = f.read().decode("utf-8", errors="ignore")
    parsed = _parse_html_words(html)

    edited = []
    for i, ow in enumerate(original_words):
        if i < len(parsed):
            txt, col = parsed[i]
            edited.append({"start": ow["start"], "end": ow["end"], "text": txt, "color": col})
        else:
            edited.append({"start": ow["start"], "end": ow["end"], "text": ow["text"], "color": DEFAULT_SAMPLE_TEXT_COLOR})
    return edited

# -------------------------- SubRip (SRT) / ASS --------------------------

def export_to_srt(words: List[dict], words_per_line: int = 5) -> str:
    i = 0; n = 1; out = []
    while i < len(words):
        chunk = words[i:i + words_per_line]
        start = seconds_to_timestamp_srt(chunk[0]["start"])
        end = seconds_to_timestamp_srt(chunk[-1]["end"])
        text = " ".join(w["text"] for w in chunk)
        out.append(f"{n}\n{start} --> {end}\n{text}\n")
        i += words_per_line; n += 1
    return "\n".join(out)

def _hex_to_ass_bgr(hex_color: str) -> str:
    if not hex_color or not hex_color.startswith("#") or len(hex_color) != 7:
        hex_color = "#FFFFFF"
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"&H00{b:02X}{g:02X}{r:02X}"

def _parse_char_level_colors(text_with_html: str) -> List[Tuple[str, str]]:
    """
    Parse HTML with character-level styling and return list of (character, color) tuples.
    """
    from html.parser import HTMLParser

    class CharParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.chars: List[Tuple[str, str]] = []
            self.color_stack: List[str] = []

        def handle_starttag(self, tag, attrs):
            if tag in ("span", "font"):
                attrs_dict = dict(attrs)
                style = attrs_dict.get("style", "")
                color = _css_color_to_hex(style)
                if not color and "color" in attrs_dict:
                    color = _css_color_to_hex(f"color:{attrs_dict['color']}")
                self.color_stack.append(color or DEFAULT_SAMPLE_TEXT_COLOR)

        def handle_endtag(self, tag):
            if tag in ("span", "font") and self.color_stack:
                self.color_stack.pop()

        def handle_data(self, data):
            current_color = self.color_stack[-1] if self.color_stack else DEFAULT_SAMPLE_TEXT_COLOR
            for char in data:
                self.chars.append((char, current_color))

    parser = CharParser()
    parser.feed(text_with_html)
    return parser.chars

def export_to_ass(words: List[dict], words_per_line: int = 5, font="Arial", size=36) -> str:
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
            # Check if word has character-level styling (HTML)
            if "html" in w and w["html"]:
                # Parse character-level colors
                chars = _parse_char_level_colors(w["html"])
                # Group consecutive characters with same color
                if chars:
                    current_color = chars[0][1]
                    current_text = chars[0][0]
                    for char, color in chars[1:]:
                        if color == current_color:
                            current_text += char
                        else:
                            col_ass = _hex_to_ass_bgr(current_color)
                            parts.append(f"{{\\c{col_ass}}}{current_text}")
                            current_color = color
                            current_text = char
                    # Add last group
                    col_ass = _hex_to_ass_bgr(current_color)
                    parts.append(f"{{\\c{col_ass}}}{current_text}")
            else:
                # Word-level color (backward compatible)
                col = _hex_to_ass_bgr(w.get("color", DEFAULT_SAMPLE_TEXT_COLOR))
                parts.append(f"{{\\c{col}}}{w['text']}")

            # Add space between words
            if w != chunk[-1]:
                parts.append(" ")

        text = "".join(parts)
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
        i += words_per_line
    return "\n".join(lines) + "\n"

def _save_temp(content: str, ext: str) -> str:
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, f"subtitles{ext}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

# -------------------------- FFmpeg helpers (burn to MP4) --------------------------

def _ffprobe_duration(path: str) -> float:
    """Return media duration in seconds using ffprobe."""
    if not shutil.which("ffprobe"):
        return 0.0
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float((p.stdout or "0").strip())
    except:
        return 0.0

def _escape_ass_for_filter(p: str) -> str:
    """
    Escape a filesystem path for ffmpeg's subtitles filter.
    """
    p = str(pathlib.Path(p).resolve())
    p = p.replace("\\", "\\\\").replace(":", "\\:")
    return "'" + p.replace("'", r"'\''") + "'"

def _make_color_canvas(out_path: str, width: int, height: int, seconds: float, bg_hex: str, fps: int = 30):
    """
    Create a solid-color video of given duration.
    """
    if not bg_hex or not bg_hex.startswith("#") or len(bg_hex) != 7:
        bg_hex = "#000000"
    color_arg = "0x" + bg_hex[1:]
    duration = max(0.1, seconds)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color={color_arg}:s={width}x{height}:r={fps}",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        out_path,
    ]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def burn_ass_on_canvas_with_audio(audio_path: str, ass_path: str, bg_hex="#000000", size="1280x720", fps=30) -> tuple[Optional[str], str]:
    """
    Create a solid-color canvas matching the audio duration, burn ASS on it, and mux audio -> MP4.
    Returns (mp4_path or None, log).
    """
    if not shutil.which("ffmpeg"):
        return None, "ffmpeg not found."

    if not os.path.exists(audio_path):
        return None, "Audio file not found."
    if not os.path.exists(ass_path):
        return None, "ASS file not found."

    try:
        w, h = [int(x) for x in size.lower().split("x")]
    except:
        w, h = 1280, 720

    dur = _ffprobe_duration(audio_path)
    if dur <= 0:
        return None, "Could not read audio duration (ffprobe)."

    tmpdir = tempfile.mkdtemp()
    canvas_mp4 = os.path.join(tmpdir, "canvas.mp4")
    out_mp4 = os.path.join(tmpdir, "subtitled.mp4")

    # 1) make canvas
    p1 = _make_color_canvas(canvas_mp4, w, h, dur, bg_hex, fps)
    if p1.returncode != 0 or not os.path.exists(canvas_mp4):
        return None, "Failed to create canvas:\n" + (p1.stderr or "")

    # 2) burn subtitles on canvas
    ass_escaped = _escape_ass_for_filter(ass_path)
    burned_mp4 = os.path.join(tmpdir, "burned.mp4")
    cmd_burn = [
        "ffmpeg", "-y",
        "-i", canvas_mp4,
        "-vf", f"subtitles={ass_escaped}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-an",
        burned_mp4
    ]
    p2 = subprocess.run(cmd_burn, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p2.returncode != 0 or not os.path.exists(burned_mp4):
        return None, "Burn step failed (subtitles filter missing?).\n" + (p2.stderr or "")

    # 3) mux original audio with the burned video
    cmd_mux = [
        "ffmpeg", "-y",
        "-i", burned_mp4, "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        out_mp4
    ]
    p3 = subprocess.run(cmd_mux, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p3.returncode != 0 or not os.path.exists(out_mp4):
        return None, "Mux step failed:\n" + (p3.stderr or "")

    return out_mp4, "✅ Created MP4 with burned subs and original audio."

# -------------------------- Gradio App --------------------------

CUSTOM_CSS = """
#lyric-editor {
    min-height: 400px;
    max-height: 600px;
    overflow-y: auto;
    border: 2px solid #00BCD4;
    border-radius: 8px;
    padding: 16px;
    background: white;
    font-size: 18px;
    line-height: 1.8;
    white-space: pre-wrap;
    word-wrap: break-word;
}

#lyric-editor:focus {
    outline: none;
    border-color: #0097A7;
    box-shadow: 0 0 0 3px rgba(0, 188, 212, 0.1);
}

#preview-canvas {
    width: 100%;
    height: 400px;
    background: #000;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

#timeline-container {
    background: #f5f5f5;
    padding: 16px;
    border-radius: 8px;
    margin-top: 16px;
}

#timeline-bar {
    width: 100%;
    height: 60px;
    background: #e0e0e0;
    border-radius: 4px;
    position: relative;
    cursor: pointer;
    margin-bottom: 12px;
}

#timeline-progress {
    height: 100%;
    background: linear-gradient(90deg, #00BCD4 0%, #0097A7 100%);
    border-radius: 4px;
    position: absolute;
    top: 0;
    left: 0;
    width: 0%;
}

#timeline-scrubber {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    width: 4px;
    height: 100%;
    background: #FF5722;
    cursor: grab;
    z-index: 10;
}

#timeline-scrubber:active {
    cursor: grabbing;
}

.playback-controls {
    display: flex;
    align-items: center;
    gap: 12px;
    justify-content: center;
}

.toolbar-btn {
    padding: 8px 16px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: white;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.2s;
}

.toolbar-btn:hover {
    background: #f0f0f0;
    border-color: #00BCD4;
}

.toolbar-btn.active {
    background: #00BCD4;
    color: white;
    border-color: #00BCD4;
}

.color-swatch {
    width: 32px;
    height: 32px;
    border-radius: 4px;
    border: 2px solid #ddd;
    cursor: pointer;
    display: inline-block;
}

#editor-toolbar {
    display: flex;
    gap: 8px;
    padding: 12px;
    background: #f9f9f9;
    border-radius: 8px;
    margin-bottom: 12px;
    flex-wrap: wrap;
    align-items: center;
}

.char-styled {
    display: inline;
}
"""

EDITOR_JS = """
<script>
let currentTime = 0;
let duration = 0;
let isPlaying = false;
let wordTimings = [];
let selectedText = '';
let editorContent = '';

// Initialize editor
function initEditor() {
    const editor = document.getElementById('lyric-editor');
    if (!editor) return;

    editor.contentEditable = true;
    editor.addEventListener('input', onEditorChange);
    editor.addEventListener('mouseup', onTextSelect);
    editor.addEventListener('keyup', onTextSelect);
}

// Handle text selection
function onTextSelect() {
    const selection = window.getSelection();
    selectedText = selection.toString();
    if (selectedText) {
        console.log('Selected:', selectedText);
    }
}

// Handle editor changes
function onEditorChange() {
    const editor = document.getElementById('lyric-editor');
    if (editor) {
        editorContent = editor.innerHTML;
        console.log('Editor content changed');
    }
}

// Get editor HTML content (for syncing with backend)
function getEditorHTML() {
    const editor = document.getElementById('lyric-editor');
    return editor ? editor.innerHTML : '';
}

// Apply color to selected text
function applyColor(color) {
    document.execCommand('styleWithCSS', false, true);
    document.execCommand('foreColor', false, color);

    // Update editor content
    onEditorChange();
}

// Apply IPA accent to selected text
function applyAccent(accentNum, accentSymbol) {
    const selection = window.getSelection();
    if (!selection.rangeCount || !accentSymbol) return;

    const range = selection.getRangeAt(0);
    const selectedText = range.toString();

    if (!selectedText) {
        alert('Please select some text first to apply the accent.');
        return;
    }

    // Insert the accent symbol after the selected text
    // This preserves the selection and adds the IPA accent
    const newText = selectedText + accentSymbol;

    // Replace selection with text + accent
    range.deleteContents();
    const textNode = document.createTextNode(newText);
    range.insertNode(textNode);

    // Update editor content
    onEditorChange();

    // Log for debugging
    console.log('Applied accent', accentNum, ':', accentSymbol, 'to', selectedText);
}

// Apply font size to selected text
function applyFontSize(size) {
    const selection = window.getSelection();
    if (!selection.rangeCount) return;

    const range = selection.getRangeAt(0);
    const span = document.createElement('span');
    span.style.fontSize = size + 'px';

    try {
        range.surroundContents(span);
    } catch (e) {
        console.error('Could not apply font size:', e);
    }

    selection.removeAllRanges();
    onEditorChange();
}

// Extract words with their HTML styling
function extractStyledWords() {
    const editor = document.getElementById('lyric-editor');
    if (!editor) return [];

    const words = [];
    const wordSpans = editor.querySelectorAll('.word');

    wordSpans.forEach(span => {
        const start = parseFloat(span.getAttribute('data-start') || 0);
        const end = parseFloat(span.getAttribute('data-end') || 0);
        const html = span.innerHTML;
        const text = span.textContent;

        words.push({
            start: start,
            end: end,
            text: text,
            html: html
        });
    });

    return words;
}

// Update word timings from backend
function setWordTimings(words) {
    wordTimings = words;
    if (words.length > 0) {
        duration = Math.max(...words.map(w => w.end));
        updateTimeDisplay();
    }
}

// Timeline scrubber
function initTimeline() {
    const timeline = document.getElementById('timeline-bar');
    const scrubber = document.getElementById('timeline-scrubber');

    if (!timeline || !scrubber) return;

    let isDragging = false;

    timeline.addEventListener('click', (e) => {
        const rect = timeline.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percent = (x / rect.width) * 100;
        updateTimeline(percent);
    });

    scrubber.addEventListener('mousedown', (e) => {
        isDragging = true;
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const timeline = document.getElementById('timeline-bar');
        const rect = timeline.getBoundingClientRect();
        const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
        const percent = (x / rect.width) * 100;
        updateTimeline(percent);
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
    });
}

function updateTimeline(percent) {
    const progress = document.getElementById('timeline-progress');
    const scrubber = document.getElementById('timeline-scrubber');

    if (progress) progress.style.width = percent + '%';
    if (scrubber) scrubber.style.left = percent + '%';

    currentTime = (percent / 100) * duration;
    updateTimeDisplay();
    updatePreview();
}

function updateTimeDisplay() {
    const display = document.getElementById('time-display');
    if (display) {
        const current = formatTime(currentTime);
        const total = formatTime(duration);
        display.textContent = current + ' / ' + total;
    }
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return String(mins).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
}

function updatePreview() {
    // Find current words based on time
    const currentWords = wordTimings.filter(w =>
        currentTime >= w.start && currentTime <= w.end
    );

    const preview = document.getElementById('preview-canvas');
    if (preview) {
        if (currentWords.length > 0) {
            const text = currentWords.map(w => {
                const color = w.color || '#FFFFFF';
                return '<span style="color: ' + color + '; font-size: 48px; margin: 0 8px;">' + w.text + '</span>';
            }).join(' ');
            preview.innerHTML = '<div>' + text + '</div>';
        } else {
            preview.innerHTML = '<div style="color: #888;">No lyrics at this time</div>';
        }
    }
}

// Play/Pause controls
function togglePlayback() {
    isPlaying = !isPlaying;
    const btn = document.getElementById('play-btn');
    if (btn) {
        btn.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
    }

    if (isPlaying) {
        playTimeline();
    }
}

function playTimeline() {
    if (!isPlaying) return;

    const step = 0.1; // 100ms steps
    currentTime += step;

    if (currentTime >= duration) {
        currentTime = duration;
        isPlaying = false;
        const btn = document.getElementById('play-btn');
        if (btn) btn.textContent = '▶ Play';
        return;
    }

    const percent = (currentTime / duration) * 100;
    updateTimeline(percent);

    setTimeout(playTimeline, 100);
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    initEditor();
    initTimeline();
});

// Make functions globally accessible
window.applyColor = applyColor;
window.applyAccent = applyAccent;
window.applyFontSize = applyFontSize;
window.togglePlayback = togglePlayback;
window.getEditorHTML = getEditorHTML;
window.extractStyledWords = extractStyledWords;
window.setWordTimings = setWordTimings;
</script>
"""

def create_app():
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title=f"Lyric Video Editor v{VERSION}",
        css=CUSTOM_CSS
    ) as demo:

        gr.HTML(
            f"""
            <div style="background:{BANNER_COLOR};color:white;padding:18px;border-radius:12px;margin-bottom:16px;text-align:center">
              <div style="font-size:22px;font-weight:700;">Lyric Video Editor</div>
              <div style="opacity:0.9;">Version {VERSION} — Create Beautiful Lyric Videos</div>
            </div>
            {EDITOR_JS}
            """
        )

        # States
        word_segments_state = gr.State([])     # original words (timestamps)
        edited_words_state = gr.State([])      # edited (with colors)
        audio_state = gr.State(None)           # current audio file
        status_box = gr.Textbox(label="Status", value="Ready.", interactive=False, lines=2)

        # Main Layout
        with gr.Row():
            # Left Column: Controls
            with gr.Column(scale=3):
                gr.Markdown("### 1️⃣ Upload & Transcribe")
                audio_input = gr.Audio(label="Upload MP3/Audio", type="filepath")
                language_dropdown = gr.Dropdown(
                    choices=[("Auto-detect", "auto"), ("Spanish", "es"), ("Hungarian", "hu"), ("English", "en")],
                    value="auto",
                    label="Language"
                )
                transcribe_btn = gr.Button("🎵 Transcribe Audio", variant="primary", size="lg")

                gr.Markdown("---")
                gr.Markdown("### 🎯 IPA Accent Configuration")
                gr.Markdown("*Configure your 4 custom accents (will sync with acentos program)*")

                with gr.Row():
                    accent1_label = gr.Textbox(label="Accent 1 Label", value="Accent 1", scale=1)
                    accent1_symbol = gr.Textbox(label="Symbol/Text", value="́", scale=1, placeholder="e.g., ́ or ˈ")

                with gr.Row():
                    accent2_label = gr.Textbox(label="Accent 2 Label", value="Accent 2", scale=1)
                    accent2_symbol = gr.Textbox(label="Symbol/Text", value="̀", scale=1, placeholder="e.g., ̀ or ˌ")

                with gr.Row():
                    accent3_label = gr.Textbox(label="Accent 3 Label", value="Accent 3", scale=1)
                    accent3_symbol = gr.Textbox(label="Symbol/Text", value="̂", scale=1, placeholder="e.g., ̂ or ː")

                with gr.Row():
                    accent4_label = gr.Textbox(label="Accent 4 Label", value="Accent 4", scale=1)
                    accent4_symbol = gr.Textbox(label="Symbol/Text", value="̃", scale=1, placeholder="e.g., ̃ or ʰ")

                update_accents_btn = gr.Button("Update Accent Buttons", size="sm")

                gr.Markdown("---")
                gr.Markdown("### ⚙️ Settings")

                words_per = gr.Slider(minimum=1, maximum=15, value=5, step=1, label="Words per line")
                font_family = gr.Dropdown(
                    choices=["Arial", "Times New Roman", "Courier New", "Georgia", "Verdana", "Impact"],
                    value="Arial", label="Font Family"
                )
                font_size = gr.Slider(minimum=20, maximum=96, value=48, step=2, label="Font Size")

                bg_color = gr.ColorPicker(value="#000000", label="Background Color")
                size_dd = gr.Dropdown(
                    choices=["1280x720", "1920x1080", "1080x1920", "1080x1080"],
                    value="1280x720",
                    label="Canvas Size"
                )

                gr.Markdown("---")
                export_mp4_btn = gr.Button("🎬 Export MP4 Video", variant="primary", size="lg")
                exported_video = gr.Video(label="Your Lyric Video")

            # Center Column: Preview
            with gr.Column(scale=5):
                gr.Markdown("### 🎥 Preview")
                preview_html = gr.HTML(
                    """
                    <div id="preview-canvas">
                        <div style="color: #888;">Upload audio and transcribe to see preview</div>
                    </div>
                    """
                )

                gr.Markdown("### ✏️ Edit Lyrics")
                # Toolbar (will be updated dynamically with accent buttons)
                def create_toolbar_html(a1_label="Accent 1", a1_sym="́", a2_label="Accent 2", a2_sym="̀",
                                       a3_label="Accent 3", a3_sym="̂", a4_label="Accent 4", a4_sym="̃"):
                    return f"""
                    <div id="editor-toolbar">
                        <div style="font-weight: bold; margin-right: 8px;">Format:</div>
                        <button class="toolbar-btn" onclick="applyColor('#FF0000')" title="Red">
                            <span style="color: #FF0000;">●</span> Red
                        </button>
                        <button class="toolbar-btn" onclick="applyColor('#FFFF00')" title="Yellow" style="background: #FFFF00;">
                            <span style="color: #000;">●</span> Yellow
                        </button>
                        <button class="toolbar-btn" onclick="applyColor('#00FF00')" title="Green">
                            <span style="color: #00FF00;">●</span> Green
                        </button>
                        <button class="toolbar-btn" onclick="applyColor('#00FFFF')" title="Cyan">
                            <span style="color: #00FFFF;">●</span> Cyan
                        </button>
                        <button class="toolbar-btn" onclick="applyColor('#0000FF')" title="Blue">
                            <span style="color: #0000FF;">●</span> Blue
                        </button>
                        <button class="toolbar-btn" onclick="applyColor('#FF00FF')" title="Magenta">
                            <span style="color: #FF00FF;">●</span> Magenta
                        </button>
                        <button class="toolbar-btn" onclick="applyColor('#FFFFFF')" title="White">
                            <span style="color: #FFFFFF; text-shadow: 0 0 1px #000;">●</span> White
                        </button>
                        <div style="width: 1px; height: 30px; background: #ddd; margin: 0 8px;"></div>
                        <button class="toolbar-btn" onclick="applyFontSize(24)" title="Small">Small</button>
                        <button class="toolbar-btn" onclick="applyFontSize(36)" title="Medium">Medium</button>
                        <button class="toolbar-btn" onclick="applyFontSize(48)" title="Large">Large</button>
                        <button class="toolbar-btn" onclick="applyFontSize(72)" title="Extra Large">XL</button>
                        <div style="width: 1px; height: 30px; background: #ddd; margin: 0 8px;"></div>
                        <div style="font-weight: bold; margin: 0 8px;">IPA Accents:</div>
                        <button class="toolbar-btn" onclick="applyAccent(1, '{a1_sym}')" title="{a1_label}">
                            {a1_label}
                        </button>
                        <button class="toolbar-btn" onclick="applyAccent(2, '{a2_sym}')" title="{a2_label}">
                            {a2_label}
                        </button>
                        <button class="toolbar-btn" onclick="applyAccent(3, '{a3_sym}')" title="{a3_label}">
                            {a3_label}
                        </button>
                        <button class="toolbar-btn" onclick="applyAccent(4, '{a4_sym}')" title="{a4_label}">
                            {a4_label}
                        </button>
                    </div>
                    """

                toolbar_html = gr.HTML(create_toolbar_html())

                # Rich text editor
                editor_html = gr.HTML(
                    '<div id="lyric-editor">Select and transcribe audio to begin editing...</div>',
                    elem_id="lyric-editor"
                )

            # Right Column: Word list/timing info
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Info")
                transcript_preview = gr.Textbox(
                    label="Transcribed Text",
                    lines=12,
                    interactive=False,
                    placeholder="Transcribed lyrics will appear here..."
                )

                gr.Markdown("### 🎨 Quick Actions")
                update_preview_btn = gr.Button("🔄 Update Preview")
                clear_formatting_btn = gr.Button("🧹 Clear All Formatting")

                gr.Markdown("---")
                gr.Markdown("### 🔗 Acentos Program Integration")
                gr.Markdown("*Import/Export to sync with your IPA acentos program*")

                with gr.Row():
                    export_config_btn = gr.Button("📤 Export Config", size="sm")
                    import_config_btn = gr.Button("📥 Import Config", size="sm")

                config_file = gr.File(label="Accent Configuration (JSON)")

                with gr.Row():
                    export_lyrics_btn = gr.Button("📤 Export Lyrics+Timing", size="sm")
                    import_lyrics_btn = gr.Button("📥 Import Lyrics+Timing", size="sm")

                lyrics_file = gr.File(label="Accented Lyrics (JSON)")

        # Timeline at the bottom
        gr.Markdown("---")
        gr.Markdown("### ⏱️ Timeline")
        timeline_html = gr.HTML(
            """
            <div id="timeline-container">
                <div id="timeline-bar">
                    <div id="timeline-progress"></div>
                    <div id="timeline-scrubber" style="left: 0%;"></div>
                </div>
                <div class="playback-controls">
                    <button id="play-btn" class="toolbar-btn" onclick="togglePlayback()">▶ Play</button>
                    <button class="toolbar-btn" onclick="updateTimeline(0)">⏮ Start</button>
                    <button class="toolbar-btn" onclick="updateTimeline(100)">⏭ End</button>
                    <span id="time-display" style="font-family: monospace; margin-left: 12px;">00:00 / 00:00</span>
                </div>
            </div>
            """
        )

        # ---------- Handlers ----------

        def do_transcribe(audio_path, lang_sel):
            if not audio_path:
                return (
                    "❌ Error: no audio file provided.",
                    [],
                    [],
                    "",
                    '<div id="lyric-editor">No audio to transcribe.</div>',
                    audio_path
                )
            try:
                lang_code = normalize_lang(lang_sel)
                yield (
                    f"Loading model…\nLanguage: {lang_sel}",
                    [],
                    [],
                    "",
                    '<div id="lyric-editor">Transcribing...</div>',
                    audio_path
                )

                words = transcribe_with_words(audio_path, lang_code)

                # Create preview text
                preview = " ".join(w["text"] for w in words[:120])
                if len(words) > 120:
                    preview += " …"

                # Create editable HTML for the editor
                editor_content = '<div id="lyric-editor" contenteditable="true">\n'
                for w in words:
                    editor_content += f'<span class="word" data-start="{w["start"]:.3f}" data-end="{w["end"]:.3f}" style="color: {DEFAULT_SAMPLE_TEXT_COLOR};">{w["text"]}</span> '
                editor_content += '\n</div>'

                # Initialize edited words with default colors
                edited = [{"start": w["start"], "end": w["end"], "text": w["text"], "color": DEFAULT_SAMPLE_TEXT_COLOR} for w in words]

                yield (
                    f"✅ Transcribed {len(words)} words.",
                    words,
                    edited,
                    preview,
                    editor_content,
                    audio_path
                )
            except Exception as e:
                yield (
                    f"❌ Error during transcription: {e}",
                    [],
                    [],
                    "",
                    '<div id="lyric-editor">Error during transcription.</div>',
                    None
                )

        transcribe_btn.click(
            fn=do_transcribe,
            inputs=[audio_input, language_dropdown],
            outputs=[status_box, word_segments_state, edited_words_state, transcript_preview, editor_html, audio_state]
        )

        def update_accent_buttons(a1_label, a1_sym, a2_label, a2_sym, a3_label, a3_sym, a4_label, a4_sym):
            """Update the toolbar with new accent button labels and symbols"""
            return create_toolbar_html(a1_label, a1_sym, a2_label, a2_sym, a3_label, a3_sym, a4_label, a4_sym)

        update_accents_btn.click(
            fn=update_accent_buttons,
            inputs=[accent1_label, accent1_symbol, accent2_label, accent2_symbol,
                   accent3_label, accent3_symbol, accent4_label, accent4_symbol],
            outputs=[toolbar_html]
        )

        def update_preview_display(words):
            """Update the preview canvas with current word styling"""
            if not words:
                return '<div id="preview-canvas"><div style="color: #888;">No lyrics to preview</div></div>'

            # Generate a sample preview showing first few words with their colors
            sample_html = '<div id="preview-canvas"><div style="line-height: 1.5;">'
            for w in words[:10]:  # Show first 10 words
                color = w.get("color", DEFAULT_SAMPLE_TEXT_COLOR)
                sample_html += f'<span style="color: {color}; font-size: 36px; margin: 0 8px;">{w["text"]}</span>'
            if len(words) > 10:
                sample_html += '<span style="color: #888; font-size: 24px;">...</span>'
            sample_html += '</div></div>'
            return sample_html

        update_preview_btn.click(
            fn=update_preview_display,
            inputs=[edited_words_state],
            outputs=[preview_html]
        )

        def clear_all_formatting(words):
            """Reset all words to default color"""
            if not words:
                return []

            cleared = []
            for w in words:
                cleared.append({
                    "start": w["start"],
                    "end": w["end"],
                    "text": w["text"],
                    "color": DEFAULT_SAMPLE_TEXT_COLOR
                })

            # Recreate editor HTML
            editor_content = '<div id="lyric-editor" contenteditable="true">\n'
            for w in cleared:
                editor_content += f'<span class="word" data-start="{w["start"]:.3f}" data-end="{w["end"]:.3f}" style="color: {DEFAULT_SAMPLE_TEXT_COLOR};">{w["text"]}</span> '
            editor_content += '\n</div>'

            return cleared, editor_content

        clear_formatting_btn.click(
            fn=clear_all_formatting,
            inputs=[edited_words_state],
            outputs=[edited_words_state, editor_html]
        )

        def handle_export_mp4(audio_path, edited_words, n_words, font, size, bg_hex, canvas_size):
            """Export the final MP4 video with lyrics"""
            if not audio_path:
                gr.Warning("Upload audio first.")
                return None, "❌ No audio file provided."

            if not edited_words:
                gr.Warning("Transcribe audio first.")
                return None, "❌ No lyrics to export."

            try:
                # Generate ASS subtitle file
                ass_content = export_to_ass(edited_words, int(n_words), font, int(size))
                ass_path = _save_temp(ass_content, ".ass")

                # Create MP4 with burned subtitles
                out_path, log = burn_ass_on_canvas_with_audio(
                    audio_path,
                    ass_path,
                    bg_hex,
                    canvas_size,
                    fps=30
                )

                if out_path:
                    return out_path, log
                else:
                    return None, log

            except Exception as e:
                return None, f"❌ Error creating video: {e}"

        export_mp4_btn.click(
            fn=handle_export_mp4,
            inputs=[audio_state, edited_words_state, words_per, font_family, font_size, bg_color, size_dd],
            outputs=[exported_video, status_box]
        )

        # Acentos Integration Handlers
        def export_accent_config(a1_label, a1_sym, a2_label, a2_sym, a3_label, a3_sym, a4_label, a4_sym):
            """Export accent configuration as JSON for acentos program"""
            import json
            config = {
                "version": "1.0",
                "accents": [
                    {"id": 1, "label": a1_label, "symbol": a1_sym},
                    {"id": 2, "label": a2_label, "symbol": a2_sym},
                    {"id": 3, "label": a3_label, "symbol": a3_sym},
                    {"id": 4, "label": a4_label, "symbol": a4_sym}
                ]
            }
            tmpdir = tempfile.mkdtemp()
            path = os.path.join(tmpdir, "accent_config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return path

        export_config_btn.click(
            fn=export_accent_config,
            inputs=[accent1_label, accent1_symbol, accent2_label, accent2_symbol,
                   accent3_label, accent3_symbol, accent4_label, accent4_symbol],
            outputs=[config_file]
        )

        def import_accent_config(config_json_file):
            """Import accent configuration from acentos program"""
            import json
            if not config_json_file:
                gr.Warning("Please upload a config file first.")
                return ["Accent 1", "́", "Accent 2", "̀", "Accent 3", "̂", "Accent 4", "̃"]

            try:
                with open(config_json_file.name, "r", encoding="utf-8") as f:
                    config = json.load(f)

                accents = config.get("accents", [])
                results = []
                for i in range(4):
                    if i < len(accents):
                        results.append(accents[i].get("label", f"Accent {i+1}"))
                        results.append(accents[i].get("symbol", ""))
                    else:
                        results.append(f"Accent {i+1}")
                        results.append("")
                return results
            except Exception as e:
                gr.Warning(f"Error importing config: {e}")
                return ["Accent 1", "́", "Accent 2", "̀", "Accent 3", "̂", "Accent 4", "̃"]

        import_config_btn.click(
            fn=import_accent_config,
            inputs=[config_file],
            outputs=[accent1_label, accent1_symbol, accent2_label, accent2_symbol,
                    accent3_label, accent3_symbol, accent4_label, accent4_symbol]
        )

        def export_lyrics_with_timing(edited_words):
            """Export lyrics with IPA accents and timing data as JSON"""
            import json
            if not edited_words:
                gr.Warning("No lyrics to export. Transcribe first.")
                return None

            data = {
                "version": "1.0",
                "words": []
            }

            for w in edited_words:
                word_data = {
                    "start": w["start"],
                    "end": w["end"],
                    "text": w["text"],
                    "color": w.get("color", DEFAULT_SAMPLE_TEXT_COLOR)
                }
                if "html" in w and w["html"]:
                    word_data["html"] = w["html"]
                data["words"].append(word_data)

            tmpdir = tempfile.mkdtemp()
            path = os.path.join(tmpdir, "lyrics_with_timing.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return path

        export_lyrics_btn.click(
            fn=export_lyrics_with_timing,
            inputs=[edited_words_state],
            outputs=[lyrics_file]
        )

        def import_lyrics_with_timing(lyrics_json_file):
            """Import lyrics with IPA accents and timing from acentos program"""
            import json
            if not lyrics_json_file:
                gr.Warning("Please upload a lyrics file first.")
                return []

            try:
                with open(lyrics_json_file.name, "r", encoding="utf-8") as f:
                    data = json.load(f)

                words = data.get("words", [])
                return words
            except Exception as e:
                gr.Warning(f"Error importing lyrics: {e}")
                return []

        import_lyrics_btn.click(
            fn=import_lyrics_with_timing,
            inputs=[lyrics_file],
            outputs=[edited_words_state]
        )

        return demo

# -------------------------- Main --------------------------

if __name__ == "__main__":
    demo = create_app()
    demo.launch()
