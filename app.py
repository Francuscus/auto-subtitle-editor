# Language Learning Subtitle Editor
# Version 1.1 - Simplified and Fast
# Banner Color: #2E7D32 (Green)

import os
import re
import json
import tempfile
from typing import List, Tuple
from html import escape as html_escape

import gradio as gr
import torch
import whisperx


# -------------------------- Config --------------------------

VERSION = "1.1"
BANNER_COLOR = "#2E7D32"  # Green for v1.1

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
    """Convert seconds to SRT timestamp format"""
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
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"
    
    print(f"[v{VERSION}] Loading WhisperX on {device} with {compute}")
    
    try:
        _asr_model = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        if "compute type" in str(e).lower():
            fallback = "int16" if device == "cpu" else "float32"
            print(f"Fallback to {fallback}")
            _asr_model = whisperx.load_model("small", device=device, compute_type=fallback)
        else:
            raise
    
    return _asr_model


# -------------------------- Transcription --------------------------

def transcribe_with_words(audio_path: str, language_code: str | None) -> Tuple[List[dict], float]:
    """Transcribe and return word-level timestamps"""
    model = get_asr_model()
    
    print("Transcribing...")
    result = model.transcribe(audio_path, language=language_code)
    segments = result["segments"]
    
    # Get duration
    duration = 0.0
    if segments:
        duration = max(duration, float(segments[-1]["end"]))
    
    # Extract words
    word_segments = []
    for seg in segments:
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                word_segments.append({
                    "start": float(w.get("start", seg["start"])),
                    "end": float(w.get("end", seg["end"])),
                    "text": w.get("word", w.get("text", "")).strip(),
                })
        else:
            # Fallback: split by spaces
            text = seg["text"].strip()
            words = text.split()
            if words:
                seg_start = float(seg["start"])
                seg_end = float(seg["end"])
                seg_dur = seg_end - seg_start
                word_dur = seg_dur / len(words)
                
                for i, word in enumerate(words):
                    w_start = seg_start + (i * word_dur)
                    w_end = w_start + word_dur
                    word_segments.append({
                        "start": round(w_start, 3),
                        "end": round(w_end, 3),
                        "text": word
                    })
    
    print(f"Got {len(word_segments)} words")
    return word_segments, duration


# -------------------------- Grouping --------------------------

def group_words_into_subtitles(word_segments: List[dict], words_per_line: int = 5) -> List[dict]:
    """Group words into subtitle lines"""
    print(f"Grouping {len(word_segments)} words...")
    
    if not word_segments:
        return []
    
    subtitles = []
    i = 0
    
    while i < len(word_segments):
        chunk = word_segments[i:i + words_per_line]
        if not chunk:
            break
        
        sub_start = chunk[0]["start"]
        sub_end = chunk[-1]["end"]
        sub_text = " ".join(w["text"] for w in chunk)
        
        # Initialize colors (white by default)
        colors = ["#FFFFFF"] * len(chunk)
        
        subtitles.append({
            "start": round(sub_start, 3),
            "end": round(sub_end, 3),
            "text": sub_text,
            "words": chunk,
            "colors": colors
        })
        
        i += words_per_line
    
    print(f"Created {len(subtitles)} subtitle lines")
    return subtitles


# -------------------------- Export --------------------------

def export_to_srt(subtitles: List[dict]) -> str:
    """Export to SRT format"""
    lines = []
    for i, sub in enumerate(subtitles, 1):
        start = seconds_to_timestamp(sub["start"])
        end = seconds_to_timestamp(sub["end"])
        text = sub["text"]
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def export_to_ass_with_colors(subtitles: List[dict], video_w: int = 1280, video_h: int = 720,
                               font_size: int = 36, font_name: str = "Arial") -> str:
    """Export to ASS with word colors"""
    
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_w}
PlayResY: {video_h}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,30,30,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    events = []
    for sub in subtitles:
        start = seconds_to_timestamp(sub["start"]).replace(",", ".")
        end = seconds_to_timestamp(sub["end"]).replace(",", ".")
        
        # Build colored text
        text_parts = []
        words = sub.get("words", [])
        colors = sub.get("colors", [])
        
        for i, word_info in enumerate(words):
            word_text = word_info["text"]
            color = colors[i] if i < len(colors) else "#FFFFFF"
            
            # Convert hex to ASS BGR format
            if color.startswith("#"):
                color = color[1:]
            
            r = int(color[0:2], 16) if len(color) >= 2 else 255
            g = int(color[2:4], 16) if len(color) >= 4 else 255
            b = int(color[4:6], 16) if len(color) >= 6 else 255
            
            ass_color = f"&H00{b:02X}{g:02X}{r:02X}"
            text_parts.append(f"{{\\c{ass_color}}}{word_text}")
        
        colored_text = " ".join(text_parts)
        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{colored_text}")
    
    return header + "\n".join(events) + "\n"


def save_file(content: str, extension: str) -> str:
    """Save to temp file"""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, f"subtitles{extension}")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path


# -------------------------- Simple Subtitle Editor HTML --------------------------

def create_simple_editor(subtitles: List[dict]) -> str:
    """Create a simple, lightweight subtitle editor"""
    
    # Limit display for performance
    if len(subtitles) > 50:
        display_subs = subtitles[:50]
        warning = f"<p style='color: orange; font-size: 14px;'>Showing first 50 of {len(subtitles)} subtitles for speed. Full export will include all.</p>"
    else:
        display_subs = subtitles
        warning = ""
    
    # Build simple HTML table
    rows_html = []
    for i, sub in enumerate(display_subs):
        time_str = f"{sub['start']:.2f} - {sub['end']:.2f}s"
        
        # Word chips with colors
        word_chips = []
        for j, word in enumerate(sub.get('words', [])):
            color = sub.get('colors', [])[j] if j < len(sub.get('colors', [])) else "#FFFFFF"
            word_chips.append(
                f'<span style="background: {color}; padding: 4px 8px; margin: 2px; '
                f'border-radius: 4px; display: inline-block; color: #000;">{word["text"]}</span>'
            )
        
        words_html = " ".join(word_chips)
        
        rows_html.append(f"""
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ddd; width: 150px;">{time_str}</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">{sub['text']}</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">{words_html}</td>
            </tr>
        """)
    
    html = f"""
    <div style="background: #f5f5f5; padding: 20px; border-radius: 8px;">
        {warning}
        <table style="width: 100%; border-collapse: collapse; background: white;">
            <thead>
                <tr style="background: {BANNER_COLOR}; color: white;">
                    <th style="padding: 12px; text-align: left;">Time</th>
                    <th style="padding: 12px; text-align: left;">Text</th>
                    <th style="padding: 12px; text-align: left;">Words (Colors)</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows_html)}
            </tbody>
        </table>
        <p style="margin-top: 20px; font-size: 14px; color: #666;">
            To change word colors: Edit the JSON data below, change the "colors" array values (hex codes like #FF0000), then click "Update Preview".
        </p>
    </div>
    """
    
    return html


# -------------------------- Gradio App --------------------------

def create_app():
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Language Learning Subtitle Editor",
        css=f"""
        .gradio-container {{max-width: 1400px !important;}}
        .version-banner {{background: {BANNER_COLOR}; color: white; padding: 20px; 
                        text-align: center; border-radius: 8px; margin-bottom: 20px;}}
        """
    ) as app:
        
        gr.HTML(f"""
        <div class="version-banner">
            <h1 style="margin: 0;">Language Learning Subtitle Editor</h1>
            <p style="margin: 5px 0 0 0;">Version {VERSION} - Perfect for Spanish and Hungarian</p>
        </div>
        """)
        
        # State
        subtitles_state = gr.State([])
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Step 1: Upload and Transcribe")
                
                audio_input = gr.Audio(label="Upload Audio or Video", type="filepath")
                
                language_dropdown = gr.Dropdown(
                    choices=[
                        ("Auto-detect", "auto"),
                        ("Spanish", "es"),
                        ("Hungarian", "hu"),
                        ("English", "en")
                    ],
                    value="auto",
                    label="Language"
                )
                
                words_per_line = gr.Slider(
                    minimum=2,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Words per subtitle line"
                )
                
                transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")
                
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready...",
                    interactive=False,
                    lines=2
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 2: View and Edit")
                
                editor_html = gr.HTML(
                    value="<p style='text-align: center; color: #888;'>Upload audio and click Transcribe to begin</p>"
                )
                
                subtitle_json = gr.JSON(label="Subtitle Data (edit colors here, then click Update Preview)")
                
                update_preview_btn = gr.Button("Update Preview from JSON")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 3: Export")
                
                gr.Markdown("""
                **SRT** = Standard subtitles (works everywhere)  
                **ASS** = Advanced subtitles with word colors (use in VLC, video editors)
                """)
                
                with gr.Row():
                    export_srt_btn = gr.Button("Export SRT")
                    export_ass_btn = gr.Button("Export ASS (with colors)")
                
                srt_file = gr.File(label="SRT File")
                ass_file = gr.File(label="ASS File")
        
        # Events
        
        def do_transcribe(audio_path, language, words_per):
            if not audio_path:
                return "Error: No audio file uploaded", [], "<p>No audio</p>"
            
            try:
                yield "Loading AI model...", [], "<p>Loading...</p>"
                
                lang_code = normalize_lang(language)
                
                yield "Transcribing audio...", [], "<p>Transcribing...</p>"
                
                word_segments, duration = transcribe_with_words(audio_path, lang_code)
                
                yield "Grouping words into subtitles...", [], "<p>Grouping...</p>"
                
                subtitles = group_words_into_subtitles(word_segments, int(words_per))
                
                editor = create_simple_editor(subtitles)
                
                success = f"Done! Created {len(subtitles)} subtitle lines from {len(word_segments)} words."
                
                return success, subtitles, editor
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                return error_msg, [], f"<p style='color: red;'>{error_msg}</p>"
        
        def update_preview(subtitles):
            if not subtitles:
                return "<p>No subtitles to preview</p>"
            editor = create_simple_editor(subtitles)
            return editor
        
        def do_export_srt(subtitles):
            if not subtitles:
                gr.Warning("No subtitles. Transcribe first.")
                return None
            srt_content = export_to_srt(subtitles)
            return save_file(srt_content, ".srt")
        
        def do_export_ass(subtitles):
            if not subtitles:
                gr.Warning("No subtitles. Transcribe first.")
                return None
            ass_content = export_to_ass_with_colors(subtitles)
            return save_file(ass_content, ".ass")
        
        # Connect
        transcribe_btn.click(
            fn=do_transcribe,
            inputs=[audio_input, language_dropdown, words_per_line],
            outputs=[status_text, subtitles_state, editor_html]
        )
        
        update_preview_btn.click(
            fn=update_preview,
            inputs=[subtitle_json],
            outputs=[editor_html]
        )
        
        export_srt_btn.click(
            fn=do_export_srt,
            inputs=[subtitles_state],
            outputs=[srt_file]
        )
        
        export_ass_btn.click(
            fn=do_export_ass,
            inputs=[subtitles_state],
            outputs=[ass_file]
        )
        
        # Sync JSON with state
        subtitles_state.change(
            fn=lambda x: x,
            inputs=[subtitles_state],
            outputs=[subtitle_json]
        )
    
    return app


# -------------------------- Launch --------------------------

if __name__ == "__main__":
    app = create_app()
    app.launch()
