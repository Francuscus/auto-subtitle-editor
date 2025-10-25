# Language Learning Subtitle Editor
# Version 1.6 - Interactive Dataframe Editor
# Banner Color: #FF6F00 (Deep Orange)

import os
import re
import tempfile
from typing import List, Tuple
import pandas as pd

import gradio as gr
import torch
import whisperx


# -------------------------- Config --------------------------

VERSION = "1.6"
BANNER_COLOR = "#FF6F00"  # Deep Orange for v1.6

LANG_MAP = {
    "auto": None, "auto-detect": None, "automatic": None,
    "hungarian": "hu", "magyar": "hu", "hun": "hu",
    "spanish": "es", "español": "es", "esp": "es",
    "english": "en", "eng": "en",
}

# Preset colors with names
COLOR_PRESETS = {
    "White": "#FFFFFF",
    "Black": "#000000",
    "Feminine/Red": "#FF6B6B",
    "Masculine/Cyan": "#4ECDC4",
    "Verb/Yellow": "#FFD93D",
    "Adjective/Green": "#95E1D3",
    "Important/Pink": "#F38181",
    "Conjugation/Purple": "#AA96DA",
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
    
    duration = 0.0
    if segments:
        duration = max(duration, float(segments[-1]["end"]))
    
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
        
        colors = ["White"] * len(chunk)
        
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


# -------------------------- Create DataFrame for Editing --------------------------

def subtitles_to_dataframe(subtitles: List[dict]) -> pd.DataFrame:
    """Convert subtitles to a DataFrame for editing"""
    rows = []
    
    for line_idx, sub in enumerate(subtitles):
        words = sub.get('words', [])
        colors = sub.get('colors', [])
        
        for word_idx, word in enumerate(words):
            rows.append({
                'Line': line_idx + 1,
                'Word#': word_idx,
                'Word': word['text'],
                'Color': colors[word_idx] if word_idx < len(colors) else "White",
                'Start': round(word['start'], 2),
                'End': round(word['end'], 2)
            })
    
    return pd.DataFrame(rows)


def dataframe_to_subtitles(df: pd.DataFrame, original_subtitles: List[dict]) -> List[dict]:
    """Convert edited DataFrame back to subtitles"""
    # Create a copy of original subtitles
    import copy
    subtitles = copy.deepcopy(original_subtitles)
    
    # Update colors and text from dataframe
    for _, row in df.iterrows():
        line_num = int(row['Line']) - 1
        word_idx = int(row['Word#'])
        
        if line_num < len(subtitles):
            # Update word text
            if word_idx < len(subtitles[line_num]['words']):
                subtitles[line_num]['words'][word_idx]['text'] = str(row['Word'])
            
            # Update color
            if word_idx < len(subtitles[line_num]['colors']):
                subtitles[line_num]['colors'][word_idx] = str(row['Color'])
    
    # Rebuild full text for each line
    for sub in subtitles:
        sub['text'] = " ".join(w['text'] for w in sub['words'])
    
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
                               default_font: str = "Arial", default_size: int = 36) -> str:
    """Export to ASS with word colors"""
    
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_w}
PlayResY: {video_h}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{default_font},{default_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,30,30,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    events = []
    for sub in subtitles:
        start = seconds_to_timestamp(sub["start"]).replace(",", ".")
        end = seconds_to_timestamp(sub["end"]).replace(",", ".")
        
        text_parts = []
        words = sub.get("words", [])
        colors = sub.get("colors", [])
        
        for i, word_info in enumerate(words):
            word_text = word_info["text"]
            color_name = colors[i] if i < len(colors) else "White"
            color_hex = COLOR_PRESETS.get(color_name, "#FFFFFF")
            
            if color_hex.startswith("#"):
                color_hex = color_hex[1:]
            
            r = int(color_hex[0:2], 16)
            g = int(color_hex[2:4], 16)
            b = int(color_hex[4:6], 16)
            
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


# -------------------------- Gradio App --------------------------

def create_app():
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Language Learning Subtitle Editor"
    ) as app:
        
        gr.HTML(f"""
        <div style="background: {BANNER_COLOR}; color: white; padding: 20px; text-align: center; border-radius: 8px; margin-bottom: 20px;">
            <h1 style="margin: 0;">Language Learning Subtitle Editor</h1>
            <p style="margin: 5px 0 0 0;">Version {VERSION} - Edit Like a Spreadsheet!</p>
        </div>
        """)
        
        gr.Markdown("""
        ## How to use:
        1. Upload audio and transcribe
        2. Edit the table below - change words, change colors
        3. Click "Update Subtitles" to save changes
        4. Export SRT or ASS files
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
                
                with gr.Row():
                    font_family = gr.Dropdown(
                        choices=["Arial", "Times New Roman", "Courier New", "Georgia", "Verdana"],
                        value="Arial",
                        label="Font (for export)"
                    )
                    
                    font_size = gr.Slider(
                        minimum=20,
                        maximum=72,
                        value=36,
                        step=2,
                        label="Size (for export)"
                    )
                
                transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")
                
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready...",
                    interactive=False
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 2: Edit Subtitles")
                
                gr.Markdown("""
                **Edit the table:**
                - Click any cell to edit
                - Change Word column to fix transcription errors
                - Change Color column using dropdown (click the cell!)
                - Available colors: White, Black, Feminine/Red, Masculine/Cyan, Verb/Yellow, Adjective/Green, Important/Pink, Conjugation/Purple
                """)
                
                editor_df = gr.Dataframe(
                    headers=["Line", "Word#", "Word", "Color", "Start", "End"],
                    datatype=["number", "number", "str", "str", "number", "number"],
                    col_count=(6, "fixed"),
                    row_count=(10, "dynamic"),
                    interactive=True,
                    wrap=True
                )
                
                update_btn = gr.Button("Update Subtitles from Table", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 3: Export")
                
                with gr.Row():
                    export_srt_btn = gr.Button("Export SRT")
                    export_ass_btn = gr.Button("Export ASS (with colors)")
                
                srt_file = gr.File(label="SRT File")
                ass_file = gr.File(label="ASS File")
        
        # Events
        
        def do_transcribe(audio_path, language, words_per):
            if not audio_path:
                return "Error: No audio file", [], pd.DataFrame()
            
            try:
                yield "Loading AI model...", [], pd.DataFrame()
                
                lang_code = normalize_lang(language)
                
                yield "Transcribing...", [], pd.DataFrame()
                
                word_segments, duration = transcribe_with_words(audio_path, lang_code)
                
                yield f"Grouping {len(word_segments)} words...", [], pd.DataFrame()
                
                subtitles = group_words_into_subtitles(word_segments, int(words_per))
                
                df = subtitles_to_dataframe(subtitles)
                
                success = f"Done! {len(subtitles)} lines, {len(df)} words. Edit the table below!"
                
                return success, subtitles, df
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                return error_msg, [], pd.DataFrame()
        
        def update_from_dataframe(df_data, original_subs):
            if df_data is None or len(df_data) == 0:
                return original_subs, "No data to update"
            
            try:
                df = pd.DataFrame(df_data)
                updated_subs = dataframe_to_subtitles(df, original_subs)
                return updated_subs, "Subtitles updated! You can now export."
            except Exception as e:
                return original_subs, f"Error updating: {str(e)}"
        
        def do_export_srt(subtitles):
            if not subtitles:
                gr.Warning("No subtitles. Transcribe first.")
                return None
            srt_content = export_to_srt(subtitles)
            return save_file(srt_content, ".srt")
        
        def do_export_ass(subtitles, font_fam, font_sz):
            if not subtitles:
                gr.Warning("No subtitles. Transcribe first.")
                return None
            ass_content = export_to_ass_with_colors(subtitles, default_font=font_fam, default_size=int(font_sz))
            return save_file(ass_content, ".ass")
        
        # Connect
        transcribe_btn.click(
            fn=do_transcribe,
            inputs=[audio_input, language_dropdown, words_per_line],
            outputs=[status_text, subtitles_state, editor_df]
        )
        
        update_btn.click(
            fn=update_from_dataframe,
            inputs=[editor_df, subtitles_state],
            outputs=[subtitles_state, status_text]
        )
        
        export_srt_btn.click(
            fn=do_export_srt,
            inputs=[subtitles_state],
            outputs=[srt_file]
        )
        
        export_ass_btn.click(
            fn=do_export_ass,
            inputs=[subtitles_state, font_family, font_size],
            outputs=[ass_file]
        )
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
