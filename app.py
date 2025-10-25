# Enhanced Colorvideo Subs with Timeline Editor
# Version 1.0 - Language Learning Focus

import os
import re
import json
import tempfile
import subprocess
from typing import List, Tuple, Dict
from html import escape as html_escape

import gradio as gr
import torch
import whisperx


# -------------------------- Utilities --------------------------

LANG_MAP = {
    "auto": None, "auto-detect": None, "automatic": None,
    "hungarian": "hu", "magyar": "hu", "hun": "hu",
    "spanish": "es", "español": "es", "esp": "es",
    "english": "en", "eng": "en",
}


def normalize_lang(s: str | None):
    if not s:
        return None
    t = s.strip().lower()
    if t in LANG_MAP:
        return LANG_MAP[t]
    m = re.search(r"\b([a-z]{2,3})\b", t)
    return m.group(1) if m else None


def seconds_to_timestamp(t: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    t = max(t, 0.0)
    h = int(t // 3600)
    t -= h * 3600
    m = int(t // 60)
    t -= m * 60
    s = int(t)
    ms = int(round((t - s) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def timestamp_to_seconds(ts: str) -> float:
    """Convert SRT timestamp to seconds"""
    try:
        time_part = ts.replace(',', '.')
        parts = time_part.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
    except:
        pass
    return 0.0


# -------------------------- ASR Model (lazy loading) --------------------------

_asr_model = None


def get_asr_model():
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"
    
    print(f"Loading WhisperX model on {device} with {compute}...")
    
    try:
        _asr_model = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        if "compute type" in str(e).lower():
            fallback = "int16" if device == "cpu" else "float32"
            print(f"Falling back to {fallback}")
            _asr_model = whisperx.load_model("small", device=device, compute_type=fallback)
        else:
            raise
    
    return _asr_model


# -------------------------- Transcription with Word Timestamps --------------------------

def transcribe_with_words(audio_path: str, language_code: str | None) -> Tuple[List[dict], float]:
    """
    Transcribe audio and return word-level timestamps.
    Returns: (word_segments, total_duration)
    word_segments format: [{"start": 0.0, "end": 1.5, "text": "Hello", "word": "Hello"}, ...]
    """
    model = get_asr_model()
    
    # Transcribe
    print("Transcribing audio...")
    result = model.transcribe(audio_path, language=language_code)
    segments = result["segments"]
    detected_language = result.get("language", language_code or "en")
    
    # Get duration
    duration = 0.0
    if segments:
        duration = max(duration, float(segments[-1]["end"]))
    
    # Extract words with timestamps
    word_segments = []
    for seg in segments:
        # WhisperX might have word-level timestamps in some cases
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                word_segments.append({
                    "start": float(w.get("start", seg["start"])),
                    "end": float(w.get("end", seg["end"])),
                    "text": w.get("word", w.get("text", "")).strip(),
                    "word": w.get("word", w.get("text", "")).strip()
                })
        else:
            # Fallback: split segment into words with estimated timing
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
                        "text": word,
                        "word": word
                    })
    
    print(f"Transcribed {len(word_segments)} words in {detected_language}")
    return word_segments, duration


# -------------------------- Subtitle Grouping --------------------------

def group_words_into_subtitles(word_segments: List[dict], words_per_line: int = 5) -> List[dict]:
    """
    Group words into subtitle lines.
    Returns: [{"start": 0.0, "end": 2.5, "text": "Hello world", "words": [...], "colors": []}, ...]
    """
    if not word_segments:
        return []
    
    subtitles = []
    i = 0
    
    while i < len(word_segments):
        # Take next N words
        chunk = word_segments[i:i + words_per_line]
        if not chunk:
            break
        
        sub_start = chunk[0]["start"]
        sub_end = chunk[-1]["end"]
        sub_text = " ".join(w["text"] for w in chunk)
        
        # Initialize colors (default: white for all words)
        colors = ["#FFFFFF"] * len(chunk)
        
        subtitles.append({
            "start": round(sub_start, 3),
            "end": round(sub_end, 3),
            "text": sub_text,
            "words": chunk,
            "colors": colors  # One color per word
        })
        
        i += words_per_line
    
    return subtitles


# -------------------------- Export Functions --------------------------

def export_to_srt(subtitles: List[dict]) -> str:
    """Export subtitles to SRT format"""
    lines = []
    for i, sub in enumerate(subtitles, 1):
        start = seconds_to_timestamp(sub["start"])
        end = seconds_to_timestamp(sub["end"])
        text = sub["text"]
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def export_to_ass_with_colors(subtitles: List[dict], video_w: int = 1280, video_h: int = 720,
                               font_size: int = 36, font_name: str = "Arial") -> str:
    """
    Export to ASS format with word-level colors.
    Uses inline color tags for each word.
    """
    
    # ASS Header
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
        
        # Build text with color tags
        text_parts = []
        words = sub.get("words", [])
        colors = sub.get("colors", [])
        
        for i, word_info in enumerate(words):
            word_text = word_info["text"]
            color = colors[i] if i < len(colors) else "#FFFFFF"
            
            # Convert hex to ASS color format (&HAABBGGRR)
            if color.startswith("#"):
                color = color[1:]
            
            # Hex to RGB
            r = int(color[0:2], 16) if len(color) >= 2 else 255
            g = int(color[2:4], 16) if len(color) >= 4 else 255
            b = int(color[4:6], 16) if len(color) >= 6 else 255
            
            # ASS color format (BGR)
            ass_color = f"&H00{b:02X}{g:02X}{r:02X}"
            
            text_parts.append(f"{{\\c{ass_color}}}{word_text}")
        
        colored_text = " ".join(text_parts)
        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{colored_text}")
    
    return header + "\n".join(events) + "\n"


def save_file(content: str, extension: str) -> str:
    """Save content to a temporary file and return the path"""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, f"subtitles{extension}")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path


# -------------------------- Timeline HTML Component --------------------------

def create_timeline_html(subtitles: List[dict], audio_path: str) -> str:
    """
    Create an interactive timeline with waveform and subtitle editor.
    """
    
    # Convert subtitles to JSON for JavaScript
    subs_json = json.dumps(subtitles, ensure_ascii=False)
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
        }}
        
        #timeline-container {{
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        #waveform {{
            width: 100%;
            height: 120px;
            background: #1a1a1a;
            border-radius: 4px;
            margin-bottom: 10px;
        }}
        
        #controls {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
        }}
        
        button {{
            background: #4a7cff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        
        button:hover {{
            background: #3a6cef;
        }}
        
        #current-time {{
            font-size: 18px;
            font-weight: bold;
            color: #4a7cff;
        }}
        
        #subtitle-list {{
            max-height: 400px;
            overflow-y: auto;
            background: #1a1a1a;
            border-radius: 4px;
            padding: 10px;
        }}
        
        .subtitle-item {{
            background: #2a2a2a;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 4px;
            border-left: 4px solid #4a7cff;
            cursor: pointer;
        }}
        
        .subtitle-item:hover {{
            background: #3a3a3a;
        }}
        
        .subtitle-item.active {{
            border-left-color: #ff4a4a;
            background: #3a2a2a;
        }}
        
        .subtitle-time {{
            color: #888;
            font-size: 12px;
            margin-bottom: 5px;
        }}
        
        .subtitle-text {{
            font-size: 16px;
            margin-bottom: 10px;
        }}
        
        .word-colorizer {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }}
        
        .word-chip {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 5px 10px;
            background: #1a1a1a;
            border-radius: 4px;
            cursor: pointer;
        }}
        
        .word-chip:hover {{
            background: #0a0a0a;
        }}
        
        .color-dot {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 2px solid #fff;
        }}
        
        .color-picker-popup {{
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            z-index: 1000;
        }}
        
        .color-picker-popup.active {{
            display: block;
        }}
        
        .overlay {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
            z-index: 999;
        }}
        
        .overlay.active {{
            display: block;
        }}
        
        .color-grid {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 10px;
            margin: 20px 0;
        }}
        
        .color-option {{
            width: 40px;
            height: 40px;
            border-radius: 4px;
            cursor: pointer;
            border: 2px solid transparent;
        }}
        
        .color-option:hover {{
            border-color: #fff;
        }}
        
        .preset-colors {{
            margin: 20px 0;
        }}
        
        .preset-btn {{
            margin: 5px;
            padding: 8px 15px;
            background: #3a3a3a;
        }}
    </style>
</head>
<body>
    <div id="timeline-container">
        <h3>Audio Timeline</h3>
        <div id="waveform"></div>
        <div id="controls">
            <button id="play-pause">▶ Play</button>
            <button id="stop">⏹ Stop</button>
            <span id="current-time">00:00</span>
            <span>/</span>
            <span id="total-time">00:00</span>
        </div>
    </div>
    
    <div id="subtitle-list"></div>
    
    <div class="overlay" id="overlay"></div>
    <div class="color-picker-popup" id="colorPicker">
        <h3>Choose Word Color</h3>
        <p id="selected-word-display"></p>
        
        <div class="preset-colors">
            <h4>Language Learning Presets:</h4>
            <button class="preset-btn" data-color="#FF6B6B">❤️ Feminine</button>
            <button class="preset-btn" data-color="#4ECDC4">💙 Masculine</button>
            <button class="preset-btn" data-color="#FFD93D">⭐ Verb</button>
            <button class="preset-btn" data-color="#95E1D3">🌿 Adjective</button>
            <button class="preset-btn" data-color="#F38181">🔴 Important</button>
            <button class="preset-btn" data-color="#AA96DA">💜 Conjugation</button>
        </div>
        
        <div class="color-grid" id="colorGrid"></div>
        
        <div>
            <input type="color" id="customColor" value="#FFFFFF">
            <button id="applyCustom">Apply Custom Color</button>
        </div>
        
        <button id="closeColorPicker">Close</button>
    </div>
    
    <script>
        // Subtitle data from Python
        let subtitles = {subs_json};
        let currentSubtitleIndex = -1;
        let currentWordIndex = -1;
        let isPlaying = false;
        let currentTime = 0;
        let duration = 0;
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            renderSubtitles();
            initializeColorPicker();
            updateTimeDisplay();
            
            // Note: Actual audio playback would require proper audio element
            // This is a simplified version for demonstration
            console.log('Timeline initialized with ' + subtitles.length + ' subtitles');
        }});
        
        function renderSubtitles() {{
            const container = document.getElementById('subtitle-list');
            container.innerHTML = '';
            
            subtitles.forEach((sub, index) => {{
                const item = document.createElement('div');
                item.className = 'subtitle-item';
                item.dataset.index = index;
                
                const timeDiv = document.createElement('div');
                timeDiv.className = 'subtitle-time';
                timeDiv.textContent = formatTime(sub.start) + ' → ' + formatTime(sub.end);
                
                const textDiv = document.createElement('div');
                textDiv.className = 'subtitle-text';
                textDiv.textContent = sub.text;
                
                const wordDiv = document.createElement('div');
                wordDiv.className = 'word-colorizer';
                
                // Render words with colors
                if (sub.words && sub.words.length > 0) {{
                    sub.words.forEach((word, wordIndex) => {{
                        const wordChip = document.createElement('div');
                        wordChip.className = 'word-chip';
                        wordChip.dataset.subIndex = index;
                        wordChip.dataset.wordIndex = wordIndex;
                        
                        const colorDot = document.createElement('div');
                        colorDot.className = 'color-dot';
                        colorDot.style.backgroundColor = sub.colors[wordIndex] || '#FFFFFF';
                        
                        const wordText = document.createElement('span');
                        wordText.textContent = word.text;
                        
                        wordChip.appendChild(colorDot);
                        wordChip.appendChild(wordText);
                        
                        wordChip.addEventListener('click', (e) => {{
                            e.stopPropagation();
                            openColorPicker(index, wordIndex, word.text);
                        }});
                        
                        wordDiv.appendChild(wordChip);
                    }});
                }}
                
                item.appendChild(timeDiv);
                item.appendChild(textDiv);
                item.appendChild(wordDiv);
                
                item.addEventListener('click', () => {{
                    seekToSubtitle(index);
                }});
                
                container.appendChild(item);
            }});
        }}
        
        function formatTime(seconds) {{
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return mins.toString().padStart(2, '0') + ':' + secs.toString().padStart(2, '0');
        }}
        
        function seekToSubtitle(index) {{
            currentSubtitleIndex = index;
            currentTime = subtitles[index].start;
            updateTimeDisplay();
            highlightCurrentSubtitle();
        }}
        
        function highlightCurrentSubtitle() {{
            document.querySelectorAll('.subtitle-item').forEach((item, index) => {{
                if (index === currentSubtitleIndex) {{
                    item.classList.add('active');
                    item.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
                }} else {{
                    item.classList.remove('active');
                }}
            }});
        }}
        
        function updateTimeDisplay() {{
            document.getElementById('current-time').textContent = formatTime(currentTime);
        }}
        
        function initializeColorPicker() {{
            const colorGrid = document.getElementById('colorGrid');
            const commonColors = [
                '#FF6B6B', '#4ECDC4', '#FFD93D', '#95E1D3', '#F38181', '#AA96DA',
                '#FFA07A', '#20B2AA', '#FFB6C1', '#87CEEB', '#DDA0DD', '#F0E68C',
                '#FFFFFF', '#CCCCCC', '#999999', '#666666', '#333333', '#000000'
            ];
            
            commonColors.forEach(color => {{
                const option = document.createElement('div');
                option.className = 'color-option';
                option.style.backgroundColor = color;
                option.dataset.color = color;
                option.addEventListener('click', () => applyColor(color));
                colorGrid.appendChild(option);
            }});
            
            // Preset buttons
            document.querySelectorAll('.preset-btn').forEach(btn => {{
                btn.addEventListener('click', () => {{
                    applyColor(btn.dataset.color);
                }});
            }});
            
            document.getElementById('applyCustom').addEventListener('click', () => {{
                const color = document.getElementById('customColor').value;
                applyColor(color);
            }});
            
            document.getElementById('closeColorPicker').addEventListener('click', closeColorPicker);
            document.getElementById('overlay').addEventListener('click', closeColorPicker);
        }}
        
        function openColorPicker(subIndex, wordIndex, wordText) {{
            currentSubtitleIndex = subIndex;
            currentWordIndex = wordIndex;
            
            document.getElementById('selected-word-display').textContent = 
                'Coloring: "' + wordText + '"';
            
            document.getElementById('overlay').classList.add('active');
            document.getElementById('colorPicker').classList.add('active');
        }}
        
        function closeColorPicker() {{
            document.getElementById('overlay').classList.remove('active');
            document.getElementById('colorPicker').classList.remove('active');
        }}
        
        function applyColor(color) {{
            if (currentSubtitleIndex >= 0 && currentWordIndex >= 0) {{
                subtitles[currentSubtitleIndex].colors[currentWordIndex] = color;
                renderSubtitles();
                
                // Send update back to Python (via Gradio custom event)
                window.parent.postMessage({{
                    type: 'subtitle_update',
                    data: subtitles
                }}, '*');
            }}
            closeColorPicker();
        }}
        
        // Playback controls (simplified - would need actual audio element)
        document.getElementById('play-pause').addEventListener('click', () => {{
            // Placeholder for audio playback
            alert('Audio playback would be integrated here. For now, click on subtitles to jump to their time.');
        }});
        
        document.getElementById('stop').addEventListener('click', () => {{
            currentTime = 0;
            currentSubtitleIndex = -1;
            updateTimeDisplay();
            highlightCurrentSubtitle();
        }});
    </script>
</body>
</html>
"""
    
    return html


# -------------------------- Main Gradio Interface --------------------------

def create_app():
    with gr.Blocks(theme=gr.themes.Soft(), title="Language Learning Subtitle Editor") as app:
        gr.Markdown("""
        # 🎓 Language Learning Subtitle Editor
        ### Perfect for Spanish & Hungarian lessons!
        
        **Features:**
        - 🎯 Word-level timing & colorization
        - 📝 Interactive timeline editor
        - 🎨 Color word endings for grammar learning
        - 🌍 Optimized for Spanish & Hungarian
        """)
        
        # State variables
        subtitles_state = gr.State([])
        audio_path_state = gr.State("")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1️⃣ Upload & Transcribe")
                
                audio_input = gr.Audio(
                    label="Upload Audio/Video",
                    type="filepath"
                )
                
                language_dropdown = gr.Dropdown(
                    choices=[
                        ("Auto-detect", "auto"),
                        ("Spanish (Español)", "es"),
                        ("Hungarian (Magyar)", "hu"),
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
                
                transcribe_btn = gr.Button("🎤 Transcribe", variant="primary", size="lg")
                
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready to transcribe...",
                    interactive=False
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 2️⃣ Edit Timeline & Colors")
                
                timeline_html = gr.HTML(
                    value="<p style='text-align:center; color:#888;'>Timeline will appear here after transcription...</p>"
                )
                
                gr.Markdown("""
                **How to use:**
                - Click on any subtitle to jump to that time
                - Click on individual words to change their color
                - Use preset colors for grammar categories (verbs, endings, etc.)
                """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 3️⃣ Export")
                
                with gr.Row():
                    export_srt_btn = gr.Button("📄 Export SRT")
                    export_ass_btn = gr.Button("🎨 Export ASS (with colors)")
                
                srt_file = gr.File(label="SRT File")
                ass_file = gr.File(label="ASS File (colored)")
                
                subtitle_json = gr.JSON(label="Subtitle Data (for debugging)", visible=False)
        
        # Event handlers
        
        def do_transcribe(audio_path, language, words_per):
            if not audio_path:
                return "⚠️ Please upload an audio file first!", [], "", "<p>No audio uploaded</p>"
            
            try:
                yield "🔄 Loading AI model...", [], audio_path, "<p>Loading...</p>"
                
                # Normalize language
                lang_code = normalize_lang(language)
                
                yield "🎤 Transcribing audio...", [], audio_path, "<p>Transcribing...</p>"
                
                # Transcribe with word timestamps
                word_segments, duration = transcribe_with_words(audio_path, lang_code)
                
                yield f"✨ Grouping {len(word_segments)} words...", [], audio_path, "<p>Processing...</p>"
                
                # Group into subtitle lines
                subtitles = group_words_into_subtitles(word_segments, int(words_per))
                
                # Create timeline HTML
                timeline = create_timeline_html(subtitles, audio_path)
                
                success_msg = f"✅ Done! {len(subtitles)} subtitle lines created."
                
                return success_msg, subtitles, audio_path, timeline
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                return error_msg, [], "", f"<p style='color:red;'>{error_msg}</p>"
        
        def do_export_srt(subtitles):
            if not subtitles:
                gr.Warning("No subtitles to export. Please transcribe first.")
                return None
            
            srt_content = export_to_srt(subtitles)
            srt_path = save_file(srt_content, ".srt")
            return srt_path
        
        def do_export_ass(subtitles):
            if not subtitles:
                gr.Warning("No subtitles to export. Please transcribe first.")
                return None
            
            ass_content = export_to_ass_with_colors(subtitles)
            ass_path = save_file(ass_content, ".ass")
            return ass_path
        
        # Connect events
        transcribe_btn.click(
            fn=do_transcribe,
            inputs=[audio_input, language_dropdown, words_per_line],
            outputs=[status_text, subtitles_state, audio_path_state, timeline_html]
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
    
    return app


# -------------------------- Launch --------------------------

if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)
