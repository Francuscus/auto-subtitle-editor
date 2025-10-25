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

VERSION = "1.3"
BANNER_COLOR = "#9C27B0"  # Purple for v1.3

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

def create_interactive_editor(subtitles: List[dict]) -> str:
    """Create a full word-processor style editor with text selection, formatting, and editing"""
    
    if not subtitles:
        return "<p>No subtitles to edit</p>"
    
    # Limit for performance
    if len(subtitles) > 30:
        display_subs = subtitles[:30]
        warning = f"<p style='color: orange;'>Showing first 30 of {len(subtitles)} lines for editing. All lines will be exported.</p>"
    else:
        display_subs = subtitles
        warning = ""
    
    # Convert subtitles to JSON for JavaScript
    subs_json = json.dumps(display_subs, ensure_ascii=False).replace("'", "\\'")
    
    html = f"""
    <div style="background: #f5f5f5; padding: 20px; border-radius: 8px; font-family: Arial;">
        {warning}
        <h3>Edit your subtitles like Microsoft Word:</h3>
        
        <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <h4 style="margin-top: 0;">Keyboard Shortcuts:</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div><strong>Ctrl+B</strong> - Bold</div>
                <div><strong>Ctrl+I</strong> - Italic</div>
                <div><strong>Ctrl+U</strong> - Underline</div>
                <div><strong>Shift+F</strong> - Font menu</div>
                <div><strong>Shift+S</strong> - Size menu</div>
                <div><strong>Select text</strong> - Color picker appears</div>
            </div>
            <p style="margin: 10px 0 0 0; font-size: 13px; color: #666;">
                You can also click and type to fix transcription errors!
            </p>
        </div>
        
        <div id="subtitle-editor" style="background: white; padding: 20px; border-radius: 8px; min-height: 300px;">
"""
    
    # Build editable text for each subtitle line
    for line_idx, sub in enumerate(display_subs):
        time_str = f"{sub['start']:.1f}s - {sub['end']:.1f}s"
        
        html += f"""
        <div style="margin-bottom: 25px; padding: 15px; background: #fafafa; border-radius: 8px; border-left: 4px solid {BANNER_COLOR};">
            <div style="color: #666; font-size: 12px; margin-bottom: 10px;">Line {line_idx + 1} | {time_str}</div>
            <div id="line_{line_idx}" 
                 class="editable-line" 
                 contenteditable="true"
                 data-line="{line_idx}"
                 style="font-size: 20px; line-height: 1.6; cursor: text; user-select: text; 
                        padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: white;
                        outline: none;">
"""
        
        # Add each word as a styled span
        words = sub.get('words', [])
        colors = sub.get('colors', [])
        
        for word_idx, word in enumerate(words):
            word_text = word['text']
            current_color = colors[word_idx] if word_idx < len(colors) else "#000000"
            
            html += f"""<span id="word_{line_idx}_{word_idx}" 
                            data-line="{line_idx}" 
                            data-word="{word_idx}"
                            style="color: {current_color};">{word_text}</span> """
        
        html += """
            </div>
        </div>
"""
    
    html += """
        </div>
        
        <!-- Context Menu (appears on text selection) -->
        <div id="context-menu" style="
            display: none;
            position: fixed;
            background: white;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 10000;
            min-width: 280px;">
            
            <div style="margin-bottom: 15px;">
                <strong>Color:</strong><br>
                <input type="color" id="color-picker" value="#000000" 
                       style="width: 100%; height: 40px; margin-top: 5px; cursor: pointer;">
            </div>
            
            <div style="margin-bottom: 15px;">
                <strong>Font (Shift+F):</strong><br>
                <select id="font-picker" style="width: 100%; padding: 8px; margin-top: 5px; font-size: 14px;">
                    <option value="Arial">Arial</option>
                    <option value="Times New Roman">Times New Roman</option>
                    <option value="Courier New">Courier New</option>
                    <option value="Georgia">Georgia</option>
                    <option value="Verdana">Verdana</option>
                    <option value="Comic Sans MS">Comic Sans MS</option>
                    <option value="Impact">Impact</option>
                    <option value="Trebuchet MS">Trebuchet MS</option>
                </select>
            </div>
            
            <div style="margin-bottom: 15px;">
                <strong>Size (Shift+S):</strong><br>
                <select id="size-picker" style="width: 100%; padding: 8px; margin-top: 5px; font-size: 14px;">
                    <option value="12">12px</option>
                    <option value="14">14px</option>
                    <option value="16">16px</option>
                    <option value="18">18px</option>
                    <option value="20" selected>20px</option>
                    <option value="24">24px</option>
                    <option value="28">28px</option>
                    <option value="32">32px</option>
                    <option value="36">36px</option>
                    <option value="42">42px</option>
                    <option value="48">48px</option>
                    <option value="56">56px</option>
                    <option value="64">64px</option>
                    <option value="72">72px</option>
                </select>
            </div>
            
            <div style="margin-bottom: 15px;">
                <strong>Style:</strong><br>
                <div style="display: flex; gap: 5px; margin-top: 5px;">
                    <button id="bold-btn" onclick="toggleBold()" 
                            style="flex: 1; padding: 8px; background: #f0f0f0; border: 1px solid #ccc; 
                                   border-radius: 4px; cursor: pointer; font-weight: bold;">B</button>
                    <button id="italic-btn" onclick="toggleItalic()" 
                            style="flex: 1; padding: 8px; background: #f0f0f0; border: 1px solid #ccc; 
                                   border-radius: 4px; cursor: pointer; font-style: italic;">I</button>
                    <button id="underline-btn" onclick="toggleUnderline()" 
                            style="flex: 1; padding: 8px; background: #f0f0f0; border: 1px solid #ccc; 
                                   border-radius: 4px; cursor: pointer; text-decoration: underline;">U</button>
                </div>
            </div>
            
            <button onclick="applyStyles()" 
                    style="width: 100%; padding: 12px; background: {BANNER_COLOR}; color: white; 
                           border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 15px;">
                Apply Changes
            </button>
            
            <button onclick="closeContextMenu()" 
                    style="width: 100%; padding: 10px; background: #ccc; color: #333; 
                           border: none; border-radius: 4px; cursor: pointer; margin-top: 8px;">
                Cancel (Esc)
            </button>
        </div>
        
        <!-- Quick Color Presets -->
        <div style="margin-top: 20px; padding: 15px; background: #fff3e0; border-radius: 8px;">
            <h4 style="margin-top: 0;">Quick Colors (select text first, then click):</h4>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <button onclick="quickColor('#FF6B6B')" style="padding: 10px 16px; background: #FF6B6B; border: none; border-radius: 4px; cursor: pointer; color: white; font-weight: bold;">Feminine/Red</button>
                <button onclick="quickColor('#4ECDC4')" style="padding: 10px 16px; background: #4ECDC4; border: none; border-radius: 4px; cursor: pointer; color: white; font-weight: bold;">Masculine/Cyan</button>
                <button onclick="quickColor('#FFD93D')" style="padding: 10px 16px; background: #FFD93D; border: none; border-radius: 4px; cursor: pointer; color: black; font-weight: bold;">Verb/Yellow</button>
                <button onclick="quickColor('#95E1D3')" style="padding: 10px 16px; background: #95E1D3; border: none; border-radius: 4px; cursor: pointer; color: black; font-weight: bold;">Adjective/Green</button>
                <button onclick="quickColor('#F38181')" style="padding: 10px 16px; background: #F38181; border: none; border-radius: 4px; cursor: pointer; color: white; font-weight: bold;">Important/Pink</button>
                <button onclick="quickColor('#AA96DA')" style="padding: 10px 16px; background: #AA96DA; border: none; border-radius: 4px; cursor: pointer; color: white; font-weight: bold;">Conjugation/Purple</button>
                <button onclick="quickColor('#000000')" style="padding: 10px 16px; background: #000000; border: none; border-radius: 4px; cursor: pointer; color: white; font-weight: bold;">Black</button>
                <button onclick="quickColor('#FFFFFF')" style="padding: 10px 16px; background: #FFFFFF; border: 2px solid #333; border-radius: 4px; cursor: pointer; color: black; font-weight: bold;">White</button>
            </div>
        </div>
        
        <script>
            let selectedRange = null;
            let subtitleData = {subs_json};
            let currentStyles = {{
                bold: false,
                italic: false,
                underline: false
            }};
            
            // Show context menu on text selection
            document.addEventListener('mouseup', function(e) {{
                const selection = window.getSelection();
                const selectedText = selection.toString().trim();
                
                if (selectedText.length > 0) {{
                    selectedRange = selection.getRangeAt(0);
                    
                    // Position context menu near selection
                    const menu = document.getElementById('context-menu');
                    menu.style.display = 'block';
                    menu.style.left = e.pageX + 'px';
                    menu.style.top = (e.pageY + 10) + 'px';
                    
                    // Get current color from selection
                    const container = selectedRange.commonAncestorContainer;
                    const element = container.nodeType === 3 ? container.parentElement : container;
                    const currentColor = window.getComputedStyle(element).color;
                    
                    // Convert RGB to hex (simplified)
                    document.getElementById('color-picker').value = rgbToHex(currentColor);
                }} else {{
                    // Hide menu if no selection
                    document.getElementById('context-menu').style.display = 'none';
                }}
            }});
            
            // Keyboard shortcuts
            document.addEventListener('keydown', function(e) {{
                const selection = window.getSelection();
                const hasSelection = selection.toString().trim().length > 0;
                
                // Ctrl+B - Bold
                if (e.ctrlKey && e.key === 'b' && hasSelection) {{
                    e.preventDefault();
                    document.execCommand('bold');
                    currentStyles.bold = !currentStyles.bold;
                }}
                
                // Ctrl+I - Italic
                if (e.ctrlKey && e.key === 'i' && hasSelection) {{
                    e.preventDefault();
                    document.execCommand('italic');
                    currentStyles.italic = !currentStyles.italic;
                }}
                
                // Ctrl+U - Underline
                if (e.ctrlKey && e.key === 'u' && hasSelection) {{
                    e.preventDefault();
                    document.execCommand('underline');
                    currentStyles.underline = !currentStyles.underline;
                }}
                
                // Shift+F - Show font picker
                if (e.shiftKey && e.key === 'F' && hasSelection) {{
                    e.preventDefault();
                    document.getElementById('font-picker').focus();
                }}
                
                // Shift+S - Show size picker
                if (e.shiftKey && e.key === 'S' && hasSelection) {{
                    e.preventDefault();
                    document.getElementById('size-picker').focus();
                }}
                
                // Esc - Close context menu
                if (e.key === 'Escape') {{
                    closeContextMenu();
                }}
            }});
            
            // Close context menu when clicking outside
            document.addEventListener('click', function(e) {{
                const menu = document.getElementById('context-menu');
                if (!menu.contains(e.target) && e.target !== menu) {{
                    const selection = window.getSelection();
                    if (selection.toString().trim().length === 0) {{
                        closeContextMenu();
                    }}
                }}
            }});
            
            function closeContextMenu() {{
                document.getElementById('context-menu').style.display = 'none';
                selectedRange = null;
            }}
            
            function toggleBold() {{
                currentStyles.bold = !currentStyles.bold;
                document.getElementById('bold-btn').style.background = 
                    currentStyles.bold ? '{BANNER_COLOR}' : '#f0f0f0';
                document.getElementById('bold-btn').style.color = 
                    currentStyles.bold ? 'white' : 'black';
            }}
            
            function toggleItalic() {{
                currentStyles.italic = !currentStyles.italic;
                document.getElementById('italic-btn').style.background = 
                    currentStyles.italic ? '{BANNER_COLOR}' : '#f0f0f0';
                document.getElementById('italic-btn').style.color = 
                    currentStyles.italic ? 'white' : 'black';
            }}
            
            function toggleUnderline() {{
                currentStyles.underline = !currentStyles.underline;
                document.getElementById('underline-btn').style.background = 
                    currentStyles.underline ? '{BANNER_COLOR}' : '#f0f0f0';
                document.getElementById('underline-btn').style.color = 
                    currentStyles.underline ? 'white' : 'black';
            }}
            
            function applyStyles() {{
                if (!selectedRange) return;
                
                const selection = window.getSelection();
                selection.removeAllRanges();
                selection.addRange(selectedRange);
                
                // Apply color
                const color = document.getElementById('color-picker').value;
                document.execCommand('foreColor', false, color);
                
                // Apply font
                const font = document.getElementById('font-picker').value;
                document.execCommand('fontName', false, font);
                
                // Apply size
                const size = document.getElementById('size-picker').value;
                const container = selectedRange.commonAncestorContainer;
                const element = container.nodeType === 3 ? container.parentElement : container;
                element.style.fontSize = size;
                
                // Apply bold/italic/underline
                if (currentStyles.bold) document.execCommand('bold');
                if (currentStyles.italic) document.execCommand('italic');
                if (currentStyles.underline) document.execCommand('underline');
                
                closeContextMenu();
                
                // Save changes
                saveAllChanges();
            }}
            
            function quickColor(color) {{
                const selection = window.getSelection();
                if (selection.toString().trim().length === 0) {{
                    alert('Please select some text first!');
                    return;
                }}
                
                document.execCommand('foreColor', false, color);
                saveAllChanges();
            }}
            
            function rgbToHex(rgb) {{
                if (rgb.startsWith('#')) return rgb;
                const match = rgb.match(/\\d+/g);
                if (!match) return '#000000';
                const r = parseInt(match[0]);
                const g = parseInt(match[1]);
                const b = parseInt(match[2]);
                return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
            }}
            
            function saveAllChanges() {{
                // Update subtitle data with edited content
                document.querySelectorAll('.editable-line').forEach(line => {{
                    const lineIdx = parseInt(line.dataset.line);
                    if (lineIdx < subtitleData.length) {{
                        subtitleData[lineIdx].text = line.innerText.trim();
                        
                        // Extract word colors
                        const words = Array.from(line.querySelectorAll('span[data-word]'));
                        subtitleData[lineIdx].colors = words.map(w => {{
                            return window.getComputedStyle(w).color;
                        }});
                    }}
                }});
                
                // Send update back to Python
                window.parent.postMessage({{
                    type: 'subtitle_update',
                    data: subtitleData
                }}, '*');
                
                console.log('Changes saved!');
            }}
            
            // Auto-save on blur
            document.querySelectorAll('.editable-line').forEach(line => {{
                line.addEventListener('blur', saveAllChanges);
            }});
        </script>
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
        font_family_state = gr.State("Arial")
        font_size_state = gr.State(36)
        
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
                
                font_family = gr.Dropdown(
                    choices=["Arial", "Times New Roman", "Courier New", "Georgia", "Verdana", "Comic Sans MS"],
                    value="Arial",
                    label="Font Family"
                )
                
                font_size = gr.Slider(
                    minimum=20,
                    maximum=72,
                    value=36,
                    step=2,
                    label="Font Size"
                )
                
                transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")
                
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready...",
                    interactive=False,
                    lines=3,
                    max_lines=5
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 2: View and Edit")
                
                editor_html = gr.HTML(
                    value="<p style='text-align: center; color: #888;'>Upload audio and click Transcribe to begin</p>"
                )
                
                gr.Markdown("""
                **How to edit colors:**
                1. Each word has a color picker next to it - click to change color
                2. OR click on words to select them (they highlight yellow)
                3. Then click a preset color button to apply to all selected words
                4. Colors save automatically when you export!
                """)
        
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
        
        def do_transcribe(audio_path, language, words_per, font_fam, font_sz):
            if not audio_path:
                return "Error: No audio file uploaded", [], "<p>No audio</p>", font_fam, font_sz
            
            try:
                yield "Loading AI model...", [], "<p>Loading...</p>", font_fam, font_sz
                print("[DEBUG] Loading model...")
                
                lang_code = normalize_lang(language)
                
                yield "Transcribing audio...", [], "<p>Transcribing...</p>", font_fam, font_sz
                print("[DEBUG] Starting transcription...")
                
                word_segments, duration = transcribe_with_words(audio_path, lang_code)
                print(f"[DEBUG] Got {len(word_segments)} words")
                
                yield f"Got {len(word_segments)} words! Now grouping...", [], "<p>Grouping...</p>", font_fam, font_sz
                print("[DEBUG] Starting grouping...")
                
                subtitles = group_words_into_subtitles(word_segments, int(words_per))
                print(f"[DEBUG] Grouping done, got {len(subtitles)} subtitles")
                
                # Show progress for each subtitle line created
                yield f"Created {len(subtitles)} subtitle lines! Building preview...", [], f"<p>Building preview with {len(subtitles)} lines...</p>", font_fam, font_sz
                print("[DEBUG] Creating HTML preview...")
                
                # Show the subtitles as we build the preview
                preview_lines = []
                for i, sub in enumerate(subtitles[:10]):  # Show first 10 as preview
                    preview_lines.append(f"{i+1}. [{sub['start']:.1f}s] {sub['text']}")
                
                preview_text = "<br>".join(preview_lines)
                if len(subtitles) > 10:
                    preview_text += f"<br>... and {len(subtitles) - 10} more lines"
                
                yield f"Preview ready! Total: {len(subtitles)} lines", [], f"<div style='background:#f0f0f0; padding:10px;'>{preview_text}</div>", font_fam, font_sz
                
                editor = create_interactive_editor(subtitles)
                print("[DEBUG] HTML preview created")
                
                success = f"Done! Created {len(subtitles)} subtitle lines from {len(word_segments)} words."
                
                return success, subtitles, editor, font_fam, font_sz
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(f"[DEBUG ERROR] {error_msg}")
                import traceback
                traceback.print_exc()
                return error_msg, [], f"<p style='color: red;'>{error_msg}</p>", font_fam, font_sz
        
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
            ass_content = export_to_ass_with_colors(subtitles, font_name=font_fam, font_size=int(font_sz))
            return save_file(ass_content, ".ass")
        
        # Connect
        transcribe_btn.click(
            fn=do_transcribe,
            inputs=[audio_input, language_dropdown, words_per_line, font_family, font_size],
            outputs=[status_text, subtitles_state, editor_html, font_family_state, font_size_state]
        )
        
        export_srt_btn.click(
            fn=do_export_srt,
            inputs=[subtitles_state],
            outputs=[srt_file]
        )
        
        export_ass_btn.click(
            fn=do_export_ass,
            inputs=[subtitles_state, font_family_state, font_size_state],
            outputs=[ass_file]
        )
    
    return app


# -------------------------- Launch --------------------------

if __name__ == "__main__":
    app = create_app()
    app.launch()
