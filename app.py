import gradio as gr
import whisperx
import torch
import pysubs2
import ffmpeg
from io import BytesIO
import tempfile
import os

# Load model once (free on HF Spaces GPU/CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
audio_model = whisperx.load_model("base", device, compute_type="float16" if device == "cuda" else "int8")
align_model, metadata = whisperx.load_align_model(language_code="es", device=device)  # Default Spanish; swap for "hu" Hungarian

def extract_audio(video_file):
    """Extract audio from video if uploaded (optional)."""
    if video_file is not None and not str(video_file).endswith('.wav') and not str(video_file).endswith('.mp3'):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            stream = ffmpeg.input(video_file.name)
            stream = ffmpeg.output(stream, tmp.name, acodec='pcm_s16le', ar=16000)
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            return tmp.name
    return video_file.name if video_file else None

def transcribe_and_align(audio_file, language):
    """Transcribe with word-level timestamps."""
    audio = whisperx.load_audio(audio_file)
    result = audio_model.transcribe(audio, language=language)
    aligned = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)
    return aligned["segments"]  # List of dicts with words/timestamps

def generate_ass(subtitle_data, edits):
    """Generate ASS with per-letter styles from edits dict (e.g., {'word_index': {'char_pos': {'font': 'Arial', 'color': '#FF0000'}}} )."""
    ass = pysubs2.SSAFile.from_string('[Script Info]\nTitle: Generated\n[Events]\nFormat: Start, End, Style, Text\n')
    style = pysubs2.SSAStyle(Fontname='Arial', Fontsize=24, PrimaryColour='&H00FFFFFF', OutlineColour='&H00000000')
    ass.styles['Default'] = style
    for seg in subtitle_data:
        start, end = seg['start'], seg['end']
        text = ' '.join([w['word'] for w in seg['words']])
        # Apply edits: For simplicity, wrap letters in tags if edited
        formatted_text = text
        if seg in edits:  # Pseudo-edits; in full app, parse from UI
            for word_idx, word_edits in edits.get(seg, {}).items():
                for char_pos, char_edit in word_edits.items():
                    char = text[char_pos]
                    formatted_text = formatted_text.replace(char, f'{{fn{char_edit.get("font","Arial")}\\c&{char_edit.get("color","#FFFFFF")}&}}{char}', 1)
        event = pysubs2.SSAEvent(start=start*1000, end=end*1000, text=formatted_text, style='Default')
        ass.events.append(event)
    return ass.to_string('ass')

def main_interface(audio_or_video, language, font_edits, color_edits):
    if not audio_or_video:
        return "Upload audio/video.", None, None, None
    
    audio_path = extract_audio(audio_or_video)
    segments = transcribe_and_align(audio_path, language)
    
    # Timeline: Simple HTML with audio player + synced text (use JS for basic waveform)
    timeline_html = "<audio controls src='file://" + audio_path + "'></audio><div id='timeline'>"
    for i, seg in enumerate(segments):
        start = seg['start']
        text = ' '.join([w['word'] for w in seg['words']])
        timeline_html += f"<div style='margin-top:{start*10}px; border-left:2px solid blue; padding:5px;'><strong>{start:.1f}s:</strong> {text}</div>"
    timeline_html += "</div><script>/* Basic sync: Play audio, highlight on time */ console.log('Timeline ready');</script>"
    
    # Edits preview: Basic form for per-letter (expandable; here, demo for first word)
    edits_html = "<h3>Edit Per Letter (Demo for first segment):</h3><form>"
    if segments:
        first_word = segments[0]['words'][0]['word']
        for j, char in enumerate(first_word):
            edits_html += f"Char '{char}': <input type='text' placeholder='Font' id='font_{j}'><input type='color' id='color_{j}' value='#FFFFFF'><br>"
    edits_html += "</form><button onclick='applyEdits()'>Preview</button><div id='preview'></div>"
    
    # Generate sample ASS (with dummy edits)
    dummy_edits = {0: {0: {'font': 'Arial', 'color': '#FF0000'}}}  # Example: First char red
    ass_content = generate_ass(segments, dummy_edits)
    
    if os.path.exists(audio_path) and audio_path != audio_or_video.name:
        os.unlink(audio_path)  # Cleanup
    
    return timeline_html, edits_html, ass_content, f"Transcribed {len(segments)} segments in {language}."

# Gradio UI
with gr.Blocks(title="Auto Subtitle Editor") as demo:
    gr.Markdown("# Free Online Subtitle Editor with WhisperX")
    with gr.Row():
        audio_input = gr.File(label="Upload Audio/Video", file_types=[".mp3", ".wav", ".mp4"])
        lang_dropdown = gr.Dropdown(choices=["es (Spanish)", "hu (Hungarian)"], value="es (Spanish)", label="Language")
    edit_font = gr.Textbox(label="Global Font (or per-letter in preview)")
    edit_color = gr.ColorPicker(label="Global Color")
    transcribe_btn = gr.Button("Transcribe & Generate Timeline")
    timeline_out = gr.HTML(label="Synced Timeline (Waveform + Text)")
    edits_out = gr.HTML(label="Per-Letter Editor Preview")
    ass_out = gr.Textbox(label="Downloadable ASS File (Fonts/Colors Baked In)", lines=10)
    download_ass = gr.File(label="Download ASS")
    
    transcribe_btn.click(
        main_interface, inputs=[audio_input, lang_dropdown, edit_font, edit_color],
        outputs=[timeline_out, edits_out, ass_out, gr.Textbox(visible=False)]
    )
    # Add download logic if needed

if __name__ == "__main__":
    demo.launch()
