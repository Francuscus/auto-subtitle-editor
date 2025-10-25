# app.py — Colorvideo Subs (v0.9.3)

from __future__ import annotations
import os, re, json, math, subprocess, uuid, shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import gradio as gr

# ASR
import torch
import whisperx

# ---------- Paths ----------
TMP = Path("/tmp")
OUT = TMP / "subs"
OUT.mkdir(exist_ok=True, parents=True)

# ---------- Language helpers ----------
LANG_CHOICES = [
    ("Auto-detect", "auto"),
    ("Spanish (es)", "es"),
    ("Hungarian (hu)", "hu"),
    ("English (en)", "en"),
]
LANG_MAP = {"auto": None, "es": "es", "hu": "hu", "en": "en",
            "spanish":"es","hungarian":"hu","english":"en"}

def norm_lang(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = s.strip().lower()
    if t in LANG_MAP:
        return LANG_MAP[t]
    m = re.search(r"\b([a-z]{2,3})\b", t)
    return m.group(1) if m else None

# ---------- Color helpers ----------
def _parse_color_any(c: str) -> Tuple[int,int,int]:
    """Return (r,g,b) from '#RRGGBB' or 'rgb(a)' strings."""
    c = (c or "").strip()
    if c.startswith("#"):
        hx = c.lstrip("#")
        if len(hx) == 3:
            hx = "".join(ch*2 for ch in hx)
        hx = (hx + "000000")[:6]
        r = int(hx[0:2], 16); g = int(hx[2:4], 16); b = int(hx[4:6], 16)
        return r,g,b
    m = re.match(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", c)
    if m:
        r,g,b = map(int, m.groups())
        return max(0,min(255,r)), max(0,min(255,g)), max(0,min(255,b))
    # default white if unknown
    return (255,255,255)

def hex_to_ass_bgr(color: str, alpha: int = 0) -> str:
    """ASS uses &HAA BB GG RR — we return like &H00BBGGRR"""
    r,g,b = _parse_color_any(color)
    a = max(0, min(255, alpha))
    return f"&H{a:02X}{b:02X}{g:02X}{r:02X}"

# ---------- ASR model ----------
_asr = None
def get_asr():
    global _asr
    if _asr is not None:
        return _asr
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"   # CPU-safe
    try:
        _asr = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError:
        # last-resort fallback on some CPUs
        _asr = whisperx.load_model("small", device=device, compute_type="int16" if device=="cpu" else "float32")
    return _asr

# ---------- Transcribe ----------
def transcribe(audio_path: str, language_code: Optional[str]) -> Tuple[List[Dict], str, float]:
    model = get_asr()
    res = model.transcribe(audio_path, language=language_code)
    segs = res["segments"]
    text = " ".join(s["text"].strip() for s in segs)
    # estimate duration
    duration = 0.0
    if segs:
        duration = float(segs[-1]["end"])
    return segs, text, duration

# ---------- Chunking ----------
def chunk_by_words(text: str, words_per: int = 5) -> List[Dict]:
    tokens = text.strip().split()
    chunks = []
    i = 0
    t = 0.0
    step = 1.5  # naive timing per chunk; refined later if aligning to segs
    while i < len(tokens):
        w = tokens[i:i+max(1,words_per)]
        s = " ".join(w)
        chunks.append({"start": t, "end": t+step, "text": s})
        t += step
        i += words_per
    return chunks

def retime_to_segments(lyrics_lines: List[str], segs: List[Dict]) -> List[Dict]:
    """Distribute user lyrics across detected segments."""
    out = []
    if not lyrics_lines:
        return out
    seg_idx = 0
    for line in lyrics_lines:
        line = line.strip()
        if not line:
            continue
        if seg_idx >= len(segs):
            # append to last segment end + small pad
            st = out[-1]["end"] + 0.2 if out else 0.0
            out.append({"start": st, "end": st+1.2, "text": line})
        else:
            st = float(segs[seg_idx]["start"])
            en = float(segs[seg_idx]["end"])
            out.append({"start": st, "end": en, "text": line})
            seg_idx += 1
    return out

# ---------- Formats ----------
def to_srt(chunks: List[Dict]) -> str:
    def fmt(t: float):
        h = int(t//3600); m = int((t%3600)//60); s = int(t%60); ms = int((t - int(t))*1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    lines = []
    for i,c in enumerate(chunks,1):
        lines += [str(i),
                  f"{fmt(c['start'])} --> {fmt(c['end'])}",
                  c['text'].strip(), ""]
    return "\n".join(lines)

def to_vtt(chunks: List[Dict]) -> str:
    def fmt(t: float):
        h = int(t//3600); m = int((t%3600)//60); s = int(t%60); ms = int((t - int(t))*1000)
        return f"{h:02}:{m:02}:{s:02}.{ms:03}"
    lines = ["WEBVTT",""]
    for c in chunks:
        lines += [f"{fmt(c['start'])} --> {fmt(c['end'])}",
                  c['text'].strip(), ""]
    return "\n".join(lines)

def to_ass(chunks: List[Dict], *,
           font: str, size: int,
           txt_color: str,
           outline_w: int, outline_color: str,
           boxed_bg: bool, bg_color: str) -> str:
    prim = hex_to_ass_bgr(txt_color)         # &H00BBGGRR
    outc = hex_to_ass_bgr(outline_color)
    back = hex_to_ass_bgr(bg_color, alpha=0x40 if boxed_bg else 0xFF)  # semi if boxed, opaque if not used

    style = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1280\n"
        "PlayResY: 720\n"
        "\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Sub,{font},{size},{prim},&H00000000,{outc},{back},0,0,0,0,100,100,0,0,1,{outline_w},0,2,60,60,36,1\n"
        "\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    def fmt(t: float):
        h = int(t//3600); m = int((t%3600)//60); s = int(t%60); cs = int(round((t - int(t))*100))
        return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

    lines = [style]
    for c in chunks:
        start = fmt(float(c["start"])); end = fmt(float(c["end"]))
        text = c["text"].replace("\n"," ").replace("\r"," ")
        lines.append(f"Dialogue: 0,{start},{end},Sub,,0,0,20,,{text}")
    return "\n".join(lines)

# ---------- Simple preview HTML ----------
def preview_html(chunks: List[Dict], font: str, size: int, color: str,
                 outline_w: int, outline_color: str, boxed: bool, bg_color: str) -> str:
    box_css = f"background:{bg_color}; padding:4px 8px; border-radius:6px;" if boxed else ""
    return f"""
<div style="display:flex;align-items:center;justify-content:center;height:220px;border:1px solid #333;border-radius:8px;background:#111;">
  <span style="font-family:{font},sans-serif;font-size:{size}px;color:{color};
               text-shadow:-{outline_w}px 0 {outline_color}, {outline_w}px 0 {outline_color},
                           0 -{outline_w}px {outline_color}, 0 {outline_w}px {outline_color}; {box_css}">
    {chunks[0]['text'] if chunks else ''}
  </span>
</div>
"""

# ---------- Build everything from either auto chunks or lyrics ----------
def build_from_chunks(chunks: List[Dict],
                      font: str, size: int,
                      txt_color: str, outline_color: str, outline_w: int,
                      boxed_bg: bool, bg_color: str) -> Tuple[str,Dict,Dict,Path,Path,Path]:
    # for table/debug
    table_data = [{"start": round(c["start"],2), "end": round(c["end"],2), "text": c["text"]} for c in chunks]

    # files
    sid = uuid.uuid4().hex
    srt_path = OUT / f"{sid}.srt"
    ass_path = OUT / f"{sid}.ass"
    vtt_path = OUT / f"{sid}.vtt"
    srt_path.write_text(to_srt(chunks), encoding="utf-8")
    vtt_path.write_text(to_vtt(chunks), encoding="utf-8")

    ass_txt = to_ass(chunks,
                     font=font, size=int(size),
                     txt_color=txt_color,
                     outline_w=int(outline_w), outline_color=outline_color,
                     boxed_bg=boxed_bg, bg_color=bg_color)
    ass_path.write_text(ass_txt, encoding="utf-8")

    prev = preview_html(chunks, font, int(size), txt_color, int(outline_w), outline_color, boxed_bg, bg_color)
    return prev, {"data":table_data}, {"chunks":chunks}, srt_path, ass_path, vtt_path

# ---------- Render MP4 with ASS overlay ----------
def render_mp4(input_media: str, ass_file: str) -> str:
    out_mp4 = OUT / f"{uuid.uuid4().hex}.mp4"
    cmd = [
        "ffmpeg","-y",
        "-i", input_media,
        "-vf", f"ass={ass_file}",
        "-c:a","copy",
        str(out_mp4)
    ]
    subprocess.run(cmd, check=True)
    return str(out_mp4)

# ---------- Gradio States ----------
srt_path_state = gr.State()
ass_path_state = gr.State()
vtt_path_state = gr.State()
input_media_state = gr.State()
chunks_state = gr.State()
duration_state = gr.State(0.0)

# ---------- Pipeline ----------
def run_pipeline(audio_path: str,
                 language_ui: str,
                 words_per_chunk: int,
                 layout_preset: str,
                 lyrics_mode: bool,
                 pasted_lyrics: str,
                 global_shift: float,
                 font: str, font_size: int, text_color: str,
                 outline_color: str, outline_w: int,
                 boxed_bg: bool, bg_color: str):

    try:
        if not audio_path:
            return ("Please upload audio/video first.", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                    gr.skip(), gr.skip(), gr.skip())

        lang = norm_lang(language_ui)
        segs, joined, dur = transcribe(audio_path, lang)

        # pick chunks
        if lyrics_mode and pasted_lyrics.strip():
            lines = [ln for ln in pasted_lyrics.splitlines() if ln.strip()]
            chunks = retime_to_segments(lines, segs)
        else:
            chunks = chunk_by_words(joined, max(1,int(words_per_chunk)))

        # apply global shift
        if abs(global_shift) > 1e-6:
            for c in chunks:
                c["start"] = max(0.0, float(c["start"]) + global_shift)
                c["end"] = max(c["start"] + 0.01, float(c["end"]) + global_shift)

        prev_html, table_json, chunks_json, srt_p, ass_p, vtt_p = build_from_chunks(
            chunks, font, font_size, text_color, outline_color, outline_w, boxed_bg, bg_color
        )

        return (prev_html, table_json, chunks_json,
                str(srt_p), str(ass_p), str(vtt_p),
                audio_path, json.dumps({"duration": dur}), json.dumps({"ok": True}))

    except Exception as e:
        return (f"Error: {e}", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())

def nudge(chunks_json: Dict, delta: float, apply: bool):
    try:
        if not apply:
            return chunks_json
        chunks = chunks_json.get("chunks", [])
        for c in chunks:
            c["start"] = max(0.0, float(c["start"]) + delta)
            c["end"] = max(c["start"] + 0.01, float(c["end"]) + delta)
        return {"chunks": chunks}
    except Exception:
        return chunks_json

def refresh_build(chunks_json: Dict,
                  font: str, font_size: int, text_color: str,
                  outline_color: str, outline_w: int,
                  boxed_bg: bool, bg_color: str):
    try:
        chunks = chunks_json.get("chunks", [])
        prev_html, table_json, chunks_json2, srt_p, ass_p, vtt_p = build_from_chunks(
            chunks, font, font_size, text_color, outline_color, outline_w, boxed_bg, bg_color
        )
        return (prev_html, table_json, chunks_json2, str(srt_p), str(ass_p), str(vtt_p))
    except Exception as e:
        return (f"Error: {e}", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())

def do_render(mp4_btn, input_media: str, ass_path: str):
    # mp4_btn is just the click event payload, unused
    if not input_media or not ass_path:
        raise gr.Error("Run the pipeline first, then render.")
    try:
        out_path = render_mp4(input_media, ass_path)
        return out_path
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"ffmpeg failed: {e}")

# ---------- UI ----------
custom_css = """
.gradio-container { max-width: 1200px !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("### 🎨 Colorvideo Subs — v0.9.3")

    with gr.Row():
        with gr.Column(scale=3):
            audio = gr.Audio(label="Audio / Video (mp3, wav, mp4…)", type="filepath")
            language = gr.Dropdown(LANG_CHOICES, value="auto", label="Language")

            words = gr.Slider(3, 8, value=5, step=1, label="Words per chunk (when not using lyrics)")
            layout = gr.Dropdown(choices=["16:9 (YouTube)","9:16 (TikTok)","1:1 (Square)"],
                                 value="16:9 (YouTube)", label="Layout preset")

            with gr.Accordion("Lyrics mode (optional)", open=False):
                use_lyrics = gr.Checkbox(label="Use pasted lyrics to retime", value=False)
                lyrics_box = gr.Textbox(label="Paste lyrics or lines here", placeholder="One line per subtitle...", lines=6)

            with gr.Accordion("Timing tools", open=False):
                shift = gr.Slider(-5, 5, value=0, step=0.1, label="Global shift (seconds)")

            run = gr.Button("Run", variant="primary")

            prev_html = gr.HTML(label="Preview")
            table = gr.Dataframe(headers=["start","end","text"], label="Segments table", interactive=False)
            chunks_json = gr.JSON(label="Chunks JSON")

            srt_dl = gr.File(label="Download SRT")
            ass_dl = gr.File(label="Download ASS")
            vtt_dl = gr.File(label="Download VTT")

            render_btn = gr.Button("Render subtitle video (MP4) 🖼️", variant="secondary")
            rendered = gr.File(label="Rendered preview")

        with gr.Column(scale=1):
            gr.Markdown("#### Subtitle Style")
            font = gr.Dropdown(["Default","Arial","Roboto","Open Sans","Lato","Montserrat"], value="Default", label="Font")
            font_size = gr.Slider(14, 72, value=36, step=1, label="Font size")

            text_color = gr.ColorPicker(value="#FFFFFF", label="Text color")         # <— now visible & used
            outline_color = gr.ColorPicker(value="#000000", label="Outline color")
            outline_w = gr.Slider(0, 6, value=2, step=1, label="Outline width (px)")

            gr.Markdown("#### Background")
            boxed_bg = gr.Checkbox(value=False, label="Boxed background behind text")
            bg_color = gr.ColorPicker(value="#000000", label="Background color")

            gr.Markdown("#### Per-line editor")
            apply_refresh = gr.Checkbox(value=True, label="Apply edits & refresh")
            nudge_minus = gr.Button("◀ Nudge -0.10s")
            nudge_plus = gr.Button("▶ Nudge +0.10s")

    # --- wire actions ---
    run_outputs = [prev_html, table, chunks_json, srt_dl, ass_dl, vtt_dl, input_media_state, duration_state, gr.JSON()]
    run.click(
        run_pipeline,
        inputs=[audio, language, words, layout, use_lyrics, lyrics_box, shift,
                font, font_size, text_color, outline_color, outline_w, boxed_bg, bg_color],
        outputs=run_outputs
    )

    nudge_minus.click(lambda cj, ap: nudge(cj, -0.10, ap), inputs=[chunks_json, apply_refresh], outputs=[chunks_json])\
               .then(refresh_build,
                     inputs=[chunks_json, font, font_size, text_color, outline_color, outline_w, boxed_bg, bg_color],
                     outputs=[prev_html, table, chunks_json, srt_dl, ass_dl, vtt_dl])

    nudge_plus.click(lambda cj, ap: nudge(cj, +0.10, ap), inputs=[chunks_json, apply_refresh], outputs=[chunks_json])\
              .then(refresh_build,
                    inputs=[chunks_json, font, font_size, text_color, outline_color, outline_w, boxed_bg, bg_color],
                    outputs=[prev_html, table, chunks_json, srt_dl, ass_dl, vtt_dl])

    render_btn.click(do_render, inputs=[render_btn, input_media_state, ass_dl], outputs=[rendered])

if __name__ == "__main__":
    demo.launch()
