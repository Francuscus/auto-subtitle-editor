# app.py — Subtitle maker (5-word chunks, lyrics timing, SRT/ASS/VTT, MP4 render)
# Version: v1.4 (2025-10-25)
#
# Highlights
# - Works with Audio OR Video uploads.
# - Languages: Auto, Hungarian (hu), Spanish (es).
# - Chunks by N words (default 5) when not using lyrics.
# - Optional lyrics mode (paste 1 line per subtitle) → auto time over detected duration.
# - Global nudge/shift (± seconds).
# - Styling: font, size, text color, outline color/width, optional boxed background color.
# - Live preview with proper colors (no theme overrides).
# - Exports SRT/ASS/VTT and burns ASS into MP4 (keeps video if input is video; for audio creates colored canvas).
# - Robust color parsing: #RGB, #RRGGBB, rgb(), rgba(), common names.

import os, re, json, uuid, math, shutil, tempfile, subprocess
from typing import List, Dict, Tuple, Optional

import gradio as gr
import torch
import whisperx

# -------------------- Language helpers --------------------
LANG_MAP = {"auto": None, "hungarian": "hu", "spanish": "es"}

def normalize_lang(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    t = label.strip().lower()
    if t in ("auto", "auto-detect", "autodetect"):
        return None
    if t in LANG_MAP:
        return LANG_MAP[t]
    m = re.search(r"\b([a-z]{2,3})\b", t)
    return m.group(1) if m else None

# -------------------- Color utilities --------------------
_NAMED = {
    "black":"#000000","white":"#ffffff","red":"#ff0000","green":"#00ff00","blue":"#0000ff",
    "yellow":"#ffff00","cyan":"#00ffff","magenta":"#ff00ff","gray":"#808080","grey":"#808080"
}

def normalize_hex(s: str, fallback="#ffffff") -> str:
    if not s:
        return fallback
    t = s.strip().lower()
    if t in _NAMED: return _NAMED[t]
    if t.startswith("#"):
        hx = t[1:]
        if len(hx)==3 and all(c in "0123456789abcdef" for c in hx):
            return "#" + "".join(c*2 for c in hx)
        if len(hx)==6 and all(c in "0123456789abcdef" for c in hx):
            return "#" + hx
        return fallback
    m = re.match(r"rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([0-9\.]+))?\s*\)", t)
    if m:
        r = max(0, min(255, int(m.group(1))))
        g = max(0, min(255, int(m.group(2))))
        b = max(0, min(255, int(m.group(3))))
        return f"#{r:02x}{g:02x}{b:02x}"
    return fallback

def hex_to_ass_bgr(hex_color: str) -> str:
    hx = normalize_hex(hex_color).lstrip("#")
    r = int(hx[0:2],16); g = int(hx[2:4],16); b = int(hx[4:6],16)
    # ASS uses BGR in &HBBGGRR&
    return f"&H{b:02X}{g:02X}{r:02X}&"

def hex_to_ffmpeg_color(hex_color: str) -> str:
    hx = normalize_hex(hex_color).lstrip("#")
    return "0x"+hx  # for lavfi color source

# -------------------- ASR (lazy) --------------------
_ASR = None
def get_asr():
    global _ASR
    if _ASR is not None:
        return _ASR
    use_cuda = torch.cuda.is_available()
    device   = "cuda" if use_cuda else "cpu"
    compute  = "float16" if use_cuda else "int8"
    try:
        _ASR = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        if "compute type" in str(e).lower():
            fallback = "int16" if device=="cpu" else "float32"
            _ASR = whisperx.load_model("small", device=device, compute_type=fallback)
        else:
            raise
    return _ASR

# -------------------- Transcribe --------------------
def transcribe(audio_path: str, lang_code: Optional[str]):
    model = get_asr()
    result = model.transcribe(audio_path, language=lang_code)
    segs = result.get("segments", [])
    text = " ".join((s.get("text") or "").strip() for s in segs).strip()
    if segs:
        t0 = float(segs[0].get("start",0.0) or 0.0)
        t1 = float(segs[-1].get("end", t0) or t0)
    else:
        t0, t1 = 0.0, 0.0
    return segs, text, t0, t1

# -------------------- Chunking --------------------
def split_by_words(text: str, start: float, end: float, n: int) -> List[Dict]:
    toks = [t for t in text.split() if t]
    if not toks: return []
    dur = max(0.0, (end or 0.0)-(start or 0.0))
    per = dur/max(1,len(toks))
    out=[]; i=0
    while i < len(toks):
        grp=toks[i:i+n]; m=len(grp)
        s=start+i*per; e=start+(i+m)*per
        out.append({"start":round(s,3),"end":round(e,3),"text":" ".join(grp)})
        i+=n
    return out

def chunk_segments(segs: List[Dict], n: int) -> List[Dict]:
    out=[]
    for s in segs:
        t=s.get("text","").strip()
        out+=split_by_words(t, float(s.get("start",0) or 0), float(s.get("end",0) or 0), n)
    return out

# -------------------- Lyrics timing --------------------
def parse_lyrics(raw: str) -> List[str]:
    return [ln.strip() for ln in raw.replace("\r\n","\n").split("\n") if ln.strip()]

def wc(s: str) -> int:
    return len([t for t in s.split() if t])

def align_lyrics(lines: List[str], t0: float, t1: float) -> List[Dict]:
    if not lines: return []
    total = sum(max(1,wc(l)) for l in lines)
    dur   = max(0.0, t1-t0)
    cur=t0; out=[]
    for ln in lines:
        w=max(1,wc(ln)); d=dur*(w/total)
        out.append({"start":round(cur,3),"end":round(cur+d,3),"text":ln})
        cur+=d
    out[-1]["end"]=round(t1,3)
    return out

# -------------------- Global shift --------------------
def shift_chunks(chunks: List[Dict], shift: float) -> List[Dict]:
    if not shift: return chunks
    out=[]
    for c in chunks:
        out.append({"start":max(0.0, round(c["start"]+shift,3)),
                    "end":  max(0.0, round(c["end"]+shift,3)),
                    "text": c["text"]})
    return out

# -------------------- Formats --------------------
def _srt_time(t: float) -> str:
    t=max(0.0,t); h=int(t//3600); m=int((t%3600)//60); s=int(t%60); ms=int(round((t-int(t))*1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def to_srt(ch: List[Dict]) -> str:
    out=[]
    for i,c in enumerate(ch,1):
        out += [str(i), f"{_srt_time(c['start'])} --> {_srt_time(c['end'])}", c["text"], ""]
    return "\n".join(out)

def _ass_time(t: float) -> str:
    t=max(0.0,t); h=int(t//3600); m=int((t%3600)//60); s=int(t%60); cs=int(round((t-int(t))*100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def to_ass(ch: List[Dict], W:int,H:int, font:str, size:int,
           text_hex:str, outline_hex:str, outline_w:int,
           bg_box:bool, bg_hex:str, margin_v:int=40) -> str:

    text_hex   = normalize_hex(text_hex)
    outline_hex= normalize_hex(outline_hex)
    bg_hex     = normalize_hex(bg_hex)

    primary = hex_to_ass_bgr(text_hex)
    outline = hex_to_ass_bgr(bg_hex if bg_box else outline_hex)
    border  = max(0, outline_w if not bg_box else max(outline_w, 6))

    hdr = [
        "[Script Info]","ScriptType: v4.00+",
        f"PlayResX: {W}", f"PlayResY: {H}",
        "ScaledBorderAndShadow: yes","",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font},{size},{primary},&H00FFFFFF,{outline},&H7F000000,"
        f"0,0,0,0,100,100,0,0,1,{border},0,2,30,30,{margin_v},1","",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    ]
    ev=[]
    for c in ch:
        txt=(c["text"] or "").replace("\n","\\N")
        ev.append(f"Dialogue: 0,{_ass_time(c['start'])},{_ass_time(c['end'])},Default,,0,0,0,,{txt}")
    return "\n".join(hdr+ev)

def _vtt_time(t: float) -> str:
    t=max(0.0,t); h=int(t//3600); m=int((t%3600)//60); s=int(t%60); ms=int(round((t-int(t))*1000))
    return f"{h:02}:{m:02}:{s:02}.{ms:03}" if h else f"{m:02}:{s:02}.{ms:03}"

def to_vtt(ch: List[Dict]) -> str:
    out=["WEBVTT",""]
    for c in ch:
        out += [f"{_vtt_time(c['start'])} --> {_vtt_time(c['end'])}", c["text"], ""]
    return "\n".join(out)

# -------------------- Preview HTML --------------------
def canvas_size(preset: str) -> Tuple[int,int]:
    return {"9:16 (TikTok)":(1080,1920),
            "1:1 (Square)": (1080,1080),
            "16:9 (YouTube)":(1920,1080)}.get(preset,(1920,1080))

def build_preview_html(ch: List[Dict], font:str, size:int,
                       text_hex:str, outline_hex:str, outline_w:int,
                       bg_box:bool, bg_hex:str) -> str:
    text_hex   = normalize_hex(text_hex)
    outline_hex= normalize_hex(outline_hex)
    bg_hex     = normalize_hex(bg_hex)

    # text shadow for outline preview
    shadow = " ,".join([
        f"-{outline_w}px 0 {outline_hex}",
        f" {outline_w}px 0 {outline_hex}",
        f" 0 -{outline_w}px {outline_hex}",
        f" 0  {outline_w}px {outline_hex}",
    ]) if outline_w>0 and not bg_box else "none"

    box_css = f"background:{bg_hex}; padding:6px 10px; border-radius:8px;" if bg_box else ""

    items=[]
    for c in ch[:10]:  # show first 10 lines in preview list
        items.append(
            f"<div style='margin:6px 0; font-family:{font},sans-serif; "
            f"font-size:{size}px; color:{text_hex}!important; text-shadow:{shadow}; {box_css}'>"
            f"{gr.utils.sanitize_html(c['text'])}</div>"
        )
    if not items:
        items = ["<i style='opacity:.6'>No chunks yet. Click Run.</i>"]
    return "<div>"+ "".join(items) +"</div>"

# -------------------- IO helpers --------------------
def save_text(tmpdir: str, name: str, content: str) -> str:
    path = os.path.join(tmpdir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

def is_probably_video(path: str) -> bool:
    ext = os.path.splitext(path.lower())[1]
    return ext in (".mp4",".mov",".mkv",".webm",".avi",".m4v")

# -------------------- Pipeline --------------------
def run_pipeline(
    media_path: str,
    language_ui: str,
    words_per_chunk: int,
    layout_preset: str,
    use_lyrics: bool,
    lyrics_text: str,
    global_shift: float,
    font_family: str,
    font_size: int,
    text_color: str,
    outline_color: str,
    outline_width: int,
    boxed_bg: bool,
    bg_color: str,
):
    if not media_path:
        raise gr.Error("Please upload an audio or video file.")

    lang_code = normalize_lang(language_ui)

    segs, _, t0, t1 = transcribe(media_path, lang_code)
    if use_lyrics and lyrics_text.strip():
        lines  = parse_lyrics(lyrics_text)
        chunks = align_lyrics(lines, t0, t1)
    else:
        chunks = chunk_segments(segs, max(1, int(words_per_chunk)))

    chunks = shift_chunks(chunks, float(global_shift))

    # outputs
    tmpdir = tempfile.mkdtemp(prefix="subs_")
    W,H = canvas_size(layout_preset)

    srt_txt = to_srt(chunks)
    ass_txt = to_ass(chunks, W,H, font_family, int(font_size),
                     text_color, outline_color, int(outline_width),
                     bool(boxed_bg), bg_color)
    vtt_txt = to_vtt(chunks)

    srt_path = save_text(tmpdir, "subtitles.srt", srt_txt)
    ass_path = save_text(tmpdir, "subtitles.ass", ass_txt)
    vtt_path = save_text(tmpdir, "subtitles.vtt", vtt_txt)

    preview_html = build_preview_html(
        chunks, font_family, int(font_size),
        text_color, outline_color, int(outline_width),
        bool(boxed_bg), bg_color
    )

    # return data and file paths
    table = [{"start":c["start"], "end":c["end"], "text":c["text"]} for c in chunks]
    return chunks, table, preview_html, W, H, srt_path, ass_path, vtt_path

def render_video(
    media_path: str,
    ass_path: str,
    width: int,
    height: int,
    bg_color: str,
    duration_hint: float = None,
):
    if not ass_path or not os.path.exists(ass_path):
        raise gr.Error("No ASS file to burn. Run first.")
    outdir = tempfile.mkdtemp(prefix="render_")
    outfile = os.path.join(outdir, "subtitle_video.mp4")

    if is_probably_video(media_path):
        # Burn onto existing video
        cmd = [
            "ffmpeg","-y","-i",media_path,
            "-vf", f"subtitles={ass_path}:original_size={width}x{height}",
            "-c:a","copy","-c:v","libx264","-pix_fmt","yuv420p", outfile
        ]
    else:
        # Audio only → make a solid background canvas, add audio, burn subs
        dur = duration_hint or 0
        color = hex_to_ffmpeg_color(bg_color or "#111111")
        cmd = [
            "ffmpeg","-y",
            "-f","lavfi","-i", f"color={color}:s={width}x{height}:r=30:d={dur if dur>0 else 60}",
            "-i", media_path,
            "-vf", f"subtitles={ass_path}:original_size={width}x{height}",
            "-shortest",
            "-c:v","libx264","-pix_fmt","yuv420p",
            "-c:a","aac","-b:a","192k",
            outfile
        ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # write stderr so user can see
        err_path = os.path.join(outdir, "ffmpeg_error.txt")
        with open(err_path,"wb") as f: f.write(e.stderr or b"")
        raise gr.Error("FFmpeg failed. See logs tab for details.")
    return outfile

# -------------------- UI --------------------
FONT_CHOICES = ["Default","Arial","Roboto","Open Sans","Lato","Noto Sans","Montserrat"]
LANG_CHOICES = [("Auto-detect","auto"), ("Hungarian (hu)","hu"), ("Spanish (es)","es")]
LAYOUTS = ["16:9 (YouTube)","1:1 (Square)","9:16 (TikTok)"]

custom_css = """
.gradio-container { max-width: 1200px !important; }
.preview-card { background: #0f1220; border-radius: 12px; padding: 12px; }
.small-hint { color: #a8b3cf; font-size: 12px; }
label, .label-wrap span { color: #cbd5f7 !important; }
"""

with gr.Blocks(title="Colorvideo Subs — v1.4", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🎬 Colorvideo Subs — v1.4\n**Transcribe → Style → Export SRT/ASS/VTT → Burn into MP4**")

    with gr.Row():
        with gr.Column(scale=3):
            media = gr.Audio(label="Audio / Video (mp3, wav, mp4, ...)", type="filepath")
            language = gr.Dropdown(LANG_CHOICES, value="auto", label="Language")
            words_n = gr.Slider(1, 8, value=5, step=1, label="Words per chunk (when not using lyrics)")

            layout = gr.Dropdown(LAYOUTS, value="16:9 (YouTube)", label="Layout preset")
            with gr.Accordion("Lyrics mode (optional)", open=False):
                use_lyrics = gr.Checkbox(label="Use pasted lyrics to retime")
                lyrics_box = gr.Textbox(lines=8, placeholder="One line = one subtitle.\n(Leave unchecked to auto-chunk by words.)",
                                        label="Paste lyrics or lines here")

            shift = gr.Slider(-5, 5, value=0, step=0.1, label="Global shift (seconds)")
            run_btn = gr.Button("Run", variant="primary")

            # Preview + data
            preview = gr.HTML(label="Preview", elem_classes=["preview-card"])
            table = gr.Dataframe(headers=["start","end","text"], datatype=["number","number","str"],
                                 label="Segments (editable after download if needed)",
                                 interactive=False, wrap=True, height=220)
        with gr.Column(scale=2):
            gr.Markdown("### Subtitle Settings")
            font = gr.Dropdown(FONT_CHOICES, value="Default", label="Font")
            font_size = gr.Slider(14, 72, value=36, step=1, label="Font size")
            text_color = gr.ColorPicker(value="#FFFFFF", label="Text color")
            outline_color = gr.ColorPicker(value="#000000", label="Outline color")
            outline_w = gr.Slider(0, 6, value=2, step=1, label="Outline width (px)")

            gr.Markdown("#### Background")
            boxed_bg = gr.Checkbox(label="Boxed background behind text")
            bg_color = gr.ColorPicker(value="#111111", label="Background color (preview & audio canvas)")

            # hidden/derived
            px = gr.Number(visible=False)
            py = gr.Number(visible=False)
            srt_path = gr.File(label="SRT", file_count="single", visible=False)
            ass_path = gr.File(label="ASS", file_count="single", visible=False)
            vtt_path = gr.File(label="VTT", file_count="single", visible=False)

    # Bottom: downloads + render
    with gr.Row():
        srt_dl = gr.File(label="Download SRT", file_count="single")
        ass_dl = gr.File(label="Download ASS", file_count="single")
        vtt_dl = gr.File(label="Download VTT", file_count="single")

    render_btn = gr.Button("Render subtitle video (MP4) 🎞️")
    rendered = gr.File(label="MP4 output", file_count="single")

    # ---- Wire up ----
    def _run_and_return(*args):
        try:
            ch, tbl, prev, W, H, srt, ass, vtt = run_pipeline(*args)
            return prev, tbl, W, H, srt, ass, vtt, srt, ass, vtt
        except Exception as e:
            return (f"<div style='color:#ff8a8a'>Error: {gr.utils.sanitize_html(str(e))}</div>",
                    [], 1920,1080, None,None,None, None,None,None)

    run_btn.click(
        _run_and_return,
        inputs=[media, language, words_n, layout, use_lyrics, lyrics_box, shift,
                font, font_size, text_color, outline_color, outline_w, boxed_bg, bg_color],
        outputs=[preview, table, px, py, srt_path, ass_path, vtt_path, srt_dl, ass_dl, vtt_dl]
    )

    def _render(media_path, ass_file, W, H, bg):
        if not ass_file: raise gr.Error("No subtitles yet. Click Run.")
        # gr.File returns dict with 'name' key; handle both string/dict
        ass_p = ass_file if isinstance(ass_file, str) else (ass_file.get("name") if ass_file else None)
        return render_video(media_path, ass_p, int(W), int(H), bg, duration_hint=None)

    render_btn.click(
        _render,
        inputs=[media, ass_path, px, py, bg_color],
        outputs=[rendered]
    )

if __name__ == "__main__":
    demo.launch()
