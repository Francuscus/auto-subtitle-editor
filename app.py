# --- Wiring (replace everything from "# Wiring" down to render_btn.click(...)) ---

# States to hold file paths + duration so we can reuse them in downloads AND rendering
srt_path_state = gr.State()
ass_path_state = gr.State()
vtt_path_state = gr.State()
duration_state = gr.State()

def _run_and_return(*args):
    try:
        # (chunks, table, preview_html, px, py, srt_path, ass_path, vtt_path, duration)
        res = run_pipeline(*args)
        chunks, table, preview_html, px, py, srt_path, ass_path, vtt_path, duration = res

        # Return visible outputs first, then the states, then file paths for DownloadButtons
        return (
            preview_html,                                # preview (HTML)
            [{"start": round(c["start"],2), "end": round(c["end"],2), "text": c["text"]} for c in chunks],  # table (JSON)
            px, py,                                      # states (unused visually)
            srt_path, ass_path, vtt_path,                # for DownloadButtons
            duration,                                    # duration state
            srt_path, ass_path, vtt_path                 # also store into States
        )
    except Exception as e:
        # surface the error in preview box
        return (f"<div style='color:#ff6b6b'>Error: {gr.utils.sanitize_html(str(e))}</div>",
                gr.skip(), gr.skip(), gr.skip(),
                gr.skip(), gr.skip(), gr.skip(),
                0.0,
                gr.skip(), gr.skip(), gr.skip())

# NOTE: DownloadButton expects the *file path* as its output.
run.click(
    _run_and_return,
    inputs=[media, language, words_per_chunk, layout, gr.Textbox(visible=False),
            font, fsize, text_color, outline_color, outline_w, boxed_bg, bg_color, global_shift],
    outputs=[
        preview,              # HTML
        table,                # JSON
        gr.State(),           # px (not shown)
        gr.State(),           # py (not shown)
        srt_dl,               # DownloadButton -> receives srt path
        ass_dl,               # DownloadButton -> receives ass path
        vtt_dl,               # DownloadButton -> receives vtt path
        duration_state,       # State: duration
        srt_path_state,       # State: srt path
        ass_path_state,       # State: ass path
        vtt_path_state        # State: vtt path
    ]
)

def _render(media_path, ass_path, layout, duration):
    if not ass_path:
        raise gr.Error("No subtitle file generated yet. Click Run first.")
    try:
        out = render_video(media_path, ass_path, layout, duration or 0.0)
        return out
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"ffmpeg failed. Check logs.\n{e}")
    except Exception as e:
        raise gr.Error(str(e))

# IMPORTANT: we now pass the ASS path from the State (not from a DownloadButton)
render_btn.click(
    _render,
    inputs=[media, ass_path_state, layout, duration_state],
    outputs=rendered
)
