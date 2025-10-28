# 🚨 Known Issues and Limitations

## Acentos Server on Hugging Face

**Issue**: When clicking the IPA transcription buttons (🇩🇴 Dominican, 🇲🇽 Mexican, etc.), you may see:
```
❌ Error: Cannot connect to acentos server. Is it running on port 5000?
```

**Why**: The acentos server (`acentos_server.py`) needs to run separately on port 5000. Hugging Face Spaces only runs one server at a time (the main Gradio app), so the acentos API server cannot run there.

**Solutions**:

### Option 1: Use Locally (Recommended for IPA features)
If you want to use the automatic IPA transcription:

1. Clone the repository to your computer
2. Install dependencies: `pip install -r requirements.txt`
3. Start acentos server: `python acentos_server.py` (in one terminal)
4. Start main app: `python app.py` (in another terminal)
5. Open http://localhost:7860 in your browser

### Option 2: Manual IPA (Works on Hugging Face)
You can still add IPA characters manually:

1. Click the **"📚 Full IPA Picker"** button in the toolbar
2. Click any IPA symbol to insert it
3. Or use the 4 configurable accent buttons

This works perfectly on Hugging Face!

---

## Color Persistence Issue

**Issue**: Colors applied in the editor don't always persist in the exported video.

**Why**: The rich text editor uses browser JavaScript to apply colors, but those changes aren't automatically synced back to the server state before video export.

**Temporary Workaround**:
- Apply colors, then click "Update Preview" before exporting
- We're working on automatic sync

**Coming Soon**: Auto-sync of editor changes to state

---

## Background Color

**Issue**: Background might appear black even when another color is selected.

**Status**: Fixed in latest version! Make sure you're using the latest code.

**If still happening**:
1. Check that you selected a color in "Background Color" picker
2. Make sure it's not set to black (#000000)
3. Try a bright color like red (#FF0000) to test

---

## Text Position

**New Feature**: You can now position text at:
- **Bottom** (default - like YouTube subtitles)
- **Center** (middle of screen)
- **Top** (title-style)

Select in the "Text Position" dropdown in Settings!

---

For more help, see:
- Main README.md
- ACENTOS_README.md (for acentos integration details)
- GitHub Issues: https://github.com/Francuscus/auto-subtitle-editor/issues
