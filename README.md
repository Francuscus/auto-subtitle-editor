# 🎵 Auto Subtitle Editor for Spanish Language Learning

**Version 5.00** — AI-Powered Spanish Verb Colorization with Transformers Whisper

A powerful tool for creating colorized, timestamped subtitles from Spanish audio. Perfect for language learners who want to visualize verb conjugations while listening to music, podcasts, or videos.

---

## ✨ What This Does

1. **Upload Spanish audio** (songs, podcasts, dialogue)
2. **Automatic transcription** using OpenAI Whisper (medium model)
3. **AI verb detection** using spaCy Spanish NLP model
4. **Automatic colorization** of verb endings to highlight conjugation patterns
5. **Export** as .srt subtitles or colorized video

---

## 🎯 Perfect For

- **Spanish learners** who want to see verb patterns in songs
- **Teachers** creating educational materials
- **Content creators** making language learning videos
- **Karaoke enthusiasts** who want colorized lyrics

---

## 🌟 Key Features

### 🤖 AI-Powered Verb Detection
- Uses **spaCy** Spanish NLP model to identify verbs (not nouns!)
- Only colors **verb endings** (minimal coloring: -o, -mos, -s, -n, etc.)
- Smart detection prevents false positives

### 🎤 Professional Transcription
- **Whisper-medium** model for high accuracy
- Word-level timestamps for perfect sync
- Supports auto-detection or forced Spanish

### 🎨 Full Customization
- Custom color picker for any word
- Font selection (16 fonts)
- Text size, bold, italic, underline
- Highlight colors

### 📝 IPA Phonetic Support
- Optional IPA transcription for pronunciation
- Letter-level IPA coloring
- Helps with Spanish accent learning

### 📤 Multiple Export Formats
- `.srt` (SubRip) - universal subtitle format
- `.ass` (Advanced SubStation) - styled subtitles
- `.vtt` (WebVTT) - web subtitles
- Colorized video export (overlays subtitles on video)

### ⏱️ Timing Control
- Timing offset slider (-5s to +5s)
- Adjust subtitle sync in real-time

---

## 📸 Demo: Verb Colorization

**Before (plain text):**
```
Yo canto una canción y bailamos juntos en la fiesta cuando lleguen los amigos
```

**After (with AI verb colorization):**
```
Yo cant[o] una canción y bailam[os] juntos en la fiesta cuando llegue[n] los amigos
         ↑                    ↑↑                                   ↑
    (1st sing.)         (1st plural)                        (3rd plural)
```

The yellow highlighting shows:
- **-o** = 1st person singular (yo canto)
- **-mos** = 1st person plural (nosotros bailamos)
- **-n** = 3rd person plural (ellos lleguen)

This helps learners **visually recognize** conjugation patterns!

---

## 🚀 Quick Start (Run Locally)

### Prerequisites
- Python 3.9, 3.10, or 3.11
- 2 GB free disk space (for models)
- 4 GB RAM minimum

### Installation

```bash
# Clone the repository
git clone https://github.com/Francuscus/auto-subtitle-editor.git
cd auto-subtitle-editor

# Install dependencies (takes 5-10 minutes)
pip install -r requirements.txt

# Run the app
python app.py
```

The app will open at **http://localhost:7860**

---

## 📖 How to Use

### Step 1: Upload Audio
- Click "**Upload Audio**" and select your Spanish audio file
- Supports: .mp3, .wav, .m4a, .ogg, .flac

### Step 2: Transcribe
- Click "**Transcribe Audio**"
- Wait 2-10 minutes (depends on audio length)
- Words appear in the editor

### Step 3: Auto-Color Verbs
- Click "**Auto-Color Spanish Verbs**"
- Status shows: "✅ Colored X verb(s) using AI (spaCy) detection"
- Verb endings are now highlighted in yellow

### Step 4: Customize (Optional)
- Select words and change colors
- Adjust font, size, formatting
- Add IPA transcription if desired

### Step 5: Export
- Download as **.srt** file for video players
- Or export as **colorized video**

---

## 📦 What's Included

```
auto-subtitle-editor/
├── app.py                    # Main Gradio application
├── requirements.txt          # Python dependencies
├── packages.txt             # System dependencies (ffmpeg)
├── editor.html              # Optional standalone HTML editor
├── QUICKSTART.html          # Visual quick start guide
└── README.md               # This file
```

---

## 🔧 Technical Details

### Dependencies
- **Gradio** 4.44.0 - Web interface
- **Transformers** 4.35+ - Whisper model
- **spaCy** 3.7+ - Spanish NLP
- **torch** 2.8.0 - Neural network backend
- **librosa** 0.10+ - Audio processing

### Why Transformers Whisper?
We switched from WhisperX to Hugging Face Transformers because:
- ✅ No ctranslate2 dependency (works everywhere)
- ✅ Compatible with Hugging Face Spaces
- ✅ Same accuracy as WhisperX
- ✅ Easier to install

### Model Sizes
- **Whisper-medium**: ~1.5 GB (high accuracy, slower)
- **spaCy es_core_news_sm**: ~15 MB (Spanish NLP)

### Performance
- **Transcription speed**: ~2-3x audio length on CPU
  - 4-minute song = 8-10 minutes transcription
  - Using GPU: ~1x audio length (real-time)

---

## 📝 Version History

### Version 5.00 (November 2025) — Indigo Banner
- **Major fix**: Proper word extraction from transformers pipeline
- Fixed verb colorization (was only coloring 1 word, now colors all verbs)
- Switched from WhisperX to transformers Whisper
- Compatible with Hugging Face Spaces

### Version 4.5 (October 2025) — Cyan Banner
- Added spaCy AI-powered verb detection
- Status indicators for AI vs rule-based detection

### Version 4.3 (October 2025) — Orange Banner
- IPA letter-level coloring
- Timing offset slider

### Version 4.1 (October 2025) — Blue Banner
- Custom color picker with black support
- Third-person verb improvements

---

## 🐛 Troubleshooting

### Transcription takes too long
- **Normal**: 2-3x audio length on CPU
- **Solution**: Use shorter audio clips, or get a GPU
- **Alternative**: Switch to whisper-small (faster, less accurate)

### "spaCy model not found"
```bash
pip install spacy
python -m spacy download es_core_news_sm
```

### Verb colorization not working
- Make sure transcription completed first
- Check status message says "AI (spaCy)" not "Rule-based"
- If still broken, check Container/Console logs for errors

### Numpy binary incompatibility
```bash
pip install numpy==1.26.4 --force-reinstall
```

---

## 🎓 Educational Background

This tool was created to help Spanish language learners **visually recognize** verb conjugation patterns. Traditional textbooks show verb tables, but seeing verbs **in context** (songs, dialogues) with **color highlighting** helps reinforce:

- Person/number agreement (yo/tú/él vs nosotros/ellos)
- Tense patterns (present, preterite, imperfect, subjunctive)
- Regular vs irregular verb endings

The minimal coloring approach (only endings) keeps text readable while highlighting the grammatically significant parts.

---

## 💡 Tips & Tricks

### Best Results
- Use **high-quality audio** (clear vocals, minimal background noise)
- Spanish songs work best (clear pronunciation)
- Podcasts and audiobooks also work well

### Workflow Optimization
1. Transcribe multiple songs in batch (run overnight)
2. Use "Download words.json" to save transcriptions
3. Import JSON into editor.html for quick styling
4. Export as .srt and reuse with different videos

### Custom Coloring
- Yellow = verbs (auto)
- Red = important vocabulary
- Blue = nouns
- Green = adjectives
- Cyan = phrases to memorize

---

## 🤝 Contributing

Found a bug? Have a feature request?

1. Open an issue at https://github.com/Francuscus/auto-subtitle-editor/issues
2. Describe what you expected vs what happened
3. Include your Python version and OS

Pull requests welcome!

---

## 📄 License

This project is open source. Feel free to use, modify, and share!

---

## 🙏 Credits

- **OpenAI Whisper** - Speech recognition model
- **spaCy** - Spanish NLP library
- **Gradio** - Web interface framework
- **Hugging Face** - Model hosting and transformers library

---

## 📧 Contact

Questions? Visit the [GitHub Issues](https://github.com/Francuscus/auto-subtitle-editor/issues) page.

---

<div align="center">

**Made with ❤️ for Spanish language learners**

Version 5.00 — Indigo Banner — November 2025

[⬆ Back to Top](#-auto-subtitle-editor-for-spanish-language-learning)

</div>
