# Acentos Integration Guide

This guide explains how to use the acentos IPA transcription integration with the lyric video editor.

## What is Acentos?

Acentos is a Spanish IPA (International Phonetic Alphabet) transcription tool that converts Spanish text into phonetic notation for 4 different dialects:

- 🇩🇴 **Dominican Spanish**
- 🇲🇽 **Mexican Spanish**
- 🇦🇷 **Rioplatense Spanish** (Argentina/Uruguay)
- 🇪🇸 **Cádiz Spanish** (Gaudita)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Acentos API Server

In a **separate terminal**, run:

```bash
python acentos_server.py
```

You should see:
```
🚀 Starting Acentos API Server...
📍 Server running at: http://localhost:5000
```

Keep this terminal running!

### 3. Start the Lyric Editor

In another terminal, run:

```bash
python app.py
```

## Usage

### Quick IPA Transcription

1. **Upload and transcribe** your Spanish audio (mp3)
2. The lyrics will appear in the editor
3. **Click one of the dialect buttons**:
   - 🇩🇴 Dominican
   - 🇲🇽 Mexican
   - 🇦🇷 Rioplatense
   - 🇪🇸 Cádiz
4. The acentos API will transcribe your lyrics to IPA!
5. The IPA transcription will update in the editor

### Status Indicator

The "Acentos API Status" field shows:
- ❌ If the acentos server is not running
- ✅ If transcription succeeded

## How It Works

```
Lyric Editor  ──HTTP──>  Acentos Server  ──JS──>  Acentos JavaScript
    (app.py)             (port 5000)              (your GitHub repo)
```

1. You click a dialect button
2. The lyric editor sends Spanish text to `http://localhost:5000/transcribe`
3. The acentos server processes it using your JavaScript code
4. IPA transcription is returned and displayed

## Troubleshooting

### "Cannot connect to acentos server"

**Problem**: The acentos_server.py is not running.

**Solution**:
```bash
python acentos_server.py
```

### "Port 5000 already in use"

**Problem**: Another app is using port 5000.

**Solution**: Change the port in both files:
- `acentos_server.py`: Change `port=5000` to `port=5001`
- `app.py`: Change `ACENTOS_API_URL = "http://localhost:5000"` to `5001`

## API Endpoints

The acentos server provides:

```
GET  /health          - Health check
GET  /dialects        - List available dialects
POST /transcribe      - Transcribe Spanish to IPA
```

### Example API Call

```bash
curl -X POST http://localhost:5000/transcribe \
  -H "Content-Type: application/json" \
  -d '{"text": "hola mundo", "dialect": "dominican"}'
```

Response:
```json
{
  "original": "hola mundo",
  "ipa": "[ˈo.la ˈmun.do]",
  "dialect": "dominican"
}
```

## Future Enhancements

- [ ] Real-time IPA preview as you type
- [ ] Word-by-word IPA replacement in editor
- [ ] Batch processing for long texts
- [ ] Cache transcriptions to avoid re-processing
- [ ] Integration with your GitHub acentos repo updates

## Maintaining Only One Codebase

The beauty of this setup is that **you only need to maintain your acentos repository**!

When you update acentos on GitHub:
1. The acentos_server.py will use the latest version
2. This lyric editor automatically benefits from improvements
3. No need to update code in two places!

---

For questions or issues, see: https://github.com/Francuscus/acentos
