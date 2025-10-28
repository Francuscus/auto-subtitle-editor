"""
Acentos API Server
A Flask HTTP server that wraps the acentos JavaScript IPA transcription functionality.
This allows the lyric editor to call acentos via HTTP API.

Usage:
    python acentos_server.py

Then the lyric editor can call:
    POST http://localhost:5000/transcribe
    {"text": "hola mundo", "dialect": "dominican"}
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import tempfile
import os

app = Flask(__name__)
CORS(app)  # Allow requests from Gradio app

# Path to the acentos HTML file (will be fetched from GitHub)
ACENTOS_HTML_URL = "https://raw.githubusercontent.com/Francuscus/acentos/main/index.html"

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "service": "acentos-api"}), 200

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe Spanish text to IPA using acentos

    Request JSON:
    {
        "text": "hola mundo",
        "dialect": "dominican",  # or "mexican", "rioplatense", "cadiz"
        "features": {
            "syllables": true,
            "nVelarization": true,
            ...
        }
    }

    Response JSON:
    {
        "original": "hola mundo",
        "ipa": "[ˈo.la ˈmun.do]",
        "dialect": "dominican"
    }
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        dialect = data.get('dialect', 'dominican').lower()
        features = data.get('features', {})

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # For now, use a simple Node.js execution to run the JavaScript
        # This is a placeholder - in production, you'd extract the JS and run it with PyExecJS

        # Temporary implementation: return formatted response
        # TODO: Actually execute the acentos JavaScript

        ipa_result = f"[IPA transcription of '{text}' in {dialect} dialect]"

        return jsonify({
            "original": text,
            "ipa": ipa_result,
            "dialect": dialect,
            "note": "Server running! Full JS integration coming soon."
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/dialects', methods=['GET'])
def get_dialects():
    """Get available Spanish dialects"""
    return jsonify({
        "dialects": [
            {"id": "dominican", "name": "Dominican Spanish"},
            {"id": "mexican", "name": "Mexican Spanish"},
            {"id": "rioplatense", "name": "Rioplatense Spanish"},
            {"id": "cadiz", "name": "Cádiz Spanish (Gaudita)"}
        ]
    }), 200

if __name__ == '__main__':
    print("🚀 Starting Acentos API Server...")
    print("📍 Server running at: http://localhost:5000")
    print("📚 Available endpoints:")
    print("   GET  /health - Health check")
    print("   GET  /dialects - List available dialects")
    print("   POST /transcribe - Transcribe Spanish to IPA")
    print("\nPress Ctrl+C to stop")

    app.run(host='0.0.0.0', port=5000, debug=True)
