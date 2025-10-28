"""
Acentos API Server
A Flask HTTP server that provides IPA transcription for Spanish text.
Converted from the JavaScript implementation at https://github.com/Francuscus/acentos

Usage:
    python acentos_server.py

Then the lyric editor can call:
    POST http://localhost:5000/transcribe
    {"text": "hola mundo", "dialect": "dominican"}
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import unicodedata

app = Flask(__name__)
CORS(app)  # Allow requests from Gradio app

# Constants
VOWELS = 'aeiou'
WEAK_VOWELS = 'iu'
CONSONANTS = 'bβdðgɣptkfθsxmnɲlrɾʝtʃɲjʃʒRθjwŋ'
VALID_CLUSTERS = ['bl', 'br', 'kl', 'kr', 'dl', 'dr', 'fl', 'fr', 'gl', 'gr', 'pl', 'pr', 'tl', 'tr']

def orthography_to_phonemes(text, dialect):
    """Convert Spanish orthography to phonemes"""
    result = (text or '').lower()

    # Basic replacements
    result = result.replace('ch', 'tʃ')
    result = result.replace('h', '')
    result = result.replace('ll', 'ʝ')
    result = result.replace('rr', 'r')
    result = result.replace('qu', 'k')

    # Ceceo distinction for Cádiz
    if dialect == 'cadiz':
        result = re.sub(r'c([ei])', r'θ\1', result)
        result = result.replace('z', 'θ')
    else:
        result = re.sub(r'c([ei])', r's\1', result)
        result = result.replace('z', 's')

    result = result.replace('c', 'k')

    # Velar fricative
    result = re.sub(r'g([ei])', r'x\1', result)
    result = re.sub(r'gu([ei])', r'g\1', result)

    # Other replacements
    result = result.replace('v', 'b')
    result = re.sub(r'y(?=[aeiou])', 'ʝ', result)
    result = re.sub(r'y$', 'i', result)
    result = re.sub(r'([aeiou])y', r'\1i', result)
    result = re.sub(r'^y', 'ʝ', result)
    result = result.replace('j', 'x')
    result = result.replace('ñ', 'ɲ')

    # Handle R (word-final trill)
    result = re.sub(r'r$', 'R', result)
    result = result.replace('r', 'ɾ')

    return result

def syllabify(phonemes, original_word):
    """Break phonemes into syllables following Spanish phonotactics"""
    strong_vowels = 'aeo'
    vowels_set = set(VOWELS)
    weak_set = set(WEAK_VOWELS)

    # Find stressed weak vowels (í, ú)
    stressed_weak_vowels = set()
    if original_word:
        vowel_index = 0
        for i, ch in enumerate(original_word):
            base = unicodedata.normalize('NFD', ch).encode('ascii', 'ignore').decode().lower()
            if ch in ['í', 'Í', 'ú', 'Ú']:
                stressed_weak_vowels.add(vowel_index)
            if base in vowels_set:
                vowel_index += 1

    # Process diphthongs
    processed = ''
    v_idx = 0
    i = 0
    while i < len(phonemes):
        a = phonemes[i]
        b = phonemes[i + 1] if i + 1 < len(phonemes) else None

        a_is_v = a in vowels_set
        b_is_v = b in vowels_set if b else False

        if a_is_v and b_is_v:
            a_weak = a in weak_set
            b_weak = b in weak_set
            a_stressed_weak = a_weak and v_idx in stressed_weak_vowels
            b_stressed_weak = b_weak and (v_idx + 1) in stressed_weak_vowels

            is_dip = (not a_stressed_weak and not b_stressed_weak) and (
                (a_weak and not b_weak) or (not a_weak and b_weak) or (a_weak and b_weak)
            )

            if is_dip:
                if a_weak:
                    processed += ('j' if a == 'i' else 'w') + b
                elif b_weak:
                    processed += a + ('j' if b == 'i' else 'w')
                i += 2
                v_idx += 2
                continue
            else:
                processed += a
                v_idx += 1
                i += 1
                continue

        processed += a
        if a_is_v:
            v_idx += 1
        i += 1

    # Split into syllables
    syllables = []
    cur = ''
    i = 0

    while i < len(processed):
        ch = processed[i]
        cur += ch

        if ch in VOWELS:
            j = i + 1
            cons = ''
            while j < len(processed) and processed[j] not in VOWELS:
                cons += processed[j]
                j += 1

            if len(cons) == 0:
                syllables.append(cur)
                cur = ''
                i += 1
                continue

            if len(cons) == 1:
                syllables.append(cur)
                cur = cons
                i += 2
                continue

            # Check for valid clusters
            last_two = cons[-2:] if len(cons) >= 2 else ''
            if last_two in VALID_CLUSTERS:
                coda = cons[:-2]
                cur += coda
                syllables.append(cur)
                cur = last_two
                i = j
            else:
                coda = cons[:-1]
                cur += coda
                syllables.append(cur)
                cur = cons[-1]
                i = j
            continue

        i += 1

    if cur:
        syllables.append(cur)

    return syllables

def apply_stress(syllables, original_word):
    """Mark primary stress with ˈ prefix"""
    if not syllables:
        return syllables

    # Check for acute accent
    acute_match = re.search(r'[áéíóúÁÉÍÓÚ]', original_word or '')
    if acute_match:
        accent_map = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
                     'Á': 'a', 'É': 'e', 'Í': 'i', 'Ó': 'o', 'Ú': 'u'}
        v = accent_map[acute_match.group(0)]

        # Find syllable with this vowel (search from end)
        for i in range(len(syllables) - 1, -1, -1):
            if v in syllables[i]:
                syllables[i] = 'ˈ' + syllables[i]
                return syllables

    # Default stress rules
    last_char = original_word[-1] if original_word else ''
    ends_in_vowel_ns = bool(re.search(r'[aeiouns]$', last_char, re.IGNORECASE))

    if len(syllables) == 1:
        idx = 0
    else:
        idx = len(syllables) - 2 if ends_in_vowel_ns else len(syllables) - 1

    if idx >= 0:
        syllables[idx] = 'ˈ' + syllables[idx]

    return syllables

def apply_rules(syllables, dialect, features):
    """Apply dialect-specific phonological rules"""
    result_syllables = []

    for index, syllable in enumerate(syllables):
        result = syllable
        had_stress = 'ˈ' in result
        if had_stress:
            result = result.replace('ˈ', '')

        is_last = index == len(syllables) - 1

        # Cádiz rules
        if dialect == 'cadiz' and features.get('liquidSubCadiz', True):
            # Replace r/ɾ/R in syllable-final position
            result = re.sub(r'[ɾrR]$', 'l̪', result)
            result = re.sub(r'[ɾrR](?=[bβdðgɣptkfθsxmnɲʝtʃjʃʒRθjwŋ])', 'l̪', result)

        if dialect == 'cadiz' and features.get('finalRDeletionCadiz', True):
            if is_last:
                result = re.sub(r'[ɾrR]$', '', result)

        if dialect == 'cadiz' and features.get('finalSDeletion', True):
            if is_last:
                result = re.sub(r'[sθ]$', 'h', result)
            result = re.sub(r'[sθ](?=[bβdðgɣptkfxmnɲʝtʃjʃʒRθ])', 'h', result)

        # Rioplatense rules
        if dialect == 'rioplatense':
            if is_last and 'R' in result:
                result = result.replace('R', 'r')
            else:
                result = result.replace('R', 'ɾ')
        else:
            result = result.replace('R', 'ɾ')

        if dialect == 'rioplatense' and features.get('sheismo', True):
            result = result.replace('ʝ', 'ʃ')

        if dialect == 'rioplatense' and features.get('sAspiration', True):
            if is_last:
                result = re.sub(r's$', 'h', result)
            result = re.sub(r's(?=[bβdðgɣptkfθxmnɲʝtʃjʃʒRθ])', 'h', result)

        if dialect == 'rioplatense' and features.get('finalRDeletion', True):
            if is_last:
                result = re.sub(r'ɾ$', '', result)

        # Dominican rules
        if dialect == 'dominican' and features.get('nVelarization', True):
            if is_last:
                result = re.sub(r'n$', 'ŋ', result)

        if dialect == 'dominican' and features.get('liquidSub', True):
            result = re.sub(r'[ɾrR]$', 'l̪', result)
            result = re.sub(r'[ɾrR](?=[bβdðgɣptkfθsxmnɲʝtʃjʃʒRθjwŋl̪])', 'l̪', result)

        if dialect == 'dominican' and features.get('sDelete', True):
            result = re.sub(r's$', '', result)
            result = re.sub(r's(?=[ptkg])', '', result)

        # Mexican rules
        if dialect == 'mexican' and features.get('assibilatedR', True):
            if not re.match(r'^[ɾrR]', result):
                result = result.replace('ɾ', 'ʂ').replace('r', 'ʂ').replace('R', 'ʂ')

        # Universal allophonic rules
        result = re.sub(r'b(?![mnɲ])', 'β', result)
        result = re.sub(r'([mnɲ])β', r'\1b', result)
        result = result.replace('d', 'ð')
        result = re.sub(r'([mnɲl̪])ð', r'\1d', result)
        result = re.sub(r'g(?![mnɲ])', 'ɣ', result)
        result = re.sub(r'([mnɲ])ɣ', r'\1g', result)
        result = re.sub(r'l(?!̪)', 'l̪', result)

        if had_stress:
            result = 'ˈ' + result

        result_syllables.append(result)

    return result_syllables

def transcribe_text(text, dialect='dominican', features=None):
    """Main transcription function"""
    if features is None:
        # Default features (all enabled)
        features = {
            'liquidSubCadiz': True,
            'nVelarization': True,
            'ceceo': True,
            'finalRDeletionCadiz': True,
            'finalSDeletion': True,
            'liquidSub': True,
            'sDelete': True,
            'assibilatedR': True,
            'sheismo': True,
            'sAspiration': True,
            'finalRDeletion': True,
            'showSyllables': True
        }

    # Clean and split text
    cleaned = re.sub(r'[.,;:!?¡¿()«»"""\'\'\\[\\]{}]', ' ', text.lower())
    words = [w.strip() for w in cleaned.split() if w.strip()]

    results = []

    for word in words:
        phonemes = orthography_to_phonemes(word, dialect)
        syllables = syllabify(phonemes, word)
        syllables = apply_stress(syllables, word)

        # Dominican initial r→ɣ rule
        if dialect == 'dominican' and syllables:
            first_syl = syllables[0]
            if re.match(r'^ˈ?[ɾrR]', first_syl):
                if first_syl.startswith('ˈɾ') or first_syl.startswith('ˈr') or first_syl.startswith('ˈR'):
                    syllables[0] = 'ˈɣ' + first_syl[2:]
                elif first_syl.startswith('ɾ') or first_syl.startswith('r') or first_syl.startswith('R'):
                    syllables[0] = 'ɣ' + first_syl[1:]

        syllables = apply_rules(syllables, dialect, features)

        if features.get('showSyllables', True):
            results.append('.'.join(syllables))
        else:
            results.append(''.join(syllables))

    return f"[{' '.join(results)}]"

# Flask endpoints
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "service": "acentos-api"}), 200

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe Spanish text to IPA using acentos logic

    Request JSON:
    {
        "text": "hola mundo",
        "dialect": "dominican",  # or "mexican", "rioplatense", "cadiz"
        "features": {
            "showSyllables": true,
            "nVelarization": true,
            ...
        }
    }

    Response JSON:
    {
        "original": "hola mundo",
        "ipa": "[ˈo.la ˈmun.ðo]",
        "dialect": "dominican"
    }
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        dialect = data.get('dialect', 'dominican').lower()
        features = data.get('features', None)

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Transcribe using our Python implementation
        ipa_result = transcribe_text(text, dialect, features)

        return jsonify({
            "original": text,
            "ipa": ipa_result,
            "dialect": dialect
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
            {"id": "cadiz", "name": "Cádiz Spanish"}
        ]
    }), 200

if __name__ == '__main__':
    print("🚀 Starting Acentos API Server...")
    print("📍 Server running at: http://localhost:5000")
    print("📚 Available endpoints:")
    print("   GET  /health - Health check")
    print("   GET  /dialects - List available dialects")
    print("   POST /transcribe - Transcribe Spanish to IPA")
    print("\n✨ Full JavaScript logic now implemented in Python!")
    print("\nPress Ctrl+C to stop")

    app.run(host='0.0.0.0', port=5000, debug=True)
