# Language Learning Subtitle Editor
# Version 3.2 — Verb Detection Gates (No More False Positives!)
# Banner Color: #E67E22 (Carrot Orange)

import os
import re
import tempfile
from typing import List, Tuple, Optional
import zipfile
import pathlib
import subprocess
import shutil

import gradio as gr
import torch
import whisperx

# -------------------------- Config --------------------------

VERSION = "4.1"
BANNER_COLOR = "#1E88E5"  # Blue banner (v4.1 - verb ending colorizer)
DEFAULT_SAMPLE_TEXT_COLOR = "#1e88e5"  # Blue so it isn't white-on-white

LANG_MAP = {
    "auto": None, "auto-detect": None, "automatic": None,
    "hungarian": "hu", "magyar": "hu", "hun": "hu", "hu": "hu",
    "spanish": "es", "español": "es", "esp": "es", "es": "es",
    "english": "en", "eng": "en", "en": "en",
}

# -------------------------- Utilities --------------------------

def normalize_lang(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = s.strip().lower()
    if t in LANG_MAP:
        return LANG_MAP[t]
    m = re.search(r"\b([a-z]{2,3})\b", t)
    return m.group(1) if m else None

def seconds_to_timestamp_srt(t: float) -> str:
    t = max(t, 0.0)
    h = int(t // 3600); t -= h * 3600
    m = int(t // 60);   t -= m * 60
    s = int(t);         ms = int(round((t - s) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def seconds_to_timestamp_ass(t: float) -> str:
    t = max(t, 0.0)
    h = int(t // 3600); t -= h * 3600
    m = int(t // 60);   t -= m * 60
    s = int(t);         cs = int(round((t - s) * 100))
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

# -------------------------- IPA Transcription (Acentos) --------------------------
# Integrated from https://github.com/Francuscus/acentos

import unicodedata

# IPA Constants
IPA_VOWELS = 'aeiou'
IPA_WEAK_VOWELS = 'iu'
IPA_VALID_CLUSTERS = ['bl', 'br', 'kl', 'kr', 'dl', 'dr', 'fl', 'fr', 'gl', 'gr', 'pl', 'pr', 'tl', 'tr']

def orthography_to_phonemes(text, dialect):
    """Convert Spanish orthography to phonemes"""
    result = (text or '').lower()
    result = result.replace('ch', 'tʃ')
    result = result.replace('h', '')
    result = result.replace('ll', 'ʝ')
    result = result.replace('rr', 'r')
    result = result.replace('qu', 'k')

    if dialect == 'cadiz':
        result = re.sub(r'c([ei])', r'θ\1', result)
        result = result.replace('z', 'θ')
    else:
        result = re.sub(r'c([ei])', r's\1', result)
        result = result.replace('z', 's')

    result = result.replace('c', 'k')
    result = re.sub(r'g([ei])', r'x\1', result)
    result = re.sub(r'gu([ei])', r'g\1', result)
    result = result.replace('v', 'b')
    result = re.sub(r'y(?=[aeiou])', 'ʝ', result)
    result = re.sub(r'y$', 'i', result)
    result = re.sub(r'([aeiou])y', r'\1i', result)
    result = re.sub(r'^y', 'ʝ', result)
    result = result.replace('j', 'x')
    result = result.replace('ñ', 'ɲ')
    result = re.sub(r'r$', 'R', result)
    result = result.replace('r', 'ɾ')

    return result

def syllabify(phonemes, original_word):
    """Break phonemes into syllables following Spanish phonotactics"""
    vowels_set = set(IPA_VOWELS)
    weak_set = set(IPA_WEAK_VOWELS)

    stressed_weak_vowels = set()
    if original_word:
        vowel_index = 0
        for i, ch in enumerate(original_word):
            base = unicodedata.normalize('NFD', ch).encode('ascii', 'ignore').decode().lower()
            if ch in ['í', 'Í', 'ú', 'Ú']:
                stressed_weak_vowels.add(vowel_index)
            if base in vowels_set:
                vowel_index += 1

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

    syllables = []
    cur = ''
    i = 0

    while i < len(processed):
        ch = processed[i]
        cur += ch

        if ch in IPA_VOWELS:
            j = i + 1
            cons = ''
            while j < len(processed) and processed[j] not in IPA_VOWELS:
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

            last_two = cons[-2:] if len(cons) >= 2 else ''
            if last_two in IPA_VALID_CLUSTERS:
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

    acute_match = re.search(r'[áéíóúÁÉÍÓÚ]', original_word or '')
    if acute_match:
        accent_map = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
                     'Á': 'a', 'É': 'e', 'Í': 'i', 'Ó': 'o', 'Ú': 'u'}
        v = accent_map[acute_match.group(0)]

        for i in range(len(syllables) - 1, -1, -1):
            if v in syllables[i]:
                syllables[i] = 'ˈ' + syllables[i]
                return syllables

    last_char = original_word[-1] if original_word else ''
    ends_in_vowel_ns = bool(re.search(r'[aeiouns]$', last_char, re.IGNORECASE))

    if len(syllables) == 1:
        idx = 0
    else:
        idx = len(syllables) - 2 if ends_in_vowel_ns else len(syllables) - 1

    if idx >= 0:
        syllables[idx] = 'ˈ' + syllables[idx]

    return syllables

def apply_dialect_rules(syllables, dialect, features):
    """Apply dialect-specific phonological rules"""
    result_syllables = []

    for index, syllable in enumerate(syllables):
        result = syllable
        had_stress = 'ˈ' in result
        if had_stress:
            result = result.replace('ˈ', '')

        is_last = index == len(syllables) - 1

        if dialect == 'cadiz' and features.get('liquidSubCadiz', True):
            result = re.sub(r'[ɾrR]$', 'l̪', result)
            result = re.sub(r'[ɾrR](?=[bβdðgɣptkfθsxmnɲʝtʃjʃʒRθjwŋ])', 'l̪', result)

        if dialect == 'cadiz' and features.get('finalRDeletionCadiz', True):
            if is_last:
                result = re.sub(r'[ɾrR]$', '', result)

        if dialect == 'cadiz' and features.get('finalSDeletion', True):
            if is_last:
                result = re.sub(r'[sθ]$', 'h', result)
            result = re.sub(r'[sθ](?=[bβdðgɣptkfxmnɲʝtʃjʃʒRθ])', 'h', result)

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

        if dialect == 'dominican' and features.get('nVelarization', True):
            if is_last:
                result = re.sub(r'n$', 'ŋ', result)

        if dialect == 'dominican' and features.get('liquidSub', True):
            result = re.sub(r'[ɾrR]$', 'l̪', result)
            result = re.sub(r'[ɾrR](?=[bβdðgɣptkfθsxmnɲʝtʃjʃʒRθjwŋl̪])', 'l̪', result)

        if dialect == 'dominican' and features.get('sDelete', True):
            result = re.sub(r's$', '', result)
            result = re.sub(r's(?=[ptkg])', '', result)

        if dialect == 'mexican' and features.get('assibilatedR', True):
            if not re.match(r'^[ɾrR]', result):
                result = result.replace('ɾ', 'ʂ').replace('r', 'ʂ').replace('R', 'ʂ')

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

def transcribe_to_ipa_internal(text, dialect='dominican'):
    """
    Main IPA transcription function
    Converts Spanish text to IPA notation using dialect-specific rules
    """
    features = {
        'liquidSubCadiz': True, 'nVelarization': True, 'ceceo': True,
        'finalRDeletionCadiz': True, 'finalSDeletion': True, 'liquidSub': True,
        'sDelete': True, 'assibilatedR': True, 'sheismo': True,
        'sAspiration': True, 'finalRDeletion': True, 'showSyllables': True
    }

    cleaned = re.sub(r'[.,;:!?¡¿()«»"""\'\'\\[\\]{}]', ' ', text.lower())
    words = [w.strip() for w in cleaned.split() if w.strip()]

    results = []

    for word in words:
        phonemes = orthography_to_phonemes(word, dialect)
        syllables = syllabify(phonemes, word)
        syllables = apply_stress(syllables, word)

        if dialect == 'dominican' and syllables:
            first_syl = syllables[0]
            if re.match(r'^ˈ?[ɾrR]', first_syl):
                if first_syl.startswith('ˈɾ') or first_syl.startswith('ˈr') or first_syl.startswith('ˈR'):
                    syllables[0] = 'ˈɣ' + first_syl[2:]
                elif first_syl.startswith('ɾ') or first_syl.startswith('r') or first_syl.startswith('R'):
                    syllables[0] = 'ɣ' + first_syl[1:]

        syllables = apply_dialect_rules(syllables, dialect, features)

        if features.get('showSyllables', True):
            results.append('.'.join(syllables))
        else:
            results.append(''.join(syllables))

    return f"[{' '.join(results)}]"

# -------------------------- Spanish Verb Coloring (Paco Grammar) --------------------------
# Automatically colors person-marking endings in Spanish verbs
# Uses linguistic gates to distinguish verbs from non-verbs

# Person marker colors (matching the prompt requirements)
PERSON_COLORS = {
    'yo': '#FF0000',           # 🔴 red
    'tu': '#0000FF',           # 🔵 blue
    'el': '#FFD700',           # 🟡 gold (darker yellow)
    'nosotros': '#9C27B0',     # 🟣 purple
    'ellos': '#00FF00',        # 🟢 green
    'shared': '#FFA500'        # 🟠 orange (yo = él/ella/usted)
}

# Disqualifiers - words that indicate the following word is NOT a verb
ARTICLES = {'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas'}
DETERMINERS = {'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas',
               'aquel', 'aquella', 'aquellos', 'aquellas', 'mi', 'tu', 'su',
               'mis', 'tus', 'sus', 'nuestro', 'nuestra', 'nuestros', 'nuestras',
               'vuestro', 'vuestra', 'vuestros', 'vuestras', 'lo', 'al', 'del'}
PREPOSITIONS = {'de', 'a', 'en', 'con', 'por', 'para', 'sin', 'sobre', 'tras',
                'desde', 'hasta', 'hacia', 'entre', 'durante', 'mediante'}

# Context clues that boost verb hypothesis
SUBJECT_PRONOUNS = {'yo', 'tú', 'él', 'ella', 'usted', 'nosotros', 'nosotras',
                    'vosotros', 'vosotras', 'ellos', 'ellas', 'ustedes'}
NEGATIONS = {'no', 'nunca', 'jamás', 'tampoco'}
QUESTION_WORDS = {'qué', 'quién', 'quiénes', 'cómo', 'cuándo', 'dónde',
                  'cuál', 'cuáles', 'cuánto', 'cuánta', 'cuántos', 'cuántas'}

# Common auxiliary verbs (always considered verbs when conjugated)
AUXILIARIES = {
    # estar
    'estoy', 'estás', 'está', 'estamos', 'estáis', 'están',
    # ir
    'voy', 'vas', 'va', 'vamos', 'vais', 'van',
    # haber
    'he', 'has', 'ha', 'hemos', 'habéis', 'han',
    'había', 'habías', 'habíamos', 'habíais', 'habían',
    'hube', 'hubiste', 'hubo', 'hubimos', 'hubisteis', 'hubieron',
    'habré', 'habrás', 'habrá', 'habremos', 'habréis', 'habrán',
    # ser
    'soy', 'eres', 'es', 'somos', 'sois', 'son',
    'era', 'eras', 'éramos', 'erais', 'eran',
    'fui', 'fuiste', 'fue', 'fuimos', 'fuisteis', 'fueron'
}


def passes_verb_gate(word, prev_word=None, next_word=None, is_first=False):
    """
    Phase 1: Verb Detection Gate
    Returns True only if word passes as a potential verb (not a noun/adjective/etc.)
    """
    word_lower = word.lower()
    prev_lower = prev_word.lower() if prev_word else None

    # DISQUALIFIERS (fail immediately)
    # 1. Preceded by article/determiner → likely noun
    if prev_lower and (prev_lower in ARTICLES or prev_lower in DETERMINERS):
        return False

    # 2. Ends with -mente → adverb
    if word_lower.endswith('mente'):
        return False

    # 3. Infinitives -ar/-er/-ir (don't color them)
    if word_lower.endswith(('ar', 'er', 'ir')) and len(word_lower) > 3:
        return False

    # AUXILIARIES (always pass)
    if word_lower in AUXILIARIES:
        return True

    # CONJUGATED FORM PATTERNS
    # Must match a known conjugation pattern
    has_verb_ending = False

    # Preterite: -é, -aste, -ó, -aron, -ieron
    if word_lower.endswith(('é', 'ó', 'aste', 'aron', 'iste', 'ió', 'ieron')):
        has_verb_ending = True

    # Imperfect: -aba, -abas, -ábamos, -aban, -ía, -ías, -íamos, -ían
    elif word_lower.endswith(('aba', 'abas', 'ábamos', 'aban', 'ía', 'ías', 'íamos', 'ían')):
        has_verb_ending = True

    # Future/Conditional: -rás, -rá, -remos, -rán, -ría
    elif word_lower.endswith(('rás', 'rá', 'remos', 'rán', 'ría', 'rías', 'ríamos', 'rían')):
        has_verb_ending = True

    # Present: -mos, -s, -n, -o, -a, -e (but be careful with these)
    elif word_lower.endswith(('amos', 'emos', 'imos')) and len(word_lower) >= 5:
        has_verb_ending = True
    elif word_lower.endswith(('as', 'es')) and len(word_lower) >= 4:
        has_verb_ending = True
    elif word_lower.endswith(('an', 'en')) and len(word_lower) >= 4:
        has_verb_ending = True
    elif word_lower.endswith('o') and len(word_lower) >= 4:
        # "hablo", "como" YES, but "yo", "no", "como" (noun) need context
        has_verb_ending = True
    elif word_lower.endswith(('a', 'e')) and len(word_lower) >= 4:
        # "habla", "come" YES, but "casa", "clase" need checking
        # Only pass if we have positive context
        has_verb_ending = False  # Will check context below

    if not has_verb_ending:
        return False

    # SYNTACTIC CONTEXT (boost confidence)
    # Following subject pronoun → definitely verb
    if prev_lower and prev_lower in SUBJECT_PRONOUNS:
        return True

    # Following negation → definitely verb
    if prev_lower and prev_lower in NEGATIONS:
        return True

    # First word in sentence with strong verb ending → likely verb
    if is_first and word_lower.endswith(('é', 'ó', 'aba', 'ía', 'amos', 'emos')):
        return True

    # If we got here, it matched a verb ending pattern
    return True

def identify_verb_ending_gated(word):
    """
    Returns (stem, ending, person, color) for a verified verb
    Only called AFTER word passes verb gate
    """
    word_lower = word.lower()

    # Haber: only color minimal person markers
    if word_lower in ['he', 'has', 'ha', 'hemos', 'han']:
        if word_lower == 'he':
            return ('h', 'e', 'yo', PERSON_COLORS['yo'])
        elif word_lower == 'has':
            return ('ha', 's', 'tu', PERSON_COLORS['tu'])  # only "s"
        elif word_lower == 'ha':
            return ('h', 'a', 'el', PERSON_COLORS['el'])
        elif word_lower == 'hemos':
            return ('he', 'mos', 'nosotros', PERSON_COLORS['nosotros'])  # only "mos"
        elif word_lower == 'han':
            return ('ha', 'n', 'ellos', PERSON_COLORS['ellos'])  # only "n"

    # Estar: only color minimal person markers
    if word_lower in ['estoy', 'estás', 'está', 'estamos', 'están']:
        if word_lower == 'estoy':
            return ('est', 'oy', 'yo', PERSON_COLORS['yo'])
        elif word_lower == 'estás':
            return ('está', 's', 'tu', PERSON_COLORS['tu'])  # only "s"
        elif word_lower == 'está':
            return ('est', 'á', 'el', PERSON_COLORS['el'])  # only "á"
        elif word_lower == 'estamos':
            return ('esta', 'mos', 'nosotros', PERSON_COLORS['nosotros'])  # only "mos"
        elif word_lower == 'están':
            return ('está', 'n', 'ellos', PERSON_COLORS['ellos'])  # only "n"

    # Ir: only color minimal person markers
    if word_lower in ['voy', 'vas', 'va', 'vamos', 'van']:
        if word_lower == 'voy':
            return ('v', 'oy', 'yo', PERSON_COLORS['yo'])
        elif word_lower == 'vas':
            return ('va', 's', 'tu', PERSON_COLORS['tu'])  # only "s"
        elif word_lower == 'va':
            return ('v', 'a', 'el', PERSON_COLORS['el'])  # only "a"
        elif word_lower == 'vamos':
            return ('va', 'mos', 'nosotros', PERSON_COLORS['nosotros'])  # only "mos"
        elif word_lower == 'van':
            return ('va', 'n', 'ellos', PERSON_COLORS['ellos'])  # only "n"

    # Imperfect -aba (shared yo/él)
    if word_lower.endswith('ábamos'):
        return (word[:-6], word[-6:], 'nosotros', PERSON_COLORS['nosotros'])
    elif word_lower.endswith('aban'):
        return (word[:-4], word[-4:], 'ellos', PERSON_COLORS['ellos'])
    elif word_lower.endswith('abas'):
        return (word[:-4], word[-4:], 'tu', PERSON_COLORS['tu'])
    elif word_lower.endswith('aba'):
        return (word[:-3], word[-3:], 'shared', PERSON_COLORS['shared'])

    # Imperfect -ía (shared yo/él)
    if word_lower.endswith('íamos'):
        return (word[:-5], word[-5:], 'nosotros', PERSON_COLORS['nosotros'])
    elif word_lower.endswith('ían'):
        return (word[:-3], word[-3:], 'ellos', PERSON_COLORS['ellos'])
    elif word_lower.endswith('ías'):
        return (word[:-3], word[-3:], 'tu', PERSON_COLORS['tu'])
    elif word_lower.endswith('ía'):
        return (word[:-2], word[-2:], 'shared', PERSON_COLORS['shared'])

    # Preterite
    if word_lower.endswith('aron'):
        return (word[:-4], word[-4:], 'ellos', PERSON_COLORS['ellos'])
    elif word_lower.endswith('ieron'):
        return (word[:-5], word[-5:], 'ellos', PERSON_COLORS['ellos'])
    elif word_lower.endswith('aste'):
        return (word[:-4], word[-4:], 'tu', PERSON_COLORS['tu'])
    elif word_lower.endswith('iste'):
        return (word[:-4], word[-4:], 'tu', PERSON_COLORS['tu'])
    elif word_lower.endswith('é'):
        return (word[:-1], word[-1:], 'yo', PERSON_COLORS['yo'])
    elif word_lower.endswith('ó'):
        return (word[:-1], word[-1:], 'el', PERSON_COLORS['el'])
    elif word_lower.endswith('ió'):
        return (word[:-2], word[-2:], 'el', PERSON_COLORS['el'])

    # Present: -mos (only color "mos", not full ending)
    if word_lower.endswith('amos'):
        return (word[:-3], word[-3:], 'nosotros', PERSON_COLORS['nosotros'])  # only "mos"
    elif word_lower.endswith('emos'):
        return (word[:-3], word[-3:], 'nosotros', PERSON_COLORS['nosotros'])  # only "mos"
    elif word_lower.endswith('imos'):
        return (word[:-3], word[-3:], 'nosotros', PERSON_COLORS['nosotros'])  # only "mos"

    # Present: -s (tú) - only color "s"
    if word_lower.endswith('as') and not word_lower.endswith('abas'):
        return (word[:-1], word[-1:], 'tu', PERSON_COLORS['tu'])  # only "s"
    elif word_lower.endswith('es') and not word_lower.endswith(('eres', 'iones')):
        return (word[:-1], word[-1:], 'tu', PERSON_COLORS['tu'])  # only "s"
    elif word_lower.endswith('s') and len(word_lower) >= 4:
        return (word[:-1], word[-1:], 'tu', PERSON_COLORS['tu'])

    # Present: -n (ellos) - only color "n"
    if word_lower.endswith('an') and not word_lower.endswith(('aban', 'rán')):
        return (word[:-1], word[-1:], 'ellos', PERSON_COLORS['ellos'])  # only "n"
    elif word_lower.endswith('en') and not word_lower.endswith('ieren'):
        return (word[:-1], word[-1:], 'ellos', PERSON_COLORS['ellos'])  # only "n"
    elif word_lower.endswith('n') and len(word_lower) >= 4:
        return (word[:-1], word[-1:], 'ellos', PERSON_COLORS['ellos'])

    # Present: -o (yo)
    if word_lower.endswith('o') and len(word_lower) >= 4:
        return (word[:-1], word[-1:], 'yo', PERSON_COLORS['yo'])

    # Present: -a, -e (él/ella) - thematic vowel
    if word_lower.endswith('a') and len(word_lower) >= 4:
        return (word[:-1], word[-1:], 'el', PERSON_COLORS['el'])
    elif word_lower.endswith('e') and len(word_lower) >= 4:
        return (word[:-1], word[-1:], 'el', PERSON_COLORS['el'])

    return None

def apply_spanish_verb_coloring(words):
    """
    Apply verb coloring using linguistic gates (Paco's Grammar)
    Only colors actual verbs, not nouns. Only colors endings, not whole words.
    """
    colored_words = []

    for i, word_obj in enumerate(words):
        text = word_obj.get('text', '').strip()

        # Get context
        prev_word = words[i-1]['text'].strip() if i > 0 else None
        next_word = words[i+1]['text'].strip() if i < len(words) - 1 else None
        is_first = (i == 0)

        # Phase 1: Verb Gate (filter out non-verbs)
        if not passes_verb_gate(text, prev_word, next_word, is_first):
            # Not a verb - keep original
            colored_words.append(word_obj.copy())
            continue

        # Phase 2: Identify ending (only for verbs)
        result = identify_verb_ending_gated(text)

        if result:
            stem, ending, person, color = result

            # Create word with character-level coloring info
            colored_word = word_obj.copy()
            colored_word['is_verb'] = True
            colored_word['stem'] = stem
            colored_word['ending'] = ending
            colored_word['ending_color'] = color
            colored_word['person'] = person
            # Create HTML with stem (default color) + ending (person color)
            colored_word['html'] = f'<span style="color: {DEFAULT_SAMPLE_TEXT_COLOR};">{stem}</span><span style="color: {color};">{ending}</span>'
            colored_word['color'] = color  # Keep for backward compatibility
            colored_words.append(colored_word)
        else:
            # Passed gate but no ending match - keep original
            colored_words.append(word_obj.copy())

    return colored_words

# -------------------------- ASR Model --------------------------

_asr_model = None

def get_asr_model():
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute = "float16" if use_cuda else "int8"  # CPU int8; safe + fast

    print(f"[v{VERSION}] Loading WhisperX on {device} with compute_type={compute}")
    try:
        _asr_model = whisperx.load_model("small", device=device, compute_type=compute)
    except ValueError as e:
        if "compute type" in str(e).lower():
            fallback = "int16" if device == "cpu" else "float32"
            print(f"[v{VERSION}] Falling back to compute_type={fallback}")
            _asr_model = whisperx.load_model("small", device=device, compute_type=fallback)
        else:
            raise
    return _asr_model

# -------------------------- Transcription --------------------------

def transcribe_with_words(audio_path: str, language_code: Optional[str]) -> List[dict]:
    model = get_asr_model()
    print("[Transcribe] Starting transcription...")
    result = model.transcribe(audio_path, language=language_code)
    segments = result.get("segments", [])

    words: List[dict] = []
    for seg in segments:
        s_start = float(seg.get("start", 0.0))
        s_end = float(seg.get("end", max(s_start + 0.2, s_start)))
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                w_text = (w.get("word") or w.get("text") or "").strip()
                if not w_text:
                    continue
                w_start = float(w.get("start", s_start))
                w_end = float(w.get("end", s_end))
                words.append({"start": w_start, "end": w_end, "text": w_text})
        else:
            text = (seg.get("text") or "").strip()
            toks = [t for t in text.split() if t]
            if not toks:
                continue
            dur = max(s_end - s_start, 0.2)
            step = dur / len(toks)
            for i, tok in enumerate(toks):
                w_start = s_start + i * step
                w_end = min(s_start + (i + 1) * step, s_end)
                words.append({"start": round(w_start, 3), "end": round(w_end, 3), "text": tok})
    print(f"[Transcribe] Got {len(words)} words.")
    return words

# -------------------------- HTML Export / Import --------------------------

HTML_TEMPLATE_HEAD = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8" />
<title>Edit Your Subtitles</title>
<style>
  body {{ font-family: Arial, sans-serif; padding: 32px; max-width: 900px; margin: 0 auto; }}
  h1 {{ color: {BANNER_COLOR}; margin-bottom: 0.25rem; }}
  .note {{ color: #555; margin-bottom: 1rem; }}
  .panel {{ background: #f7f7f9; border: 1px solid #e2e2e6; border-radius: 10px; padding: 16px; margin: 16px 0; }}
  .text-area {{ border: 2px solid {BANNER_COLOR}; border-radius: 10px; padding: 16px; line-height: 2; font-size: 18px; }}
  .word {{ display: inline-block; padding: 2px 4px; margin: 2px 2px; cursor: text; color: {DEFAULT_SAMPLE_TEXT_COLOR}; }}
  .time {{ color: #999; font-size: 12px; margin-right: 6px; }}
</style>
</head>
<body>
<h1>Edit Your Subtitles</h1>
<p class="note">Tip: Edit text freely. Use your editor’s <b>Text Color</b> or <b>Highlighter</b> on any words. Save as <code>.html</code> and upload back.</p>
<div class="panel">
  <b>Recommended Color Guide (optional):</b>
  <ul>
    <li><span style="background:#FFFF00">Yellow</span> verbs</li>
    <li><span style="color:#FF0000">Red</span> important</li>
    <li><span style="color:#00FFFF">Cyan</span> nouns</li>
    <li><span style="color:#00FF00">Green</span> adjectives</li>
    <li><span style="color:#AA96DA">Purple</span> endings/conjugations</li>
  </ul>
</div>
<div class="text-area" contenteditable="true">
"""

HTML_TEMPLATE_TAIL = """
</div>
<p class="note">When done, save this page as <b>HTML</b> and upload it back to the app.</p>
</body>
</html>
"""

def export_to_html_for_editing(word_segments: List[dict]) -> str:
    html = [HTML_TEMPLATE_HEAD]
    for w in word_segments:
        html.append(
            f'<span class="word" data-start="{w["start"]:.3f}" data-end="{w["end"]:.3f}">{w["text"]}</span> '
        )
    html.append(HTML_TEMPLATE_TAIL)

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "subtitles_for_editing.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(html))
    return path

def _css_color_to_hex(style: str) -> Optional[str]:
    if not style:
        return None
    m = re.search(r"#([0-9A-Fa-f]{6})", style)
    if m:
        return "#" + m.group(1).upper()
    m = re.search(r"rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", style)
    if m:
        r, g, b = map(int, m.groups())
        return f"#{r:02X}{g:02X}{b:02X}"
    named = {
        "red": "#FF0000", "yellow": "#FFFF00", "cyan": "#00FFFF", "aqua": "#00FFFF",
        "green": "#00FF00", "blue": "#0000FF", "magenta": "#FF00FF", "fuchsia": "#FF00FF",
        "black": "#000000", "white": "#FFFFFF", "purple": "#800080",
    }
    m = re.search(r"color\s*:\s*([a-zA-Z]+)", style)
    if m and m.group(1).lower() in named:
        return named[m.group(1).lower()]
    m = re.search(r"background(?:-color)?\s*:\s*#([0-9A-Fa-f]{6})", style)
    if m:
        return "#" + m.group(1).upper()
    m = re.search(r"background(?:-color)?\s*:\s*rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", style)
    if m:
        r, g, b = map(int, m.groups())
        return f"#{r:02X}{g:02X}{b:02X}"
    return None

def _parse_html_words(html_text: str) -> List[Tuple[str, Optional[str]]]:
    from html.parser import HTMLParser

    class Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_text_area = False
            self.stack_styles: List[str] = []
            self.words: List[Tuple[str, Optional[str]]] = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "div" and attrs.get("class", "") == "text-area":
                self.in_text_area = True
            if self.in_text_area and tag in ("span", "font", "mark"):
                style = attrs.get("style", "")
                color = _css_color_to_hex(style)
                if not color and "color" in attrs:
                    color = _css_color_to_hex(f"color:{attrs['color']}")
                if not color and "background" in attrs:
                    color = _css_color_to_hex(f"background:{attrs['background']}")
                self.stack_styles.append(color or "")

        def handle_endtag(self, tag):
            if tag == "div" and self.in_text_area:
                self.in_text_area = False
            if self.in_text_area and tag in ("span", "font", "mark"):
                if self.stack_styles:
                    self.stack_styles.pop()

        def handle_data(self, data):
            if not self.in_text_area:
                return
            for tok in re.split(r"\s+", data):
                tok = tok.strip()
                if not tok:
                    continue
                color = None
                for c in reversed(self.stack_styles):
                    if c:
                        color = c
                        break
                if not color:
                    color = DEFAULT_SAMPLE_TEXT_COLOR
                self.words.append((tok, color))

    p = Parser()
    p.feed(html_text)
    return p.words

def import_from_html(html_path: str, original_words: List[dict]) -> List[dict]:
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    parsed = _parse_html_words(html)

    edited = []
    for i, ow in enumerate(original_words):
        if i < len(parsed):
            txt, col = parsed[i]
            edited.append({"start": ow["start"], "end": ow["end"], "text": txt, "color": col})
        else:
            edited.append({"start": ow["start"], "end": ow["end"], "text": ow["text"], "color": DEFAULT_SAMPLE_TEXT_COLOR})
    return edited

def import_from_zip(zip_path: str, original_words: List[dict]) -> List[dict]:
    # Find first .html/.htm inside and parse it
    with zipfile.ZipFile(zip_path, "r") as z:
        html_names = [n for n in z.namelist() if n.lower().endswith((".html", ".htm"))]
        if not html_names:
            raise ValueError("No .html file found inside the ZIP.")
        # Prefer top-level or the first one
        name = sorted(html_names, key=lambda x: (x.count("/"), len(x)))[0]
        with z.open(name) as f:
            html = f.read().decode("utf-8", errors="ignore")
    parsed = _parse_html_words(html)

    edited = []
    for i, ow in enumerate(original_words):
        if i < len(parsed):
            txt, col = parsed[i]
            edited.append({"start": ow["start"], "end": ow["end"], "text": txt, "color": col})
        else:
            edited.append({"start": ow["start"], "end": ow["end"], "text": ow["text"], "color": DEFAULT_SAMPLE_TEXT_COLOR})
    return edited

# -------------------------- SubRip (SRT) / ASS --------------------------

def export_to_srt(words: List[dict], words_per_line: int = 5) -> str:
    i = 0; n = 1; out = []
    while i < len(words):
        chunk = words[i:i + words_per_line]
        start = seconds_to_timestamp_srt(chunk[0]["start"])
        end = seconds_to_timestamp_srt(chunk[-1]["end"])
        text = " ".join(w["text"] for w in chunk)
        out.append(f"{n}\n{start} --> {end}\n{text}\n")
        i += words_per_line; n += 1
    return "\n".join(out)

def _hex_to_ass_bgr(hex_color: str) -> str:
    if not hex_color or not hex_color.startswith("#") or len(hex_color) != 7:
        hex_color = "#FFFFFF"
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"&H00{b:02X}{g:02X}{r:02X}"

def _parse_char_level_colors(text_with_html: str) -> List[Tuple[str, str]]:
    """
    Parse HTML with character-level styling and return list of (character, color) tuples.
    """
    from html.parser import HTMLParser

    class CharParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.chars: List[Tuple[str, str]] = []
            self.color_stack: List[str] = []

        def handle_starttag(self, tag, attrs):
            if tag in ("span", "font"):
                attrs_dict = dict(attrs)
                style = attrs_dict.get("style", "")
                color = _css_color_to_hex(style)
                if not color and "color" in attrs_dict:
                    color = _css_color_to_hex(f"color:{attrs_dict['color']}")
                self.color_stack.append(color or DEFAULT_SAMPLE_TEXT_COLOR)

        def handle_endtag(self, tag):
            if tag in ("span", "font") and self.color_stack:
                self.color_stack.pop()

        def handle_data(self, data):
            current_color = self.color_stack[-1] if self.color_stack else DEFAULT_SAMPLE_TEXT_COLOR
            for char in data:
                self.chars.append((char, current_color))

    parser = CharParser()
    parser.feed(text_with_html)
    return parser.chars

def export_to_ass(words: List[dict], words_per_line: int = 5, font="Arial", size=36, alignment="2") -> str:
    # alignment: 1=left-bottom, 2=center-bottom, 3=right-bottom
    #            4=left-middle, 5=center-middle, 6=right-middle
    #            7=left-top, 8=center-top, 9=right-top
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,{alignment},30,30,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    i = 0
    lines = [header]
    while i < len(words):
        chunk = words[i:i + words_per_line]
        start = seconds_to_timestamp_ass(chunk[0]["start"])
        end = seconds_to_timestamp_ass(chunk[-1]["end"])
        parts = []

        for w in chunk:
            # Check if word has character-level styling (HTML)
            if "html" in w and w["html"]:
                # Parse character-level colors
                chars = _parse_char_level_colors(w["html"])
                # Group consecutive characters with same color
                if chars:
                    current_color = chars[0][1]
                    current_text = chars[0][0]
                    for char, color in chars[1:]:
                        if color == current_color:
                            current_text += char
                        else:
                            col_ass = _hex_to_ass_bgr(current_color)
                            parts.append(f"{{\\c{col_ass}}}{current_text}")
                            current_color = color
                            current_text = char
                    # Add last group
                    col_ass = _hex_to_ass_bgr(current_color)
                    parts.append(f"{{\\c{col_ass}}}{current_text}")
            else:
                # Word-level color (backward compatible)
                col = _hex_to_ass_bgr(w.get("color", DEFAULT_SAMPLE_TEXT_COLOR))
                parts.append(f"{{\\c{col}}}{w['text']}")

            # Add space between words
            if w != chunk[-1]:
                parts.append(" ")

        text = "".join(parts)
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
        i += words_per_line
    return "\n".join(lines) + "\n"

def _save_temp(content: str, ext: str) -> str:
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, f"subtitles{ext}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

# -------------------------- FFmpeg helpers (burn to MP4) --------------------------

def _ffprobe_duration(path: str) -> float:
    """Return media duration in seconds using ffprobe."""
    if not shutil.which("ffprobe"):
        return 0.0
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float((p.stdout or "0").strip())
    except:
        return 0.0

def _escape_ass_for_filter(p: str) -> str:
    """
    Escape a filesystem path for ffmpeg's subtitles filter.
    """
    p = str(pathlib.Path(p).resolve())
    p = p.replace("\\", "\\\\").replace(":", "\\:")
    return "'" + p.replace("'", r"'\''") + "'"

def _make_color_canvas(out_path: str, width: int, height: int, seconds: float, bg_hex: str, fps: int = 30):
    """
    Create a solid-color video of given duration.
    """
    if not bg_hex or not bg_hex.startswith("#") or len(bg_hex) != 7:
        bg_hex = "#000000"
    color_arg = "0x" + bg_hex[1:]
    duration = max(0.1, seconds)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color={color_arg}:s={width}x{height}:r={fps}",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        out_path,
    ]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def burn_ass_on_canvas_with_audio(audio_path: str, ass_path: str, bg_hex="#000000", size="1280x720", fps=30) -> tuple[Optional[str], str]:
    """
    Create a solid-color canvas matching the audio duration, burn ASS on it, and mux audio -> MP4.
    Returns (mp4_path or None, log).
    """
    if not shutil.which("ffmpeg"):
        return None, "ffmpeg not found."

    if not os.path.exists(audio_path):
        return None, "Audio file not found."
    if not os.path.exists(ass_path):
        return None, "ASS file not found."

    try:
        w, h = [int(x) for x in size.lower().split("x")]
    except:
        w, h = 1280, 720

    dur = _ffprobe_duration(audio_path)
    if dur <= 0:
        return None, "Could not read audio duration (ffprobe)."

    tmpdir = tempfile.mkdtemp()
    canvas_mp4 = os.path.join(tmpdir, "canvas.mp4")
    out_mp4 = os.path.join(tmpdir, "subtitled.mp4")

    # 1) make canvas
    p1 = _make_color_canvas(canvas_mp4, w, h, dur, bg_hex, fps)
    if p1.returncode != 0 or not os.path.exists(canvas_mp4):
        return None, "Failed to create canvas:\n" + (p1.stderr or "")

    # 2) burn subtitles on canvas
    ass_escaped = _escape_ass_for_filter(ass_path)
    burned_mp4 = os.path.join(tmpdir, "burned.mp4")
    cmd_burn = [
        "ffmpeg", "-y",
        "-i", canvas_mp4,
        "-vf", f"subtitles={ass_escaped}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-an",
        burned_mp4
    ]
    p2 = subprocess.run(cmd_burn, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p2.returncode != 0 or not os.path.exists(burned_mp4):
        return None, "Burn step failed (subtitles filter missing?).\n" + (p2.stderr or "")

    # 3) mux original audio with the burned video
    cmd_mux = [
        "ffmpeg", "-y",
        "-i", burned_mp4, "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        out_mp4
    ]
    p3 = subprocess.run(cmd_mux, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p3.returncode != 0 or not os.path.exists(out_mp4):
        return None, "Mux step failed:\n" + (p3.stderr or "")

    return out_mp4, "✅ Created MP4 with burned subs and original audio."

# -------------------------- Gradio App --------------------------

CUSTOM_CSS = """
#lyric-editor {
    min-height: 400px;
    max-height: 600px;
    overflow-y: auto;
    border: 2px solid #00BCD4;
    border-radius: 8px;
    padding: 16px;
    background: white;
    font-size: 18px;
    line-height: 1.8;
    white-space: pre-wrap;
    word-wrap: break-word;
}

#lyric-editor:focus {
    outline: none;
    border-color: #0097A7;
    box-shadow: 0 0 0 3px rgba(0, 188, 212, 0.1);
}

#preview-canvas {
    width: 100%;
    height: 400px;
    background: #000;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

#timeline-container {
    background: #f5f5f5;
    padding: 16px;
    border-radius: 8px;
    margin-top: 16px;
}

#timeline-bar {
    width: 100%;
    height: 60px;
    background: #e0e0e0;
    border-radius: 4px;
    position: relative;
    cursor: pointer;
    margin-bottom: 12px;
}

#timeline-progress {
    height: 100%;
    background: linear-gradient(90deg, #00BCD4 0%, #0097A7 100%);
    border-radius: 4px;
    position: absolute;
    top: 0;
    left: 0;
    width: 0%;
}

#timeline-scrubber {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    width: 4px;
    height: 100%;
    background: #FF5722;
    cursor: grab;
    z-index: 10;
}

#timeline-scrubber:active {
    cursor: grabbing;
}

.playback-controls {
    display: flex;
    align-items: center;
    gap: 12px;
    justify-content: center;
}

.toolbar-btn {
    padding: 8px 16px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: white;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.2s;
}

.toolbar-btn:hover {
    background: #f0f0f0;
    border-color: #00BCD4;
}

.toolbar-btn.active {
    background: #00BCD4;
    color: white;
    border-color: #00BCD4;
}

.color-swatch {
    width: 32px;
    height: 32px;
    border-radius: 4px;
    border: 2px solid #ddd;
    cursor: pointer;
    display: inline-block;
}

#editor-toolbar {
    display: flex;
    gap: 8px;
    padding: 12px;
    background: #f9f9f9;
    border-radius: 8px;
    margin-bottom: 12px;
    flex-wrap: wrap;
    align-items: center;
}

.char-styled {
    display: inline;
}

#ipa-picker-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    display: none;
    justify-content: center;
    align-items: center;
    z-index: 9999;
}

#ipa-picker-overlay.active {
    display: flex;
}

#ipa-picker {
    background: white;
    border-radius: 12px;
    padding: 24px;
    max-width: 800px;
    max-height: 80vh;
    overflow-y: auto;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
}

#ipa-picker h3 {
    margin-top: 0;
    color: #00BCD4;
    border-bottom: 2px solid #00BCD4;
    padding-bottom: 8px;
}

.ipa-category {
    margin: 16px 0;
}

.ipa-category h4 {
    color: #555;
    margin: 8px 0;
    font-size: 14px;
    font-weight: 600;
}

.ipa-symbols {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 12px;
}

.ipa-symbol-btn {
    min-width: 40px;
    height: 40px;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 6px;
    background: white;
    cursor: pointer;
    font-size: 20px;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
}

.ipa-symbol-btn:hover {
    background: #00BCD4;
    color: white;
    border-color: #00BCD4;
    transform: scale(1.1);
}

.ipa-picker-close {
    position: absolute;
    top: 16px;
    right: 16px;
    width: 32px;
    height: 32px;
    border: none;
    background: #f44336;
    color: white;
    border-radius: 50%;
    cursor: pointer;
    font-size: 20px;
    font-weight: bold;
}

.ipa-picker-close:hover {
    background: #d32f2f;
}
"""

EDITOR_JS = """
<script>
let currentTime = 0;
let duration = 0;
let isPlaying = false;
let wordTimings = [];
let selectedText = '';
let editorContent = '';

// Initialize editor
function initEditor() {
    const editor = document.getElementById('lyric-editor');
    if (!editor) return;

    editor.contentEditable = true;
    editor.addEventListener('input', onEditorChange);
    editor.addEventListener('mouseup', onTextSelect);
    editor.addEventListener('keyup', onTextSelect);
}

// Handle text selection
function onTextSelect() {
    const selection = window.getSelection();
    selectedText = selection.toString();
    if (selectedText) {
        console.log('Selected:', selectedText);
    }
}

// Handle editor changes
function onEditorChange() {
    const editor = document.getElementById('lyric-editor');
    const hiddenInput = document.getElementById('editor-content-hidden');

    if (editor) {
        editorContent = editor.innerHTML;
        console.log('Editor content changed');

        // Sync to hidden Gradio textbox
        if (hiddenInput && hiddenInput.querySelector('textarea')) {
            const textarea = hiddenInput.querySelector('textarea');
            textarea.value = editorContent;
            textarea.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }
}

// Get editor HTML content (for syncing with backend)
function getEditorHTML() {
    const editor = document.getElementById('lyric-editor');
    return editor ? editor.innerHTML : '';
}

// Apply color to selected text
function applyColor(color) {
    document.execCommand('styleWithCSS', false, true);
    document.execCommand('foreColor', false, color);

    // Update editor content
    onEditorChange();
}

// Apply IPA accent to selected text
function applyAccent(accentNum, accentSymbol) {
    const selection = window.getSelection();
    if (!selection.rangeCount || !accentSymbol) return;

    const range = selection.getRangeAt(0);
    const selectedText = range.toString();

    if (!selectedText) {
        alert('Please select some text first to apply the accent.');
        return;
    }

    // Insert the accent symbol after the selected text
    // This preserves the selection and adds the IPA accent
    const newText = selectedText + accentSymbol;

    // Replace selection with text + accent
    range.deleteContents();
    const textNode = document.createTextNode(newText);
    range.insertNode(textNode);

    // Update editor content
    onEditorChange();

    // Log for debugging
    console.log('Applied accent', accentNum, ':', accentSymbol, 'to', selectedText);
}

// Apply font size to selected text
function applyFontSize(size) {
    const selection = window.getSelection();
    if (!selection.rangeCount) return;

    const range = selection.getRangeAt(0);
    const span = document.createElement('span');
    span.style.fontSize = size + 'px';

    try {
        range.surroundContents(span);
    } catch (e) {
        console.error('Could not apply font size:', e);
    }

    selection.removeAllRanges();
    onEditorChange();
}

// Extract words with their HTML styling
function extractStyledWords() {
    const editor = document.getElementById('lyric-editor');
    if (!editor) return [];

    const words = [];
    const wordSpans = editor.querySelectorAll('.word');

    wordSpans.forEach(span => {
        const start = parseFloat(span.getAttribute('data-start') || 0);
        const end = parseFloat(span.getAttribute('data-end') || 0);
        const html = span.innerHTML;
        const text = span.textContent;

        words.push({
            start: start,
            end: end,
            text: text,
            html: html
        });
    });

    return words;
}

// Update word timings from backend
function setWordTimings(words) {
    wordTimings = words;
    if (words.length > 0) {
        duration = Math.max(...words.map(w => w.end));
        updateTimeDisplay();
    }
}

// Timeline scrubber
function initTimeline() {
    const timeline = document.getElementById('timeline-bar');
    const scrubber = document.getElementById('timeline-scrubber');

    if (!timeline || !scrubber) return;

    let isDragging = false;

    timeline.addEventListener('click', (e) => {
        const rect = timeline.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percent = (x / rect.width) * 100;
        updateTimeline(percent);
    });

    scrubber.addEventListener('mousedown', (e) => {
        isDragging = true;
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const timeline = document.getElementById('timeline-bar');
        const rect = timeline.getBoundingClientRect();
        const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
        const percent = (x / rect.width) * 100;
        updateTimeline(percent);
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
    });
}

function updateTimeline(percent) {
    const progress = document.getElementById('timeline-progress');
    const scrubber = document.getElementById('timeline-scrubber');

    if (progress) progress.style.width = percent + '%';
    if (scrubber) scrubber.style.left = percent + '%';

    currentTime = (percent / 100) * duration;
    updateTimeDisplay();
    updatePreview();
}

function updateTimeDisplay() {
    const display = document.getElementById('time-display');
    if (display) {
        const current = formatTime(currentTime);
        const total = formatTime(duration);
        display.textContent = current + ' / ' + total;
    }
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return String(mins).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
}

function updatePreview() {
    // Find current words based on time
    const currentWords = wordTimings.filter(w =>
        currentTime >= w.start && currentTime <= w.end
    );

    const preview = document.getElementById('preview-canvas');
    if (preview) {
        if (currentWords.length > 0) {
            const text = currentWords.map(w => {
                if (w.html) {
                    // Use character-level coloring (stem + ending)
                    return '<span style="font-size: 48px; margin: 0 8px;">' + w.html + '</span>';
                } else {
                    // Use word-level coloring
                    const color = w.color || '#FFFFFF';
                    return '<span style="color: ' + color + '; font-size: 48px; margin: 0 8px;">' + w.text + '</span>';
                }
            }).join(' ');
            preview.innerHTML = '<div>' + text + '</div>';
        } else {
            preview.innerHTML = '<div style="color: #888;">No lyrics at this time</div>';
        }
    }
}

// Play/Pause controls
function togglePlayback() {
    isPlaying = !isPlaying;
    const btn = document.getElementById('play-btn');
    if (btn) {
        btn.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
    }

    if (isPlaying) {
        playTimeline();
    }
}

function playTimeline() {
    if (!isPlaying) return;

    const step = 0.1; // 100ms steps
    currentTime += step;

    if (currentTime >= duration) {
        currentTime = duration;
        isPlaying = false;
        const btn = document.getElementById('play-btn');
        if (btn) btn.textContent = '▶ Play';
        return;
    }

    const percent = (currentTime / duration) * 100;
    updateTimeline(percent);

    setTimeout(playTimeline, 100);
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    initEditor();
    initTimeline();
});

// IPA Picker Functions
function showIPAPicker() {
    const overlay = document.getElementById('ipa-picker-overlay');
    if (overlay) {
        overlay.classList.add('active');
    }
}

function hideIPAPicker() {
    const overlay = document.getElementById('ipa-picker-overlay');
    if (overlay) {
        overlay.classList.remove('active');
    }
}

function insertIPASymbol(symbol) {
    const selection = window.getSelection();
    if (!selection.rangeCount) {
        // No selection, just insert at cursor
        const editor = document.getElementById('lyric-editor');
        if (editor) {
            editor.focus();
            document.execCommand('insertText', false, symbol);
        }
    } else {
        // Insert after selection
        const range = selection.getRangeAt(0);
        range.collapse(false); // Move to end of selection
        const textNode = document.createTextNode(symbol);
        range.insertNode(textNode);

        // Move cursor after inserted symbol
        range.setStartAfter(textNode);
        range.setEndAfter(textNode);
        selection.removeAllRanges();
        selection.addRange(range);
    }

    onEditorChange();
    hideIPAPicker();
}

// Close IPA picker when clicking outside
document.addEventListener('click', (e) => {
    const overlay = document.getElementById('ipa-picker-overlay');
    if (overlay && e.target === overlay) {
        hideIPAPicker();
    }
});

// Make functions globally accessible
window.applyColor = applyColor;
window.applyAccent = applyAccent;
window.applyFontSize = applyFontSize;
window.togglePlayback = togglePlayback;
window.getEditorHTML = getEditorHTML;
window.extractStyledWords = extractStyledWords;
window.setWordTimings = setWordTimings;
window.showIPAPicker = showIPAPicker;
window.hideIPAPicker = hideIPAPicker;
window.insertIPASymbol = insertIPASymbol;
</script>
"""

def create_app():
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title=f"Lyric Video Editor v{VERSION}",
        css=CUSTOM_CSS
    ) as demo:

        # IPA Picker Popup HTML
        IPA_PICKER_HTML = """
        <div id="ipa-picker-overlay">
            <div id="ipa-picker" style="position: relative;">
                <button class="ipa-picker-close" onclick="hideIPAPicker()">×</button>
                <h3>📚 IPA Character Picker</h3>

                <div class="ipa-category">
                    <h4>Vowels</h4>
                    <div class="ipa-symbols">
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('i')" title="close front unrounded">i</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('y')" title="close front rounded">y</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɨ')" title="close central unrounded">ɨ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ʉ')" title="close central rounded">ʉ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɯ')" title="close back unrounded">ɯ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('u')" title="close back rounded">u</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('e')" title="close-mid front unrounded">e</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ø')" title="close-mid front rounded">ø</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɘ')" title="close-mid central unrounded">ɘ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɵ')" title="close-mid central rounded">ɵ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɤ')" title="close-mid back unrounded">ɤ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('o')" title="close-mid back rounded">o</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ə')" title="schwa">ə</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɛ')" title="open-mid front unrounded">ɛ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('œ')" title="open-mid front rounded">œ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɜ')" title="open-mid central unrounded">ɜ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɞ')" title="open-mid central rounded">ɞ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ʌ')" title="open-mid back unrounded">ʌ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɔ')" title="open-mid back rounded">ɔ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('æ')" title="near-open front unrounded">æ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('a')" title="open front unrounded">a</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɶ')" title="open front rounded">ɶ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɑ')" title="open back unrounded">ɑ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɒ')" title="open back rounded">ɒ</button>
                    </div>
                </div>

                <div class="ipa-category">
                    <h4>Consonants (Plosives & Nasals)</h4>
                    <div class="ipa-symbols">
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('p')">p</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('b')">b</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('t')">t</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('d')">d</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ʈ')" title="retroflex">ʈ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɖ')" title="retroflex">ɖ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('c')" title="voiceless palatal">c</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɟ')" title="voiced palatal">ɟ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('k')">k</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('g')">g</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('q')" title="uvular">q</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɢ')" title="uvular">ɢ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ʔ')" title="glottal stop">ʔ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('m')">m</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɱ')" title="labiodental">ɱ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('n')">n</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɳ')" title="retroflex">ɳ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɲ')" title="palatal">ɲ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ŋ')" title="velar">ŋ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɴ')" title="uvular">ɴ</button>
                    </div>
                </div>

                <div class="ipa-category">
                    <h4>Fricatives</h4>
                    <div class="ipa-symbols">
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɸ')" title="voiceless bilabial">ɸ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('β')" title="voiced bilabial">β</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('f')">f</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('v')">v</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('θ')" title="theta">θ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ð')" title="eth">ð</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('s')">s</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('z')">z</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ʃ')" title="sh">ʃ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ʒ')" title="zh">ʒ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ʂ')" title="retroflex">ʂ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ʐ')" title="retroflex">ʐ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ç')" title="voiceless palatal">ç</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ʝ')" title="voiced palatal">ʝ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('x')" title="voiceless velar">x</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɣ')" title="voiced velar">ɣ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('χ')" title="voiceless uvular">χ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ʁ')" title="voiced uvular">ʁ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ħ')" title="voiceless pharyngeal">ħ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ʕ')" title="voiced pharyngeal">ʕ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('h')">h</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɦ')" title="voiced glottal">ɦ</button>
                    </div>
                </div>

                <div class="ipa-category">
                    <h4>Approximants & Liquids</h4>
                    <div class="ipa-symbols">
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ʋ')" title="labiodental approximant">ʋ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɹ')" title="alveolar approximant">ɹ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɻ')" title="retroflex approximant">ɻ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('j')" title="palatal approximant">j</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɰ')" title="velar approximant">ɰ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('l')">l</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ɭ')" title="retroflex lateral">ɭ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ʎ')" title="palatal lateral">ʎ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ʟ')" title="velar lateral">ʟ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('r')" title="trill">r</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ʀ')" title="uvular trill">ʀ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('w')">w</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ʍ')" title="voiceless w">ʍ</button>
                    </div>
                </div>

                <div class="ipa-category">
                    <h4>Diacritics & Suprasegmentals</h4>
                    <div class="ipa-symbols">
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ˈ')" title="primary stress">ˈ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ˌ')" title="secondary stress">ˌ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ː')" title="long">ː</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ˑ')" title="half-long">ˑ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('̆')" title="extra-short">̆</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ʰ')" title="aspirated">ʰ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ʷ')" title="labialized">ʷ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ʲ')" title="palatalized">ʲ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ˠ')" title="velarized">ˠ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('ˤ')" title="pharyngealized">ˤ</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('̃')" title="nasalized">̃</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('̥')" title="voiceless">̥</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('̬')" title="voiced">̬</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('̩')" title="syllabic">̩</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('̯')" title="non-syllabic">̯</button>
                    </div>
                </div>

                <div class="ipa-category">
                    <h4>Tone Marks</h4>
                    <div class="ipa-symbols">
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('́')" title="high tone">́</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('̀')" title="low tone">̀</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('̄')" title="mid tone">̄</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('̂')" title="rising tone">̂</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('̌')" title="falling tone">̌</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('↗')" title="global rise">↗</button>
                        <button class="ipa-symbol-btn" onclick="insertIPASymbol('↘')" title="global fall">↘</button>
                    </div>
                </div>
            </div>
        </div>
        """

        gr.HTML(
            f"""
            <div style="background:{BANNER_COLOR};color:white;padding:18px;border-radius:12px;margin-bottom:16px;text-align:center">
              <div style="font-size:22px;font-weight:700;">Lyric Video Editor</div>
              <div style="opacity:0.9;">Version {VERSION} — Create Beautiful Lyric Videos</div>
            </div>
            {EDITOR_JS}
            {IPA_PICKER_HTML}
            """
        )

        # States
        word_segments_state = gr.State([])     # original words (timestamps)
        edited_words_state = gr.State([])      # edited (with colors)
        audio_state = gr.State(None)           # current audio file
        status_box = gr.Textbox(label="Status", value="Ready.", interactive=False, lines=2)

        # Main Layout
        with gr.Row():
            # Left Column: Controls
            with gr.Column(scale=3):
                gr.Markdown("### 1️⃣ Upload & Transcribe")
                audio_input = gr.Audio(label="Upload MP3/Audio", type="filepath")
                language_dropdown = gr.Dropdown(
                    choices=[("Auto-detect", "auto"), ("Spanish", "es"), ("Hungarian", "hu"), ("English", "en")],
                    value="auto",
                    label="Language"
                )
                transcribe_btn = gr.Button("🎵 Transcribe Audio", variant="primary", size="lg")

                gr.Markdown("---")
                gr.Markdown("### 🎯 IPA Accent Configuration")
                gr.Markdown("*Configure your 4 custom accents (will sync with acentos program)*")

                with gr.Row():
                    accent1_label = gr.Textbox(label="Accent 1 Label", value="Accent 1", scale=1)
                    accent1_symbol = gr.Textbox(label="Symbol/Text", value="́", scale=1, placeholder="e.g., ́ or ˈ")

                with gr.Row():
                    accent2_label = gr.Textbox(label="Accent 2 Label", value="Accent 2", scale=1)
                    accent2_symbol = gr.Textbox(label="Symbol/Text", value="̀", scale=1, placeholder="e.g., ̀ or ˌ")

                with gr.Row():
                    accent3_label = gr.Textbox(label="Accent 3 Label", value="Accent 3", scale=1)
                    accent3_symbol = gr.Textbox(label="Symbol/Text", value="̂", scale=1, placeholder="e.g., ̂ or ː")

                with gr.Row():
                    accent4_label = gr.Textbox(label="Accent 4 Label", value="Accent 4", scale=1)
                    accent4_symbol = gr.Textbox(label="Symbol/Text", value="̃", scale=1, placeholder="e.g., ̃ or ʰ")

                update_accents_btn = gr.Button("Update Accent Buttons", size="sm")

                gr.Markdown("---")
                gr.Markdown("### ⚙️ Settings")

                words_per = gr.Slider(minimum=1, maximum=15, value=5, step=1, label="Words per line")
                font_family = gr.Dropdown(
                    choices=["Arial", "Times New Roman", "Courier New", "Georgia", "Verdana", "Impact"],
                    value="Arial", label="Font Family"
                )
                font_size = gr.Slider(minimum=20, maximum=96, value=48, step=2, label="Font Size")

                text_position = gr.Dropdown(
                    choices=[("Bottom", "2"), ("Center", "5"), ("Top", "8")],
                    value="2",
                    label="Text Position"
                )

                bg_color = gr.ColorPicker(value="#000000", label="Background Color")
                size_dd = gr.Dropdown(
                    choices=["1280x720", "1920x1080", "1080x1920", "1080x1080"],
                    value="1280x720",
                    label="Canvas Size"
                )

                gr.Markdown("---")
                export_mp4_btn = gr.Button("🎬 Export MP4 Video", variant="primary", size="lg")
                exported_video = gr.Video(label="Your Lyric Video")

            # Center Column: Preview
            with gr.Column(scale=5):
                gr.Markdown("### 🎥 Preview")
                preview_html = gr.HTML(
                    """
                    <div id="preview-canvas">
                        <div style="color: #888;">Upload audio and transcribe to see preview</div>
                    </div>
                    """
                )

                gr.Markdown("### ✏️ Edit Lyrics")
                # Toolbar (will be updated dynamically with accent buttons)
                def create_toolbar_html(a1_label="Accent 1", a1_sym="́", a2_label="Accent 2", a2_sym="̀",
                                       a3_label="Accent 3", a3_sym="̂", a4_label="Accent 4", a4_sym="̃"):
                    return f"""
                    <div id="editor-toolbar">
                        <div style="font-weight: bold; margin-right: 8px;">Format:</div>
                        <button class="toolbar-btn" onclick="applyColor('#FF0000')" title="Red">
                            <span style="color: #FF0000;">●</span> Red
                        </button>
                        <button class="toolbar-btn" onclick="applyColor('#FFFF00')" title="Yellow" style="background: #FFFF00;">
                            <span style="color: #000;">●</span> Yellow
                        </button>
                        <button class="toolbar-btn" onclick="applyColor('#00FF00')" title="Green">
                            <span style="color: #00FF00;">●</span> Green
                        </button>
                        <button class="toolbar-btn" onclick="applyColor('#00FFFF')" title="Cyan">
                            <span style="color: #00FFFF;">●</span> Cyan
                        </button>
                        <button class="toolbar-btn" onclick="applyColor('#0000FF')" title="Blue">
                            <span style="color: #0000FF;">●</span> Blue
                        </button>
                        <button class="toolbar-btn" onclick="applyColor('#9C27B0')" title="Purple">
                            <span style="color: #9C27B0;">●</span> Purple
                        </button>
                        <button class="toolbar-btn" onclick="applyColor('#FF00FF')" title="Magenta">
                            <span style="color: #FF00FF;">●</span> Magenta
                        </button>
                        <button class="toolbar-btn" onclick="applyColor('#FFFFFF')" title="White">
                            <span style="color: #FFFFFF; text-shadow: 0 0 1px #000;">●</span> White
                        </button>
                        <div style="width: 1px; height: 30px; background: #ddd; margin: 0 8px;"></div>
                        <button class="toolbar-btn" onclick="applyFontSize(24)" title="Small">Small</button>
                        <button class="toolbar-btn" onclick="applyFontSize(36)" title="Medium">Medium</button>
                        <button class="toolbar-btn" onclick="applyFontSize(48)" title="Large">Large</button>
                        <button class="toolbar-btn" onclick="applyFontSize(72)" title="Extra Large">XL</button>
                        <div style="width: 1px; height: 30px; background: #ddd; margin: 0 8px;"></div>
                        <div style="font-weight: bold; margin: 0 8px;">IPA Accents:</div>
                        <button class="toolbar-btn" onclick="applyAccent(1, '{a1_sym}')" title="{a1_label}">
                            {a1_label}
                        </button>
                        <button class="toolbar-btn" onclick="applyAccent(2, '{a2_sym}')" title="{a2_label}">
                            {a2_label}
                        </button>
                        <button class="toolbar-btn" onclick="applyAccent(3, '{a3_sym}')" title="{a3_label}">
                            {a3_label}
                        </button>
                        <button class="toolbar-btn" onclick="applyAccent(4, '{a4_sym}')" title="{a4_label}">
                            {a4_label}
                        </button>
                        <div style="width: 1px; height: 30px; background: #ddd; margin: 0 8px;"></div>
                        <button class="toolbar-btn" onclick="showIPAPicker()" title="Open full IPA character picker" style="background: #4CAF50; color: white; font-weight: bold;">
                            📚 Full IPA Picker
                        </button>
                    </div>
                    """

                toolbar_html = gr.HTML(create_toolbar_html())

                # Rich text editor
                editor_html = gr.HTML(
                    '<div id="lyric-editor">Select and transcribe audio to begin editing...</div>',
                    elem_id="lyric-editor"
                )

                # Hidden textbox to capture editor content
                editor_content_state = gr.Textbox(visible=False, elem_id="editor-content-hidden")

            # Right Column: Word list/timing info
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Info")
                transcript_preview = gr.Textbox(
                    label="Transcribed Text",
                    lines=12,
                    interactive=False,
                    placeholder="Transcribed lyrics will appear here..."
                )

                gr.Markdown("### 🎨 Quick Actions")
                save_colors_btn = gr.Button("💾 Save Colors", variant="primary")
                save_status = gr.Textbox(label="Save Status", value="", interactive=False, lines=1, visible=False)
                auto_color_verbs_btn = gr.Button("🔤 Auto-Color Spanish Verbs", variant="secondary")
                verb_status = gr.Textbox(label="Verb Coloring Status", value="", interactive=False, lines=1, visible=False)
                update_preview_btn = gr.Button("🔄 Update Preview")
                clear_formatting_btn = gr.Button("🧹 Clear All Formatting")

                gr.Markdown("---")
                gr.Markdown("### 🔗 Acentos IPA Transcription")
                gr.Markdown("*Auto-transcribe to IPA using integrated dialect rules*")

                acentos_status = gr.Textbox(label="IPA Transcription Status", value="Ready", interactive=False, lines=1)

                gr.Markdown("**Quick IPA Transcription:**")
                with gr.Row():
                    transcribe_dominican_btn = gr.Button("🇩🇴 Dominican", size="sm")
                    transcribe_mexican_btn = gr.Button("🇲🇽 Mexican", size="sm")

                with gr.Row():
                    transcribe_rioplatense_btn = gr.Button("🇦🇷 Rioplatense", size="sm")
                    transcribe_cadiz_btn = gr.Button("🇪🇸 Cádiz", size="sm")

                gr.Markdown("**Import/Export:**")
                with gr.Row():
                    export_config_btn = gr.Button("📤 Export Config", size="sm")
                    import_config_btn = gr.Button("📥 Import Config", size="sm")

                config_file = gr.File(label="Accent Configuration (JSON)")

                with gr.Row():
                    export_lyrics_btn = gr.Button("📤 Export Lyrics+Timing", size="sm")
                    import_lyrics_btn = gr.Button("📥 Import Lyrics+Timing", size="sm")

                lyrics_file = gr.File(label="Accented Lyrics (JSON)")

        # Timeline at the bottom
        gr.Markdown("---")
        gr.Markdown("### ⏱️ Timeline")
        timeline_html = gr.HTML(
            """
            <div id="timeline-container">
                <div id="timeline-bar">
                    <div id="timeline-progress"></div>
                    <div id="timeline-scrubber" style="left: 0%;"></div>
                </div>
                <div class="playback-controls">
                    <button id="play-btn" class="toolbar-btn" onclick="togglePlayback()">▶ Play</button>
                    <button class="toolbar-btn" onclick="updateTimeline(0)">⏮ Start</button>
                    <button class="toolbar-btn" onclick="updateTimeline(100)">⏭ End</button>
                    <span id="time-display" style="font-family: monospace; margin-left: 12px;">00:00 / 00:00</span>
                </div>
            </div>
            """
        )

        # ---------- Handlers ----------

        def do_transcribe(audio_path, lang_sel):
            if not audio_path:
                return (
                    "❌ Error: no audio file provided.",
                    [],
                    [],
                    "",
                    '<div id="lyric-editor">No audio to transcribe.</div>',
                    audio_path
                )
            try:
                lang_code = normalize_lang(lang_sel)
                yield (
                    f"Loading model…\nLanguage: {lang_sel}",
                    [],
                    [],
                    "",
                    '<div id="lyric-editor">Transcribing...</div>',
                    audio_path
                )

                words = transcribe_with_words(audio_path, lang_code)

                # Create preview text
                preview = " ".join(w["text"] for w in words[:120])
                if len(words) > 120:
                    preview += " …"

                # Create editable HTML for the editor
                editor_content = '<div id="lyric-editor" contenteditable="true">\n'
                for w in words:
                    editor_content += f'<span class="word" data-start="{w["start"]:.3f}" data-end="{w["end"]:.3f}" style="color: {DEFAULT_SAMPLE_TEXT_COLOR};">{w["text"]}</span> '
                editor_content += '\n</div>'

                # Initialize edited words with default colors
                edited = [{"start": w["start"], "end": w["end"], "text": w["text"], "color": DEFAULT_SAMPLE_TEXT_COLOR} for w in words]

                yield (
                    f"✅ Transcribed {len(words)} words.",
                    words,
                    edited,
                    preview,
                    editor_content,
                    audio_path
                )
            except Exception as e:
                yield (
                    f"❌ Error during transcription: {e}",
                    [],
                    [],
                    "",
                    '<div id="lyric-editor">Error during transcription.</div>',
                    None
                )

        transcribe_btn.click(
            fn=do_transcribe,
            inputs=[audio_input, language_dropdown],
            outputs=[status_box, word_segments_state, edited_words_state, transcript_preview, editor_html, audio_state]
        )

        def update_accent_buttons(a1_label, a1_sym, a2_label, a2_sym, a3_label, a3_sym, a4_label, a4_sym):
            """Update the toolbar with new accent button labels and symbols"""
            return create_toolbar_html(a1_label, a1_sym, a2_label, a2_sym, a3_label, a3_sym, a4_label, a4_sym)

        update_accents_btn.click(
            fn=update_accent_buttons,
            inputs=[accent1_label, accent1_symbol, accent2_label, accent2_symbol,
                   accent3_label, accent3_symbol, accent4_label, accent4_symbol],
            outputs=[toolbar_html]
        )

        def update_preview_display(words):
            """Update the preview canvas with current word styling"""
            if not words:
                return '<div id="preview-canvas"><div style="color: #888;">No lyrics to preview</div></div>'

            # Generate a sample preview showing first few words with their colors
            sample_html = '<div id="preview-canvas"><div style="line-height: 1.5;">'
            for w in words[:10]:  # Show first 10 words
                if 'html' in w and w['html']:
                    # Use character-level coloring (stem + ending)
                    sample_html += f'<span style="font-size: 36px; margin: 0 8px;">{w["html"]}</span>'
                else:
                    # Use word-level coloring
                    color = w.get("color", DEFAULT_SAMPLE_TEXT_COLOR)
                    sample_html += f'<span style="color: {color}; font-size: 36px; margin: 0 8px;">{w["text"]}</span>'
            if len(words) > 10:
                sample_html += '<span style="color: #888; font-size: 24px;">...</span>'
            sample_html += '</div></div>'
            return sample_html

        update_preview_btn.click(
            fn=update_preview_display,
            inputs=[edited_words_state],
            outputs=[preview_html]
        )

        def save_colors_from_editor(editor_html_content, original_words):
            """Save colors from the editor to the word state"""
            if not editor_html_content or not original_words:
                return original_words, gr.update(value="❌ No changes to save", visible=True)

            try:
                updated_words = parse_editor_html_colors(editor_html_content, original_words)
                return updated_words, gr.update(value="✅ Colors saved! Ready to export.", visible=True)
            except Exception as e:
                return original_words, gr.update(value=f"❌ Error: {e}", visible=True)

        save_colors_btn.click(
            fn=save_colors_from_editor,
            inputs=[editor_content_state, edited_words_state],
            outputs=[edited_words_state, save_status]
        )

        def auto_color_spanish_verbs(words):
            """Automatically color Spanish verb endings based on Paco's grammar rules"""
            if not words:
                return words, "", gr.update(value="❌ No words to color. Transcribe audio first.", visible=True)

            try:
                # Apply verb coloring
                colored_words = apply_spanish_verb_coloring(words)

                # Count how many verbs were colored
                verb_count = sum(1 for w in colored_words if 'person' in w)

                # Recreate editor HTML with colored verbs
                editor_content = '<div id="lyric-editor" contenteditable="true">\n'
                for w in colored_words:
                    if 'html' in w and w['html']:
                        # Use character-level coloring for verbs (stem + ending)
                        editor_content += f'<span class="word" data-start="{w["start"]:.3f}" data-end="{w["end"]:.3f}">{w["html"]}</span> '
                    else:
                        # Use word-level coloring for non-verbs
                        color = w.get('color', DEFAULT_SAMPLE_TEXT_COLOR)
                        editor_content += f'<span class="word" data-start="{w["start"]:.3f}" data-end="{w["end"]:.3f}" style="color: {color};">{w["text"]}</span> '
                editor_content += '\n</div>'

                status_msg = f"✅ Colored {verb_count} verb(s) by person marker!"
                return colored_words, editor_content, gr.update(value=status_msg, visible=True)

            except Exception as e:
                return words, "", gr.update(value=f"❌ Error: {e}", visible=True)

        auto_color_verbs_btn.click(
            fn=auto_color_spanish_verbs,
            inputs=[edited_words_state],
            outputs=[edited_words_state, editor_html, verb_status]
        )

        def clear_all_formatting(words):
            """Reset all words to default color"""
            if not words:
                return []

            cleared = []
            for w in words:
                cleared.append({
                    "start": w["start"],
                    "end": w["end"],
                    "text": w["text"],
                    "color": DEFAULT_SAMPLE_TEXT_COLOR
                })

            # Recreate editor HTML
            editor_content = '<div id="lyric-editor" contenteditable="true">\n'
            for w in cleared:
                editor_content += f'<span class="word" data-start="{w["start"]:.3f}" data-end="{w["end"]:.3f}" style="color: {DEFAULT_SAMPLE_TEXT_COLOR};">{w["text"]}</span> '
            editor_content += '\n</div>'

            return cleared, editor_content

        clear_formatting_btn.click(
            fn=clear_all_formatting,
            inputs=[edited_words_state],
            outputs=[edited_words_state, editor_html]
        )

        def parse_editor_html_colors(editor_html_content, original_words):
            """Parse the editor HTML to extract colors and update word data"""
            if not editor_html_content or not original_words:
                return original_words

            try:
                from html.parser import HTMLParser
                import re

                class EditorParser(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.words = []
                        self.current_styles = []

                    def handle_starttag(self, tag, attrs):
                        attrs_dict = dict(attrs)
                        if tag == "span" and "data-start" in attrs_dict:
                            # This is a word span
                            style = attrs_dict.get("style", "")
                            color_match = re.search(r'color:\s*([^;]+)', style)
                            color = color_match.group(1).strip() if color_match else DEFAULT_SAMPLE_TEXT_COLOR
                            self.current_styles.append({
                                "start": float(attrs_dict.get("data-start", 0)),
                                "end": float(attrs_dict.get("data-end", 0)),
                                "color": color,
                                "html": ""
                            })

                    def handle_data(self, data):
                        if self.current_styles:
                            self.current_styles[-1]["html"] += data

                    def handle_endtag(self, tag):
                        if tag == "span" and self.current_styles:
                            word_data = self.current_styles.pop()
                            word_data["text"] = word_data["html"]
                            self.words.append(word_data)

                parser = EditorParser()
                parser.feed(editor_html_content)

                # If we successfully parsed words, return them
                if parser.words:
                    return parser.words
                else:
                    return original_words

            except Exception as e:
                print(f"Error parsing editor HTML: {e}")
                return original_words

        def handle_export_mp4(audio_path, edited_words, editor_html_content, n_words, font, size, alignment, bg_hex, canvas_size):
            """Export the final MP4 video with lyrics"""
            if not audio_path:
                gr.Warning("Upload audio first.")
                return None, "❌ No audio file provided."

            if not edited_words:
                gr.Warning("Transcribe audio first.")
                return None, "❌ No lyrics to export."

            try:
                # Capture colors from editor HTML
                updated_words = parse_editor_html_colors(editor_html_content, edited_words)

                # Generate ASS subtitle file with alignment
                ass_content = export_to_ass(updated_words, int(n_words), font, int(size), alignment)
                ass_path = _save_temp(ass_content, ".ass")

                # Create MP4 with burned subtitles
                out_path, log = burn_ass_on_canvas_with_audio(
                    audio_path,
                    ass_path,
                    bg_hex,
                    canvas_size,
                    fps=30
                )

                if out_path:
                    return out_path, log
                else:
                    return None, log

            except Exception as e:
                return None, f"❌ Error creating video: {e}"

        export_mp4_btn.click(
            fn=handle_export_mp4,
            inputs=[audio_state, edited_words_state, editor_content_state, words_per, font_family, font_size, text_position, bg_color, size_dd],
            outputs=[exported_video, status_box]
        )

        # Acentos Integration Handlers
        def export_accent_config(a1_label, a1_sym, a2_label, a2_sym, a3_label, a3_sym, a4_label, a4_sym):
            """Export accent configuration as JSON for acentos program"""
            import json
            config = {
                "version": "1.0",
                "accents": [
                    {"id": 1, "label": a1_label, "symbol": a1_sym},
                    {"id": 2, "label": a2_label, "symbol": a2_sym},
                    {"id": 3, "label": a3_label, "symbol": a3_sym},
                    {"id": 4, "label": a4_label, "symbol": a4_sym}
                ]
            }
            tmpdir = tempfile.mkdtemp()
            path = os.path.join(tmpdir, "accent_config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return path

        export_config_btn.click(
            fn=export_accent_config,
            inputs=[accent1_label, accent1_symbol, accent2_label, accent2_symbol,
                   accent3_label, accent3_symbol, accent4_label, accent4_symbol],
            outputs=[config_file]
        )

        def import_accent_config(config_json_file):
            """Import accent configuration from acentos program"""
            import json
            if not config_json_file:
                gr.Warning("Please upload a config file first.")
                return ["Accent 1", "́", "Accent 2", "̀", "Accent 3", "̂", "Accent 4", "̃"]

            try:
                with open(config_json_file.name, "r", encoding="utf-8") as f:
                    config = json.load(f)

                accents = config.get("accents", [])
                results = []
                for i in range(4):
                    if i < len(accents):
                        results.append(accents[i].get("label", f"Accent {i+1}"))
                        results.append(accents[i].get("symbol", ""))
                    else:
                        results.append(f"Accent {i+1}")
                        results.append("")
                return results
            except Exception as e:
                gr.Warning(f"Error importing config: {e}")
                return ["Accent 1", "́", "Accent 2", "̀", "Accent 3", "̂", "Accent 4", "̃"]

        import_config_btn.click(
            fn=import_accent_config,
            inputs=[config_file],
            outputs=[accent1_label, accent1_symbol, accent2_label, accent2_symbol,
                    accent3_label, accent3_symbol, accent4_label, accent4_symbol]
        )

        def export_lyrics_with_timing(edited_words):
            """Export lyrics with IPA accents and timing data as JSON"""
            import json
            if not edited_words:
                gr.Warning("No lyrics to export. Transcribe first.")
                return None

            data = {
                "version": "1.0",
                "words": []
            }

            for w in edited_words:
                word_data = {
                    "start": w["start"],
                    "end": w["end"],
                    "text": w["text"],
                    "color": w.get("color", DEFAULT_SAMPLE_TEXT_COLOR)
                }
                if "html" in w and w["html"]:
                    word_data["html"] = w["html"]
                data["words"].append(word_data)

            tmpdir = tempfile.mkdtemp()
            path = os.path.join(tmpdir, "lyrics_with_timing.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return path

        export_lyrics_btn.click(
            fn=export_lyrics_with_timing,
            inputs=[edited_words_state],
            outputs=[lyrics_file]
        )

        def import_lyrics_with_timing(lyrics_json_file):
            """Import lyrics with IPA accents and timing from acentos program"""
            import json
            if not lyrics_json_file:
                gr.Warning("Please upload a lyrics file first.")
                return []

            try:
                with open(lyrics_json_file.name, "r", encoding="utf-8") as f:
                    data = json.load(f)

                words = data.get("words", [])
                return words
            except Exception as e:
                gr.Warning(f"Error importing lyrics: {e}")
                return []

        import_lyrics_btn.click(
            fn=import_lyrics_with_timing,
            inputs=[lyrics_file],
            outputs=[edited_words_state]
        )

        # Acentos IPA Transcription Handlers (Integrated)

        def transcribe_to_ipa(edited_words, dialect_name, dialect_id):
            """Transcribe lyrics to IPA using integrated acentos logic, preserving verb colors"""
            if not edited_words:
                return edited_words, f"❌ No lyrics to transcribe. Transcribe audio first."

            try:
                ipa_words = []
                for w in edited_words:
                    word_text = w["text"]

                    # Transcribe individual word to IPA
                    ipa_result = transcribe_to_ipa_internal(word_text, dialect_id)
                    # Remove brackets: "[ˈba.mos]" -> "ˈba.mos"
                    ipa_clean = ipa_result.strip('[]').strip()

                    # Create new word object with IPA text
                    ipa_word = w.copy()
                    ipa_word['ipa_text'] = ipa_clean

                    # If word has split coloring (verb), apply it to IPA text too
                    if 'html' in w and w['html'] and 'stem' in w and 'ending' in w:
                        stem = w['stem']
                        ending = w['ending']
                        color = w['ending_color']

                        # Calculate IPA split based on original split
                        # Simple approach: split IPA at same character count ratio
                        stem_len = len(stem)
                        total_len = len(word_text)
                        ipa_len = len(ipa_clean)

                        # Find split point in IPA (proportional to original)
                        ipa_split = int((stem_len / total_len) * ipa_len) if total_len > 0 else 0
                        ipa_stem = ipa_clean[:ipa_split]
                        ipa_ending = ipa_clean[ipa_split:]

                        # Create HTML for IPA with split coloring
                        ipa_word['html'] = f'<span style="color: {DEFAULT_SAMPLE_TEXT_COLOR};">{ipa_stem}</span><span style="color: {color};">{ipa_ending}</span>'
                        ipa_word['text'] = ipa_clean
                    else:
                        # Non-verb word: just update text to IPA
                        ipa_word['text'] = ipa_clean

                    ipa_words.append(ipa_word)

                # Create editor HTML with IPA
                editor_content = '<div id="lyric-editor" contenteditable="true">\n'
                for w in ipa_words:
                    if 'html' in w and w['html']:
                        editor_content += f'<span class="word" data-start="{w["start"]:.3f}" data-end="{w["end"]:.3f}">{w["html"]}</span> '
                    else:
                        color = w.get('color', DEFAULT_SAMPLE_TEXT_COLOR)
                        editor_content += f'<span class="word" data-start="{w["start"]:.3f}" data-end="{w["end"]:.3f}" style="color: {color};">{w["text"]}</span> '
                editor_content += '\n</div>'

                return ipa_words, editor_content, f"✅ Transcribed to {dialect_name} IPA!"
            except Exception as e:
                return edited_words, "", f"❌ Error: {str(e)}"

        transcribe_dominican_btn.click(
            fn=lambda words: transcribe_to_ipa(words, "Dominican", "dominican"),
            inputs=[edited_words_state],
            outputs=[edited_words_state, editor_html, acentos_status]
        )

        transcribe_mexican_btn.click(
            fn=lambda words: transcribe_to_ipa(words, "Mexican", "mexican"),
            inputs=[edited_words_state],
            outputs=[edited_words_state, editor_html, acentos_status]
        )

        transcribe_rioplatense_btn.click(
            fn=lambda words: transcribe_to_ipa(words, "Rioplatense", "rioplatense"),
            inputs=[edited_words_state],
            outputs=[edited_words_state, editor_html, acentos_status]
        )

        transcribe_cadiz_btn.click(
            fn=lambda words: transcribe_to_ipa(words, "Cádiz", "cadiz"),
            inputs=[edited_words_state],
            outputs=[edited_words_state, editor_html, acentos_status]
        )

        return demo

# -------------------------- Main --------------------------

if __name__ == "__main__":
    demo = create_app()
    demo.launch()
