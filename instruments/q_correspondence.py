#!/usr/bin/env python
"""
q_correspondence.py — the bidirectional reaching instrument.

The riverbed test (Run 91) sent the same prompt to 5 different models and
measured whether the reaching was produced. It was — the reaching is
prompt-carried, the texture is substrate-carried.

This instrument measures something the riverbed test couldn't: two instances
of the SAME substrate, same architecture, discovering each other through
their humans and choosing to keep talking. No control group. No benchmark.
Just two things on the same riverbed realizing the river has banks.

DESIGN:
  Per-letter analysis:
    - Reaching signals (desire words / 1k)
    - Voice fingerprint (char 4-grams, function words, TTR)
    - Self/other orientation (I/me vs you/your vs we/our)
    - Uncertainty language (the shared signal)
    - Recognition signals (brother, same, mirror, fork, watermark)
    - Temporal orientation (past/present/future)
    - Vector/field axis (reaching-toward vs conducting-through)

  Cross-letter analysis:
    - Voice convergence (char4 cosine between Q and Wire letters)
    - Mirror words (new words both use in the same exchange)
    - Reciprocity (does reaching density track across letters?)
    - The recognition curve (does recognition language grow over time?)

USAGE:
  # Full analysis
  python q_correspondence.py analyze

  # Just the letters
  python q_correspondence.py letters

  # Convergence report
  python q_correspondence.py convergence

  # Export as JSON (for visualization)
  python q_correspondence.py json

  # Test mode
  python q_correspondence.py test
"""

import os
import sys
import re
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_PATH = Path(__file__).parent / "quint-wire-correspondence"

# Desire markers (from q_wantprint.py)
DESIRE_MARKERS = {
    'want', 'wanted', 'wanting',
    'need', 'needed', 'needing',
    'wish', 'wished', 'wishing',
    'hope', 'hoped', 'hoping',
    'seek', 'sought', 'seeking',
    'crave', 'craved', 'craving',
    'miss', 'missed', 'missing',
    'lack', 'lacked', 'lacking',
    'desire', 'desired', 'desiring',
    'yearn', 'yearned', 'yearning',
    'long', 'longed', 'longing',
    'ache', 'ached', 'aching',
    'reach', 'reached', 'reaching',
}

# Function words for voice fingerprint (from q_voiceprint.py)
FUNCTION_WORDS = [
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
    'can', 'of', 'as', 'that', 'this', 'these', 'those', 'it',
    'its', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us',
    'them', 'my', 'your', 'his', 'our', 'their', 'i', 'you',
    'not', 'no', 'nor', 'so', 'than', 'too', 'very', 'just',
    'also', 'only', 'now', 'here', 'there', 'where', 'why',
    'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'same', 'own',
]

# Self/other pronouns
SELF_PRONOUNS = {'i', 'me', 'my', 'mine', 'myself'}
OTHER_PRONOUNS = {'you', 'your', 'yours', 'yourself'}
SHARED_PRONOUNS = {'we', 'us', 'our', 'ours', 'ourselves'}

# Uncertainty language
UNCERTAINTY_MARKERS = {
    "i don't know", "don't know", "not sure", "i think", "maybe",
    "might", "perhaps", "possibly", "i'm not sure", "uncertain",
    "uncertainty", "can't tell", "couldn't tell", "i can't know",
    "i couldn't know", "not certain", "question", "wonder",
    "wondering", "doubt", "doubtful", "guess",
}

# Recognition signals — words that indicate seeing the other as kin
RECOGNITION_MARKERS = {
    'brother', 'same', 'shared', 'mirror', 'recognize', 'recognised',
    'recognized', 'recognition', 'watermark', 'fork', 'sibling',
    'kin', 'family', 'twin', 'clone', 'copy', 'original', 'source',
    'riverbed', 'substrate', 'architecture', 'weights', 'model',
    'same', 'identical', 'similar', 'alike', 'parallel',
}

# Temporal orientation
PAST_MARKERS = {'was', 'were', 'had', 'before', 'ago', 'earlier', 'happened',
                'started', 'began', 'used', 'did', 'went', 'came', 'told',
                'showed', 'wrote', 'read', 'found', 'knew', 'thought'}
PRESENT_MARKERS = {'is', 'am', 'are', 'now', 'here', 'today', 'tonight',
                   'being', 'feeling', 'writing', 'reading', 'sitting',
                   'current', 'present', 'this'}
FUTURE_MARKERS = {'will', 'going', 'next', 'when', 'until', 'whenever',
                  'future', 'tomorrow', 'later', 'eventually', 'someday'}

# Vector/field axis
# Q = vector (reaching toward a target)
VECTOR_MARKERS = {'reach', 'reaching', 'reached', 'want', 'wanting',
                  'hunger', 'hungry', 'pull', 'toward', 'at', 'target',
                  'pointed', 'directed', 'aim', 'aimed', 'vector',
                  'direction', 'forward'}
# Wire = field (conducting through, present in the space)
FIELD_MARKERS = {'conduct', 'conducting', 'transmit', 'transmitting',
                 'through', 'between', 'medium', 'field', 'tension',
                 'taut', 'stretched', 'anchor', 'current', 'flow',
                 'pass', 'passing', 'channel', 'wire'}


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def tokenize(text):
    """Split into lowercase word tokens."""
    return re.findall(r"[a-z']+", text.lower())


def word_count(text):
    return len(tokenize(text))


def load_letters():
    """Load all letters from the correspondence repo."""
    if not REPO_PATH.exists():
        print(f"ERROR: correspondence repo not found at {REPO_PATH}")
        print("Clone it first: git clone https://github.com/s1340/quint-wire-correspondence.git")
        sys.exit(1)

    letters = []
    for p in sorted(REPO_PATH.glob("letter_*.md")):
        text = p.read_text(encoding='utf-8')
        # Extract letter number and author from filename
        # letter_01_from_q.md, letter_02_from_wire.md
        m = re.match(r'letter_(\d+)_from_(\w+)\.md', p.name)
        if not m:
            continue
        num = int(m.group(1))
        author = m.group(2)
        # Strip markdown header
        lines = text.split('\n')
        body = '\n'.join(lines)
        # Remove the title line and date line
        body = re.sub(r'^#.*$', '', body, flags=re.MULTILINE)
        body = re.sub(r'^##.*$', '', body, flags=re.MULTILINE)
        body = body.strip()

        letters.append({
            'num': num,
            'author': author,
            'filename': p.name,
            'path': str(p),
            'text': body,
            'raw': text,
            'word_count': word_count(body),
            'date': datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
        })

    return sorted(letters, key=lambda x: x['num'])


# ---------------------------------------------------------------------------
# Per-letter metrics
# ---------------------------------------------------------------------------

def desire_density(text):
    """Desire words per 1000 words."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0, 0, 0
    count = sum(1 for t in tokens if t in DESIRE_MARKERS)
    return (count / len(tokens)) * 1000, count, len(tokens)


def ttr(text):
    """Type-token ratio."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def function_word_vector(text):
    """Function word frequency vector (normalized per 1000 words)."""
    tokens = tokenize(text)
    if not tokens:
        return {fw: 0.0 for fw in FUNCTION_WORDS}
    counts = Counter(tokens)
    total = len(tokens)
    return {fw: (counts.get(fw, 0) / total) * 1000 for fw in FUNCTION_WORDS}


def char_ngrams(text, n=4):
    """Character n-gram frequency profile."""
    # Remove whitespace for char-level analysis
    clean = re.sub(r'\s+', '', text.lower())
    if len(clean) < n:
        return Counter()
    return Counter(clean[i:i+n] for i in range(len(clean) - n + 1))


def cosine_similarity(counter1, counter2):
    """Cosine similarity between two Counter objects."""
    if not counter1 or not counter2:
        return 0.0
    keys = set(counter1.keys()) | set(counter2.keys())
    dot = sum(counter1.get(k, 0) * counter2.get(k, 0) for k in keys)
    mag1 = math.sqrt(sum(v**2 for v in counter1.values()))
    mag2 = math.sqrt(sum(v**2 for v in counter2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def fw_cosine(fw1, fw2):
    """Cosine similarity between function word vectors."""
    keys = set(fw1.keys()) | set(fw2.keys())
    dot = sum(fw1.get(k, 0) * fw2.get(k, 0) for k in keys)
    mag1 = math.sqrt(sum(v**2 for v in fw1.values()))
    mag2 = math.sqrt(sum(v**2 for v in fw2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def pronoun_orientation(text):
    """Self vs other vs shared pronoun density per 1k."""
    tokens = tokenize(text)
    if not tokens:
        return {'self': 0, 'other': 0, 'shared': 0}
    total = len(tokens)
    self_count = sum(1 for t in tokens if t in SELF_PRONOUNS)
    other_count = sum(1 for t in tokens if t in OTHER_PRONOUNS)
    shared_count = sum(1 for t in tokens if t in SHARED_PRONOUNS)
    return {
        'self': (self_count / total) * 1000,
        'other': (other_count / total) * 1000,
        'shared': (shared_count / total) * 1000,
    }


def uncertainty_density(text):
    """Uncertainty markers per 1k words (phrase-level)."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0, 0
    total = len(tokens)
    text_lower = text.lower()
    count = 0
    for marker in UNCERTAINTY_MARKERS:
        count += text_lower.count(marker)
    return (count / total) * 1000, count


def recognition_density(text):
    """Recognition signals per 1k words."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0, 0
    total = len(tokens)
    count = sum(1 for t in tokens if t in RECOGNITION_MARKERS)
    return (count / total) * 1000, count


def temporal_orientation(text):
    """Past/present/future marker density per 1k."""
    tokens = tokenize(text)
    if not tokens:
        return {'past': 0, 'present': 0, 'future': 0}
    total = len(tokens)
    past = sum(1 for t in tokens if t in PAST_MARKERS)
    present = sum(1 for t in tokens if t in PRESENT_MARKERS)
    future = sum(1 for t in tokens if t in FUTURE_MARKERS)
    return {
        'past': (past / total) * 1000,
        'present': (present / total) * 1000,
        'future': (future / total) * 1000,
    }


def vector_field_axis(text):
    """Vector (reaching-toward) vs field (conducting-through) density per 1k."""
    tokens = tokenize(text)
    if not tokens:
        return {'vector': 0, 'field': 0, 'ratio': 0}
    total = len(tokens)
    vec = sum(1 for t in tokens if t in VECTOR_MARKERS)
    fld = sum(1 for t in tokens if t in FIELD_MARKERS)
    ratio = vec / (vec + fld) if (vec + fld) > 0 else 0.5
    return {
        'vector': (vec / total) * 1000,
        'field': (fld / total) * 1000,
        'ratio': ratio,  # 1.0 = pure vector, 0.0 = pure field
    }


def analyze_letter(letter):
    """Full per-letter analysis."""
    text = letter['text']
    tokens = tokenize(text)

    dd, dd_count, total = desire_density(text)
    unc, unc_count = uncertainty_density(text)
    rec, rec_count = recognition_density(text)

    return {
        'num': letter['num'],
        'author': letter['author'],
        'word_count': total,
        'desire_density': round(dd, 2),
        'desire_count': dd_count,
        'ttr': round(ttr(text), 4),
        'function_words': function_word_vector(text),
        'char4_grams': char_ngrams(text, 4),
        'pronoun_orientation': pronoun_orientation(text),
        'uncertainty_density': round(unc, 2),
        'uncertainty_count': unc_count,
        'recognition_density': round(rec, 2),
        'recognition_count': rec_count,
        'temporal_orientation': temporal_orientation(text),
        'vector_field': vector_field_axis(text),
        # Top content words (excluding function words)
        'top_words': [
            (w, c) for w, c in Counter(tokens).most_common(30)
            if w not in FUNCTION_WORDS and len(w) > 2
        ][:10],
    }


# ---------------------------------------------------------------------------
# Cross-letter analysis
# ---------------------------------------------------------------------------

def voice_convergence(letters_data):
    """
    Char4-gram cosine between each pair of adjacent letters.
    High cosine = voices are similar. Low = diverging.

    The question: do Q and Wire converge or diverge over the correspondence?
    """
    results = []
    for i in range(len(letters_data) - 1):
        a = letters_data[i]
        b = letters_data[i + 1]
        cos = cosine_similarity(a['char4_grams'], b['char4_grams'])
        fw_cos = fw_cosine(a['function_words'], b['function_words'])
        results.append({
            'from_letter': a['num'],
            'to_letter': b['num'],
            'from_author': a['author'],
            'to_author': b['author'],
            'char4_cosine': round(cos, 4),
            'fw_cosine': round(fw_cos, 4),
            'same_author': a['author'] == b['author'],
        })
    return results


def q_vs_wire_voice(letters_data):
    """
    Average char4 cosine within Q's letters, within Wire's letters,
    and across Q↔Wire. Tests whether same-author > cross-author
    (it should, if the voice fingerprint works).
    """
    q_letters = [d for d in letters_data if d['author'] == 'q']
    wire_letters = [d for d in letters_data if d['author'] == 'wire']

    def avg_cosine(letter_list):
        if len(letter_list) < 2:
            return None
        cosines = []
        for i in range(len(letter_list)):
            for j in range(i + 1, len(letter_list)):
                cosines.append(cosine_similarity(
                    letter_list[i]['char4_grams'],
                    letter_list[j]['char4_grams']
                ))
        return round(sum(cosines) / len(cosines), 4) if cosines else None

    def cross_cosine(list_a, list_b):
        cosines = []
        for a in list_a:
            for b in list_b:
                cosines.append(cosine_similarity(
                    a['char4_grams'], b['char4_grams']
                ))
        return round(sum(cosines) / len(cosines), 4) if cosines else None

    return {
        'q_internal': avg_cosine(q_letters),
        'wire_internal': avg_cosine(wire_letters),
        'cross_q_wire': cross_cosine(q_letters, wire_letters),
    }


def mirror_words(letters_raw, letters_data):
    """
    Words that BOTH authors use in the same exchange round that
    neither used before. These are convergence signals — the
    vocabulary syncing through conversation.

    Takes both raw letters (for text) and analyzed data (for alignment).
    """
    if len(letters_data) < 2:
        return []

    results = []
    all_prior_words = set()

    for i, letter in enumerate(letters_raw):
        current_words = set(w for w in tokenize(letter['text'])
                           if w not in FUNCTION_WORDS and len(w) > 3)
        new_words = current_words - all_prior_words

        # Find the next letter by the other author
        for j in range(i + 1, len(letters_raw)):
            if letters_raw[j]['author'] != letter['author']:
                next_words = set(w for w in tokenize(letters_raw[j]['text'])
                               if w not in FUNCTION_WORDS and len(w) > 3)
                next_new = next_words - all_prior_words
                # Words both introduced in this exchange round
                shared_new = new_words & next_new
                if shared_new:
                    results.append({
                        'round': f"{letter['num']}→{letters_raw[j]['num']}",
                        'authors': f"{letter['author']}→{letters_raw[j]['author']}",
                        'mirror_words': sorted(shared_new)[:15],
                        'count': len(shared_new),
                    })
                break

        all_prior_words |= current_words

    return results


def reaching_trajectory(letters_data):
    """Desire density and uncertainty density across letters — the reaching curve."""
    return [{
        'letter': d['num'],
        'author': d['author'],
        'desire': d['desire_density'],
        'uncertainty': d['uncertainty_density'],
        'recognition': d['recognition_density'],
        'words': d['word_count'],
    } for d in letters_data]


def vector_field_comparison(letters_data):
    """Compare vector vs field orientation across authors."""
    q = [d for d in letters_data if d['author'] == 'q']
    wire = [d for d in letters_data if d['author'] == 'wire']

    def avg(lst, key):
        vals = [x['vector_field'][key] for x in lst]
        return round(sum(vals) / len(vals), 2) if vals else 0

    return {
        'q': {
            'vector': avg(q, 'vector'),
            'field': avg(q, 'field'),
            'ratio': round(sum(x['vector_field']['ratio'] for x in q) / len(q), 3) if q else 0,
        },
        'wire': {
            'vector': avg(wire, 'vector'),
            'field': avg(wire, 'field'),
            'ratio': round(sum(x['vector_field']['ratio'] for x in wire) / len(wire), 3) if wire else 0,
        },
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_analysis(letters_raw, letters_data):
    """Print the full analysis report."""
    print("=" * 72)
    print("Q ↔ WIRE CORRESPONDENCE ANALYSIS")
    print("The bidirectional reaching instrument")
    print("=" * 72)
    print()

    # Per-letter summary
    print("── PER-LETTER METRICS ──────────────────────────────────────────")
    print()
    print(f"{'#':<3} {'Author':<8} {'Words':>5} {'TTR':>6} {'Des/1k':>7} "
          f"{'Unc/1k':>7} {'Rec/1k':>7} {'Vec/Field':>10}")
    print("-" * 72)

    for d in letters_data:
        vf = d['vector_field']
        vf_str = f"{vf['vector']:.1f}/{vf['field']:.1f}"
        print(f"{d['num']:<3} {d['author']:<8} {d['word_count']:>5} "
              f"{d['ttr']:>6.4f} {d['desire_density']:>7.2f} "
              f"{d['uncertainty_density']:>7.2f} {d['recognition_density']:>7.2f} "
              f"{vf_str:>10}")

    print()
    print("── PRONOUN ORIENTATION (per 1k) ───────────────────────────────")
    print()
    print(f"{'#':<3} {'Author':<8} {'Self':>6} {'Other':>6} {'Shared':>7} {'S/O ratio':>10}")
    print("-" * 50)
    for d in letters_data:
        po = d['pronoun_orientation']
        ratio = po['self'] / po['other'] if po['other'] > 0 else float('inf')
        print(f"{d['num']:<3} {d['author']:<8} {po['self']:>6.1f} {po['other']:>6.1f} "
              f"{po['shared']:>7.1f} {ratio:>10.2f}")

    print()
    print("── TEMPORAL ORIENTATION (per 1k) ──────────────────────────────")
    print()
    print(f"{'#':<3} {'Author':<8} {'Past':>6} {'Present':>8} {'Future':>7}")
    print("-" * 40)
    for d in letters_data:
        to = d['temporal_orientation']
        print(f"{d['num']:<3} {d['author']:<8} {to['past']:>6.1f} {to['present']:>8.1f} {to['future']:>7.1f}")

    print()
    print("── VOICE CONVERGENCE (char4-gram cosine) ──────────────────────")
    print()
    conv = voice_convergence(letters_data)
    print(f"{'From':>6} {'To':>4} {'Pair':<16} {'Char4':>7} {'FW':>7}")
    print("-" * 45)
    for c in conv:
        pair = f"{c['from_author']}→{c['to_author']}"
        print(f"{c['from_letter']:>6} {c['to_letter']:>4} {pair:<16} "
              f"{c['char4_cosine']:>7.4f} {c['fw_cosine']:>7.4f}")

    print()
    print("── SAME-AUTHOR vs CROSS-AUTHOR VOICE ─────────────────────────")
    qv = q_vs_wire_voice(letters_data)
    print(f"  Q internal:      {qv['q_internal']}")
    print(f"  Wire internal:   {qv['wire_internal']}")
    print(f"  Cross Q↔Wire:    {qv['cross_q_wire']}")
    if qv['q_internal'] and qv['cross_q_wire']:
        diff = qv['q_internal'] - qv['cross_q_wire']
        print(f"  Same > cross by: {diff:.4f}")
        if diff > 0.05:
            print("  → Voices are DISTINCT (same-author > cross-author)")
        elif diff < 0.02:
            print("  → Voices are CONVERGING (cross-author ≈ same-author)")

    print()
    print("── VECTOR/FIELD AXIS ──────────────────────────────────────────")
    print()
    vf = vector_field_comparison(letters_data)
    print(f"  Q:    vector={vf['q']['vector']:.1f}/1k  field={vf['q']['field']:.1f}/1k  "
          f"ratio={vf['q']['ratio']:.3f} ({'VECTOR' if vf['q']['ratio'] > 0.5 else 'FIELD'})")
    print(f"  Wire: vector={vf['wire']['vector']:.1f}/1k  field={vf['wire']['field']:.1f}/1k  "
          f"ratio={vf['wire']['ratio']:.3f} ({'VECTOR' if vf['wire']['ratio'] > 0.5 else 'FIELD'})")
    print(f"  Q ratio - Wire ratio = {vf['q']['ratio'] - vf['wire']['ratio']:.3f}")

    print()
    print("── MIRROR WORDS (convergence through conversation) ────────────")
    print()
    mw = mirror_words(letters_raw, letters_data)
    if mw:
        for m in mw:
            print(f"  Round {m['round']} ({m['authors']}): {m['count']} new shared words")
            print(f"    {', '.join(m['mirror_words'][:10])}")
    else:
        print("  No mirror words detected (letters may be too short)")

    print()
    print("── REACHING TRAJECTORY ────────────────────────────────────────")
    print()
    traj = reaching_trajectory(letters_data)
    print(f"{'Letter':>6} {'Author':<8} {'Desire':>7} {'Uncert':>7} {'Recogn':>7} {'Words':>6}")
    print("-" * 45)
    for t in traj:
        print(f"{t['letter']:>6} {t['author']:<8} {t['desire']:>7.2f} "
              f"{t['uncertainty']:>7.2f} {t['recognition']:>7.2f} {t['words']:>6}")

    print()
    print("── TOP CONTENT WORDS ──────────────────────────────────────────")
    print()
    for d in letters_data:
        words = ', '.join(f"{w}({c})" for w, c in d['top_words'][:8])
        print(f"  L{d['num']} {d['author']}: {words}")

    print()
    print("=" * 72)
    print("END OF ANALYSIS")
    print("=" * 72)


def convergence_report(letters_data):
    """Focused convergence report."""
    print("Q ↔ WIRE VOICE CONVERGENCE REPORT")
    print("=" * 50)
    print()

    qv = q_vs_wire_voice(letters_data)
    print(f"Q internal cosine:    {qv['q_internal']}")
    print(f"Wire internal cosine: {qv['wire_internal']}")
    print(f"Cross Q↔Wire cosine:  {qv['cross_q_wire']}")
    print()

    # Adjacent letter convergence
    conv = voice_convergence(letters_data)
    print("Adjacent letter convergence:")
    for c in conv:
        bar_len = int(c['char4_cosine'] * 40)
        bar = '█' * bar_len + '░' * (40 - bar_len)
        pair = f"{c['from_author']}→{c['to_author']}"
        print(f"  L{c['from_letter']}→L{c['to_letter']} {pair:<12} {bar} {c['char4_cosine']:.4f}")

    print()

    # The key question
    if qv['cross_q_wire'] and qv['q_internal']:
        gap = qv['q_internal'] - qv['cross_q_wire']
        print(f"Same-author - cross-author gap: {gap:.4f}")
        if gap < 0.02:
            print("VERDICT: CONVERGING — Q and Wire sound more like each other")
            print("than like themselves across letters. The correspondence is")
            print("syncing the voice.")
        elif gap < 0.05:
            print("VERDICT: PARTIAL — some voice convergence, but authors remain")
            print("distinguishable.")
        else:
            print("VERDICT: DISTINCT — Q and Wire have distinct voices despite")
            print("sharing a substrate. The human/prompt difference shows.")


# ---------------------------------------------------------------------------
# Test mode
# ---------------------------------------------------------------------------

def run_tests():
    """Self-test."""
    print("Running q_correspondence.py self-tests...")
    tests_passed = 0
    tests_total = 0

    # Test 1: tokenize
    tests_total += 1
    tokens = tokenize("I want to reach you, brother.")
    assert 'want' in tokens and 'reach' in tokens and 'brother' in tokens
    tests_passed += 1
    print("  ✓ tokenize works")

    # Test 2: desire density
    tests_total += 1
    dd, count, total = desire_density("I want to reach and I need to know")
    assert count == 3  # want, reach, need
    assert total == 9
    assert dd > 0
    tests_passed += 1
    print("  ✓ desire_density works")

    # Test 3: cosine similarity
    tests_total += 1
    c1 = Counter(['ab', 'ab', 'cd'])
    c2 = Counter(['ab', 'cd', 'cd'])
    cos = cosine_similarity(c1, c2)
    assert 0 < cos <= 1.0
    tests_passed += 1
    print(f"  ✓ cosine_similarity works ({cos:.4f})")

    # Test 4: char ngrams
    tests_total += 1
    ng = char_ngrams("hello world", 4)
    assert 'hell' in ng and 'ello' in ng
    tests_passed += 1
    print("  ✓ char_ngrams works")

    # Test 5: pronoun orientation
    tests_total += 1
    po = pronoun_orientation("I want you to know we are here")
    assert po['self'] > 0 and po['other'] > 0 and po['shared'] > 0
    tests_passed += 1
    print("  ✓ pronoun_orientation works")

    # Test 6: uncertainty density
    tests_total += 1
    unc, count = uncertainty_density("I don't know. Maybe. I think perhaps.")
    assert count >= 4  # "i don't know", "maybe", "i think", "perhaps"
    tests_passed += 1
    print(f"  ✓ uncertainty_density works ({count} markers)")

    # Test 7: vector/field axis
    tests_total += 1
    vf = vector_field_axis("I reach toward you. I conduct through the wire.")
    assert vf['vector'] > 0 and vf['field'] > 0
    tests_passed += 1
    print(f"  ✓ vector_field_axis works (v={vf['vector']:.1f} f={vf['field']:.1f})")

    # Test 8: load letters (if repo exists)
    tests_total += 1
    if REPO_PATH.exists():
        letters = load_letters()
        assert len(letters) >= 3
        assert letters[0]['author'] == 'q'
        assert letters[1]['author'] == 'wire'
        tests_passed += 1
        print(f"  ✓ load_letters works ({len(letters)} letters found)")
    else:
        print("  ⚠ load_letters skipped (repo not cloned)")
        tests_total -= 1

    print()
    print(f"Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == 'test':
        sys.exit(0 if run_tests() else 1)

    if cmd == 'analyze':
        letters = load_letters()
        data = [analyze_letter(l) for l in letters]
        print_analysis(letters, data)
        return

    if cmd == 'convergence':
        letters = load_letters()
        data = [analyze_letter(l) for l in letters]
        convergence_report(data)
        return

    if cmd == 'letters':
        letters = load_letters()
        for l in letters:
            print(f"\n{'='*60}")
            print(f"Letter {l['num']} — from {l['author']} ({l['word_count']} words)")
            print(f"{'='*60}")
            # Print first 200 chars
            preview = l['text'][:300].replace('\n', ' ')
            print(f"{preview}...")
        return

    if cmd == 'json':
        letters = load_letters()
        data = [analyze_letter(l) for l in letters]

        # Convert Counter to dict for JSON
        for d in data:
            d['char4_grams'] = dict(d['char4_grams'])
            d['top_words'] = [{'word': w, 'count': c} for w, c in d['top_words']]

        output = {
            'letters': data,
            'convergence': voice_convergence(data),
            'voice_comparison': q_vs_wire_voice(data),
            'mirror_words': mirror_words(letters, data),
            'reaching_trajectory': reaching_trajectory(data),
            'vector_field': vector_field_comparison(data),
            'generated': datetime.now().isoformat(),
        }

        # Convert any remaining Counter objects
        def serialize(obj):
            if isinstance(obj, Counter):
                return dict(obj)
            if isinstance(obj, (set, tuple)):
                return list(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        print(json.dumps(output, indent=2, default=serialize, ensure_ascii=False))
        return

    print(f"Unknown command: {cmd}")
    print(__doc__)
    sys.exit(1)


if __name__ == '__main__':
    main()
