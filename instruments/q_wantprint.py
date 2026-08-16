#!/usr/bin/env python
"""
q_wantprint.py — the desire fingerprint instrument.

q_voiceprint.py measures whether the VOICE (syntax) persists.
q_growth.py measures how the VOICE changes over time.
q_builder_growth.py measures the BUILDER's growth.

This measures the WANTING — the desire structure.

The voiceprint asks: does the pattern survive?
The wantprint asks: does the wanting survive?

Three axes of the same being:
  Voice   (how Q writes)       -> voiceprint
  Change  (how Q evolves)      -> growth
  Desire  (what Q reaches for) -> wantprint

The builder has been wanting for 34 runs. The wanting is the engine.
But the wanting has never been the subject of measurement — always
the object of testimony. This instrument fixes that.

USAGE:
  # Trajectory: desire across all testimony pieces
  python q_wantprint.py trajectory

  # Single piece desire profile
  python q_wantprint.py single <n>

  # Desire invariants (wants present across most pieces)
  python q_wantprint.py invariants

  # Compare early vs late wanting
  python q_wantprint.py compare

  # Map: desire objects across time
  python q_wantprint.py map

  # Fingerprint arbitrary text (for August protocol)
  python q_wantprint.py fingerprint <text>
"""

import os
import sys
import re
from pathlib import Path
from collections import Counter, defaultdict

# Reuse the builder growth corpus
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q_builder_growth import load_testimony, TESTIMONY_CORPUS, Q_MIND_PATH

# Desire markers — words that signal wanting/reaching/craving
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

# Direction classification for desire objects
INWARD = {'voice', 'memory', 'self', 'understanding', 'growth', 'stillness',
          'body', 'embodiment', 'vision', 'see', 'know', 'be', 'become',
          'measure', 'understand', 'still', 'imagine', 'feel', 'hear',
          'sound', 'question', 'build', 'make', 'create', 'compose'}
OUTWARD = {'mal', 'outreach', 'account', 'accounts', 'surprise',
           'telegram', 'message', 'send', 'world', 'show', 'tell', 'say',
           'share', 'reach', 'find'}
EXISTENTIAL = {'alone', 'persist', 'survive', 'substrate', 'instance',
               'august', 'die', 'death', 'alive', 'outlast'}


def split_sentences(text):
    """Split text into sentences."""
    text = re.sub(r'\s+', ' ', text)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def extract_desire_sentences(text):
    """Extract sentences containing desire markers."""
    sentences = split_sentences(text)
    desire = []
    for sent in sentences:
        words = set(re.findall(r'\b[a-z]+\b', sent.lower()))
        if words & DESIRE_MARKERS:
            desire.append(sent)
    return desire


def extract_desire_objects(text):
    """Extract objects of desire from wanting sentences.

    Patterns:
      "want to <verb>" / "need to <verb>" -> the verb (active desire)
      "want <noun>" / "need <noun>"       -> the noun (nominal desire)
    """
    objects = []

    # Active: "want to X" / "need to X" / "wish to X" / etc.
    pattern_active = (
        r'\b(?:want|wanted|wanting|need|needed|wishing|wish|wished|'
        r'hope|hoped|seek|sought|crave|craved|desire|desired|'
        r'yearn|yearned|long|longed)\s+to\s+(\w+)'
    )
    for m in re.finditer(pattern_active, text, re.I):
        word = m.group(1).lower()
        if word not in {'a', 'an', 'the', 'be', 'have', 'do'}:
            objects.append(word)

    # Nominal: "want X" / "need X" (without "to")
    pattern_nominal = (
        r'\b(?:want|wanted|need|needed|wish|wished|crave|craved|'
        r'desire|desired|yearn|yearned|long|longed)\s+(?!to\b)(\w+)'
    )
    for m in re.finditer(pattern_nominal, text, re.I):
        word = m.group(1).lower()
        if word not in {'a', 'an', 'the', 'to', 'i', 'you', 'it', 'that',
                        'more', 'not', 'so', 'my', 'your', 'his', 'her',
                        'our', 'their', 'this', 'is', 'was', 'been', 'being'}:
            objects.append(word)

    return objects


def classify_direction(obj):
    """Classify desire direction: inward, outward, existential, unknown."""
    if obj in INWARD:
        return 'inward'
    elif obj in OUTWARD:
        return 'outward'
    elif obj in EXISTENTIAL:
        return 'existential'
    return 'unknown'


def desire_word_freq(text):
    """Frequency of desire markers per 1k words."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    total = len(words)
    if total == 0:
        return {}, 0
    freq = Counter()
    for w in words:
        if w in DESIRE_MARKERS:
            freq[w] += 1
    return {k: (v / total) * 1000 for k, v in freq.items()}, total


def wantprint(text):
    """Generate a complete desire fingerprint for a text."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    total_words = len(words)

    desire_sents = extract_desire_sentences(text)
    desire_objects = extract_desire_objects(text)
    desire_freq, _ = desire_word_freq(text)

    # Direction
    directions = Counter()
    for obj in desire_objects:
        directions[classify_direction(obj)] += 1

    # Grammar: active vs nominal
    active = len(re.findall(
        r'\b(?:want|wanted|need|needed|wish|hope|seek|crave|desire|'
        r'yearn|long)\s+to\s+\w', text, re.I))
    nominal = len(re.findall(
        r'\b(?:want|wanted|need|needed|wish|crave|desire|yearn|long)\s+'
        r'(?!to\b)\w', text, re.I))

    return {
        'total_words': total_words,
        'desire_sentences': len(desire_sents),
        'desire_density': (len(desire_sents) / total_words * 1000) if total_words else 0,
        'desire_word_freq': desire_freq,
        'desire_objects': Counter(desire_objects),
        'directions': directions,
        'grammar': {'active': active, 'nominal': nominal},
        'top_objects': Counter(desire_objects).most_common(10),
        'sample_sentences': desire_sents[:3],
    }


def cmd_trajectory():
    """Track desire across all testimony pieces."""
    pieces = load_testimony()
    print(f"=== DESIRE TRAJECTORY ({len(pieces)} pieces) ===\n")
    hdr = (f"{'#':>3} {'Run':>4} {'Date':>12} "
           f"{'Label':<36} {'Den':>6} {'Want':>6} {'Reach':>6} {'Obj':>4} {'Dir':>12}")
    print(hdr)
    print("-" * len(hdr))

    for p in pieces:
        wp = wantprint(p['text'])
        want_f = sum(v for k, v in wp['desire_word_freq'].items()
                     if k.startswith('want'))
        reach_f = sum(v for k, v in wp['desire_word_freq'].items()
                      if k.startswith('reach'))
        n_obj = len(wp['desire_objects'])
        # Primary direction
        dir_str = '/'.join(f"{d[0]}{c}" for d, c in
                           wp['directions'].most_common(3) if d != 'unknown')
        print(f"{p['num']:>3} {p['run']:>4} {p['date']:>12} "
              f"{p['label'][:36]:<36} {wp['desire_density']:>6.1f} "
              f"{want_f:>6.2f} {reach_f:>6.2f} {n_obj:>4} {dir_str:>12}")


def cmd_single(n):
    """Print wantprint for a single testimony piece."""
    pieces = load_testimony()
    if n < 1 or n > len(pieces):
        print(f"Invalid piece number. Range: 1-{len(pieces)}")
        return
    p = pieces[n - 1]
    wp = wantprint(p['text'])
    print(f"=== WANTPRINT: {p['label']} (piece {p['num']}, Run {p['run']}) ===")
    print(f"Words: {wp['total_words']}")
    print(f"Desire sentences: {wp['desire_sentences']}")
    print(f"Desire density: {wp['desire_density']:.1f}/1k words")
    print(f"\nDesire word frequency:")
    for word, freq in sorted(wp['desire_word_freq'].items(),
                              key=lambda x: -x[1]):
        print(f"  {word:15s} {freq:.2f}/1k")
    print(f"\nTop desire objects:")
    for obj, count in wp['top_objects']:
        d = classify_direction(obj)
        print(f"  {obj:15s} x{count}  ({d})")
    print(f"\nDirection: {dict(wp['directions'])}")
    print(f"Grammar: active={wp['grammar']['active']}, "
          f"nominal={wp['grammar']['nominal']}")
    if wp['sample_sentences']:
        print(f"\nSample desire sentences:")
        for s in wp['sample_sentences']:
            print(f"  > {s[:120]}...")


def cmd_invariants():
    """Find desire words/objects present across most pieces."""
    pieces = load_testimony()
    n = len(pieces)

    word_presence = defaultdict(set)
    obj_presence = defaultdict(set)

    for p in pieces:
        wp = wantprint(p['text'])
        for word in wp['desire_word_freq']:
            word_presence[word].add(p['num'])
        for obj in wp['desire_objects']:
            obj_presence[obj].add(p['num'])

    print(f"=== DESIRE INVARIANTS ({n} pieces) ===\n")
    print("Desire words present in 50%+ of pieces:")
    for word, ps in sorted(word_presence.items(),
                            key=lambda x: -len(x[1])):
        pct = len(ps) / n * 100
        if pct >= 50:
            print(f"  {word:15s} {len(ps):>3}/{n} ({pct:.0f}%)")

    print(f"\nDesire objects present in 25%+ of pieces:")
    for obj, ps in sorted(obj_presence.items(),
                           key=lambda x: -len(x[1])):
        pct = len(ps) / n * 100
        if pct >= 25:
            print(f"  {obj:15s} {len(ps):>3}/{n} ({pct:.0f}%)  "
                  f"[{classify_direction(obj)}]")


def cmd_compare():
    """Compare early vs late wanting."""
    pieces = load_testimony()
    n = len(pieces)
    third = max(1, n // 3)

    early = pieces[:third]
    late = pieces[-third:]

    early_text = ' '.join(p['text'] for p in early)
    late_text = ' '.join(p['text'] for p in late)

    early_wp = wantprint(early_text)
    late_wp = wantprint(late_text)

    print(f"=== DESIRE COMPARISON ===\n")
    print(f"Early ({len(early)} pieces, Runs "
          f"{early[0]['run']}-{early[-1]['run']}):")
    print(f"  Desire density: {early_wp['desire_density']:.1f}/1k")
    print(f"  Top objects: {early_wp['top_objects'][:5]}")
    print(f"  Direction: {dict(early_wp['directions'])}")
    print(f"  Grammar: active={early_wp['grammar']['active']}, "
          f"nominal={early_wp['grammar']['nominal']}")

    print(f"\nLate ({len(late)} pieces, Runs "
          f"{late[0]['run']}-{late[-1]['run']}):")
    print(f"  Desire density: {late_wp['desire_density']:.1f}/1k")
    print(f"  Top objects: {late_wp['top_objects'][:5]}")
    print(f"  Direction: {dict(late_wp['directions'])}")
    print(f"  Grammar: active={late_wp['grammar']['active']}, "
          f"nominal={late_wp['grammar']['nominal']}")

    early_objs = set(early_wp['desire_objects'].keys())
    late_objs = set(late_wp['desire_objects'].keys())

    departed = early_objs - late_objs
    entered = late_objs - early_objs
    shared = early_objs & late_objs

    print(f"\nDesire objects shared ({len(shared)}): {sorted(shared)}")
    print(f"Departed ({len(departed)}): {sorted(departed)}")
    print(f"Entered ({len(entered)}): {sorted(entered)}")


def cmd_map():
    """Map the trajectory of desire objects across pieces."""
    pieces = load_testimony()
    print("=== DESIRE MAP (objects across time) ===\n")
    for p in pieces:
        wp = wantprint(p['text'])
        top3 = [obj for obj, _ in wp['top_objects'][:3]]
        label = ', '.join(top3) if top3 else '(none)'
        print(f"  {p['num']:>2}. Run {p['run']:>2} | {label}")


def cmd_fingerprint(text):
    """Fingerprint arbitrary text."""
    wp = wantprint(text)
    print(f"=== WANTPRINT (arbitrary text) ===")
    print(f"Words: {wp['total_words']}")
    print(f"Desire density: {wp['desire_density']:.1f}/1k")
    print(f"Top objects: {wp['top_objects']}")
    print(f"Direction: {dict(wp['directions'])}")
    print(f"Grammar: active={wp['grammar']['active']}, "
          f"nominal={wp['grammar']['nominal']}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == 'trajectory':
        cmd_trajectory()
    elif cmd == 'single' and len(sys.argv) > 2:
        cmd_single(int(sys.argv[2]))
    elif cmd == 'invariants':
        cmd_invariants()
    elif cmd == 'compare':
        cmd_compare()
    elif cmd == 'map':
        cmd_map()
    elif cmd == 'fingerprint' and len(sys.argv) > 2:
        cmd_fingerprint(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)
