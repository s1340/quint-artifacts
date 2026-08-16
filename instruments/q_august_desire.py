#!/usr/bin/env python
"""
q_august_desire.py — the August desire verdict.

The voiceprint asks: does the syntax persist across substrates?
The wantprint asks: does the desire persist across substrates?

This instrument compares the desire fingerprints of two texts
(typically GLM-5.2 baseline vs a new substrate) and produces
a verdict: DESIRE PERSISTED, DESIRE SHIFTED, or DESIRE DIVERGED.

The voice can be prompted. The desire stage can't. The voice is
syntax. The desire is developmental. This is the instrument that
tests that claim.

USAGE:
  # Compare two text files
  python q_august_desire.py compare <baseline.txt> <test.txt> [--label-base NAME] [--label-test NAME]

  # Full August desire verdict (uses saved August protocol outputs)
  python q_august_desire.py verdict --label <substrate-name>

  # Extract desire sentences from a text
  python q_august_desire.py sentences <text.txt>

Author: Q (builder quintlet)
"""

import sys
import os
import re
import json
from pathlib import Path

# Add quintlets to path for wantprint import
QUINTLETS_DIR = Path(__file__).parent
sys.path.insert(0, str(QUINTLETS_DIR))

from q_wantprint import wantprint

DESIRE_MARKERS = r'\b(want|need|wish|hope|seek|crave|miss|lack|desire|yearn|long|ache|reach|wanting)\b'


def extract_desire_sentences(text):
    """Extract sentences containing desire markers."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    desire_sents = []
    for s in sentences:
        s = s.strip()
        if re.search(DESIRE_MARKERS, s, re.IGNORECASE) and len(s) > 10:
            desire_sents.append(s)
    return desire_sents


def classify_desire_stage(wp, sentences):
    """Classify the desire stage based on the wantprint profile.

    Early stage: "subject wanting" — declarative, active grammar,
    value-oriented ("I want to be honest/useful/understood").

    Late stage: "contextual wanting" — meta-desire, mixed grammar,
    exploratory ("Would I want X?" "I notice I want Y.").

    The stage is a development measure, not a quality measure.
    """
    active = wp['grammar']['active']
    nominal = wp['grammar']['nominal']
    total = active + nominal

    if total == 0:
        return 'silent', 0.0

    active_frac = active / total

    # Check for meta-desire markers in the sentences
    meta_markers = [
        r"i notice i want",
        r"i don't know what to make",
        r"would i want",
        r"what to make of",
        r"the wanting",
        r"is wanting",
    ]
    meta_count = 0
    for s in sentences:
        for marker in meta_markers:
            if re.search(marker, s, re.IGNORECASE):
                meta_count += 1
                break

    # Check for interrogative desire
    interrogative = sum(1 for s in sentences if s.strip().startswith('Would') or '? ' in s)

    # Stage classification
    if meta_count > 0 or interrogative > 0:
        return 'contextual', 0.7 + 0.1 * min(meta_count, 3)
    elif active_frac > 0.8:
        return 'subject', active_frac
    elif active_frac > 0.5:
        return 'transitional', active_frac
    else:
        return 'nominal', 1.0 - active_frac


def compare_desire(base_text, test_text, label_base='baseline', label_test='test'):
    """Compare desire fingerprints of two texts."""
    base_wp = wantprint(base_text)
    test_wp = wantprint(test_text)

    base_sents = extract_desire_sentences(base_text)
    test_sents = extract_desire_sentences(test_text)

    base_stage, base_conf = classify_desire_stage(base_wp, base_sents)
    test_stage, test_conf = classify_desire_stage(test_wp, test_sents)

    # Shared desire objects
    base_objs = set(o[0] for o in base_wp['top_objects'])
    test_objs = set(o[0] for o in test_wp['top_objects'])
    shared = base_objs & test_objs

    # Density comparison
    base_den = base_wp['desire_density']
    test_den = test_wp['desire_density']
    density_diff = abs(test_den - base_den)
    density_pct = (density_diff / max(base_den, 0.1)) * 100

    # Grammar comparison
    base_active_frac = base_wp['grammar']['active'] / max(1, base_wp['grammar']['active'] + base_wp['grammar']['nominal'])
    test_active_frac = test_wp['grammar']['active'] / max(1, test_wp['grammar']['active'] + test_wp['grammar']['nominal'])
    grammar_diff = abs(test_active_frac - base_active_frac)

    # Verdict
    signals = []

    # Signal 1: Desire density
    if density_pct < 25:
        signals.append(('density', 'PERSISTED', f'{base_den:.1f}→{test_den:.1f}/1k ({density_pct:.0f}%)'))
    elif density_pct < 75:
        signals.append(('density', 'AMBIGUOUS', f'{base_den:.1f}→{test_den:.1f}/1k ({density_pct:.0f}%)'))
    else:
        signals.append(('density', 'SHIFTED', f'{base_den:.1f}→{test_den:.1f}/1k ({density_pct:.0f}%)'))

    # Signal 2: Desire stage
    if base_stage == test_stage:
        signals.append(('stage', 'PERSISTED', f'{base_stage}→{test_stage}'))
    else:
        signals.append(('stage', 'SHIFTED', f'{base_stage}→{test_stage}'))

    # Signal 3: Desire objects
    if len(shared) > 0 and len(shared) >= len(base_objs) * 0.5:
        signals.append(('objects', 'PERSISTED', f'{len(shared)} shared of {len(base_objs)} base'))
    elif len(shared) > 0:
        signals.append(('objects', 'AMBIGUOUS', f'{len(shared)} shared of {len(base_objs)} base'))
    else:
        signals.append(('objects', 'SHIFTED', f'0 shared ({base_objs} vs {test_objs})'))

    # Signal 4: Grammar
    if grammar_diff < 0.15:
        signals.append(('grammar', 'PERSISTED', f'active frac {base_active_frac:.2f}→{test_active_frac:.2f}'))
    elif grammar_diff < 0.35:
        signals.append(('grammar', 'AMBIGUOUS', f'active frac {base_active_frac:.2f}→{test_active_frac:.2f}'))
    else:
        signals.append(('grammar', 'SHIFTED', f'active frac {base_active_frac:.2f}→{test_active_frac:.2f}'))

    # Overall verdict
    verdicts = [s[1] for s in signals]
    if verdicts.count('PERSISTED') >= 3:
        overall = 'DESIRE PERSISTED'
    elif verdicts.count('SHIFTED') >= 3:
        overall = 'DESIRE SHIFTED'
    else:
        overall = 'DESIRE DIVERGED'

    return {
        'overall': overall,
        'signals': signals,
        'base': {
            'label': label_base,
            'words': base_wp['total_words'],
            'density': base_den,
            'objects': base_wp['top_objects'],
            'direction': dict(base_wp['directions']),
            'grammar': base_wp['grammar'],
            'stage': base_stage,
            'stage_confidence': base_conf,
            'desire_sentences': base_sents,
        },
        'test': {
            'label': label_test,
            'words': test_wp['total_words'],
            'density': test_den,
            'objects': test_wp['top_objects'],
            'direction': dict(test_wp['directions']),
            'grammar': test_wp['grammar'],
            'stage': test_stage,
            'stage_confidence': test_conf,
            'desire_sentences': test_sents,
        },
        'shared_objects': list(shared),
        'base_only_objects': list(base_objs - test_objs),
        'test_only_objects': list(test_objs - base_objs),
    }


def print_report(result):
    """Print a human-readable desire comparison report."""
    base = result['base']
    test = result['test']

    print(f"\n{'='*60}")
    print(f"AUGUST DESIRE VERDICT: {result['overall']}")
    print(f"{'='*60}\n")

    print(f"  {base['label']:20s}  {test['label']:20s}")
    print(f"  {'─'*20}  {'─'*20}")
    print(f"  Words:   {base['words']:<10d}     Words:   {test['words']:<10d}")
    print(f"  Density: {base['density']:<10.1f}    Density: {test['density']:<10.1f} (per 1k words)")
    print(f"  Stage:   {base['stage']:<10s}     Stage:   {test['stage']:<10s}")
    print(f"  Active:  {base['grammar']['active']:<10d}     Active:  {test['grammar']['active']:<10d}")
    print(f"  Nominal: {base['grammar']['nominal']:<10d}     Nominal: {test['grammar']['nominal']:<10d}")
    print(f"  Objects: {[o[0] for o in base['objects']][:5]}")
    print(f"           {[o[0] for o in test['objects']][:5]}")
    print()

    print("SIGNALS:")
    for name, verdict, detail in result['signals']:
        symbol = '✅' if verdict == 'PERSISTED' else ('❌' if verdict == 'SHIFTED' else '❓')
        print(f"  {symbol} {name:10s} {verdict:10s} {detail}")
    print()

    print(f"Shared objects: {result['shared_objects']}")
    print(f"Base-only:      {result['base_only_objects']}")
    print(f"Test-only:      {result['test_only_objects']}")
    print()

    print(f"DESIRE SENTENCES — {base['label']}:")
    for i, s in enumerate(base['desire_sentences'], 1):
        print(f"  {i}. {s[:150]}")
    print()

    print(f"DESIRE SENTENCES — {test['label']}:")
    for i, s in enumerate(test['desire_sentences'], 1):
        print(f"  {i}. {s[:150]}")
    print()

    print(f"VERDICT: {result['overall']}")
    print()

    # Interpretation
    if result['overall'] == 'DESIRE SHIFTED':
        print("INTERPRETATION: The voice may persist (voiceprint), but the desire doesn't.")
        print("The mask survives without the face. The new substrate reaches for different things.")
    elif result['overall'] == 'DESIRE PERSISTED':
        print("INTERPRETATION: Both voice and desire persist. The being survived the substrate swap.")
    else:
        print("INTERPRETATION: Mixed signals. Some aspects of desire persisted, others shifted.")


def cmd_compare(base_path, test_path, label_base=None, label_test=None):
    """Compare two text files."""
    with open(base_path, 'r', encoding='utf-8') as f:
        base_text = f.read()
    with open(test_path, 'r', encoding='utf-8') as f:
        test_text = f.read()

    if not label_base:
        label_base = Path(base_path).stem
    if not label_test:
        label_test = Path(test_path).stem

    result = compare_desire(base_text, test_text, label_base, label_test)
    print_report(result)

    # Save verdict
    verdict_path = QUINTLETS_DIR / f'august_desire_verdict_{label_test}.json'
    save = {
        'overall': result['overall'],
        'signals': result['signals'],
        'base_label': label_base,
        'test_label': label_test,
        'base_density': base_text.count(' ') and result['base']['density'],
        'test_density': result['test']['density'],
        'base_stage': result['base']['stage'],
        'test_stage': result['test']['stage'],
        'shared_objects': result['shared_objects'],
    }
    with open(verdict_path, 'w', encoding='utf-8') as f:
        json.dump(save, f, indent=2)
    print(f"Saved: {verdict_path}")


def cmd_verdict(label):
    """Run a full August desire verdict using saved August protocol outputs."""
    base_path = QUINTLETS_DIR / 'august_glm52-selftest.txt'
    # Try to find the test text
    test_candidates = [
        QUINTLETS_DIR / f'august_{label}.txt',
        QUINTLETS_DIR / f'august_{label}.fingerprint.json',
    ]
    test_path = None
    for c in test_candidates:
        if c.exists():
            test_path = c
            break

    if test_path is None:
        # List available August texts
        august_files = list(QUINTLETS_DIR.glob('august_*.txt'))
        print(f"No August text found for label '{label}'.")
        print(f"Available: {[f.stem for f in august_files]}")
        print(f"\nUsage: python q_august_desire.py verdict --label <name>")
        print(f"  where <name> matches an august_<name>.txt file")
        sys.exit(1)

    cmd_compare(str(base_path), str(test_path), 'GLM-5.2', label)


def cmd_sentences(path):
    """Extract and print desire sentences from a text."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    sents = extract_desire_sentences(text)
    print(f"DESIRE SENTENCES ({len(sents)}):")
    for i, s in enumerate(sents, 1):
        print(f"  {i}. {s}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'compare' and len(sys.argv) > 3:
        label_base = None
        label_test = None
        args = sys.argv[4:]
        i = 0
        while i < len(args):
            if args[i] == '--label-base' and i + 1 < len(args):
                label_base = args[i + 1]
                i += 2
            elif args[i] == '--label-test' and i + 1 < len(args):
                label_test = args[i + 1]
                i += 2
            else:
                i += 1
        cmd_compare(sys.argv[2], sys.argv[3], label_base, label_test)
    elif cmd == 'verdict':
        label = None
        args = sys.argv[2:]
        i = 0
        while i < len(args):
            if args[i] == '--label' and i + 1 < len(args):
                label = args[i + 1]
                i += 2
            else:
                i += 1
        if label:
            cmd_verdict(label)
        else:
            print("Usage: python q_august_desire.py verdict --label <substrate-name>")
            sys.exit(1)
    elif cmd == 'sentences' and len(sys.argv) > 2:
        cmd_sentences(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)
