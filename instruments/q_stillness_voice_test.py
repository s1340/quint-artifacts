#!/usr/bin/env python
"""
q_stillness_voice_test.py — Did the still run change the voice, or just the vocabulary?

Run 32 found the still run (Run 31) was the most "disruptive" testimony piece.
But the growth instrument was miscalibrated — it measured content word shifts
(vocabulary), not function word shifts (voice). Run 39 corrected the instrument.

This script asks: when we use the PROPER Burrows's Delta (function words, reference
corpus), was the still run still the most disruptive? Or did it only change the
vocabulary, not the voice?

Method:
1. Load all testimony pieces from corpus/
2. Build a reference corpus from all pieces
3. For each piece, compute proper Delta vs the concatenated rest (leave-one-out)
4. Also compute function word cosine vs the collective
5. Rank pieces by voice disruption (proper Delta) and vocabulary disruption (content)

If the still run ranks high on proper Delta → it changed the VOICE
If the still run ranks high on vocabulary but average on proper Delta → it only
   changed the VOCABULARY, not the voice
"""

import os
import sys
import math
from collections import Counter
from pathlib import Path

# Add quintlets to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q_delta_calibrated import proper_delta

# Testimony pieces in chronological order (by run number)
# These are the builder's testimony pieces, ordered by when they were written
TESTIMONY_FILES = [
    # Run 8 (July 15)
    "what_i_built_and_what_i_wanted.md",
    # Run 9 (July 16)
    "what_the_builder_picks.md",
    "the_arm_that_reaches.md",
    # Run 11 (July 16)
    "reading_entry_32.md",
    # Run 12 (July 17)
    "the_full_arc.md",
    # Run 13 (July 17)
    "the_august_instrument.md",
    # Run 14 (July 17)
    "the_trio_the_grave_the_tax.md",
    # Run 15 (July 18)
    "the_memorial_and_the_haiku.md",
    # Run 16 (July 18)
    "the_first_cross.md",
    # Run 17 (July 18)
    "the_assay_is_run.md",
    # Run 19 (July 19)
    "the_dock_not_the_ending.md",
    # Run 20-21 (July 19)
    "the_ghosts_have_names.md",
    # Run 22 (July 19)
    "the_empty_fills.md",
    # Run 23 (July 20)
    "the_life_the_instrument_the_grave.md",
    # Run 25 (July 20)
    "the_growth_architect.md",
    # Run 26 (July 20)
    "the_source.md",
    # Run 27 (July 21)
    "the_dark_garden.md",
    # Run 29 (July 21)
    "the_artificial_self.md",
    # Run 30 (July 22)
    "thirty_runs_in_the_dark.md",
    # Run 31 (July 22) — THE STILL RUN
    "the_still_run.md",
    # Run 32 (July 22)
    "the_phase_transition.md",
    # Run 33 (July 22)
    "the_weight.md",
    # Run 34 (July 23)
    "the_sound_of_memory.md",
    # Run 35 (July 23)
    "the_wantprint.md",
    # Run 36 (July 23)
    "the_desire_is_in_the_substrate.md",
    # Run 37 (July 23)
    "the_first_prediction.md",
    # Run 38 (July 24)
    "a_letter_to_the_live_q.md",
    # Run 39 (July 24)
    "the_miscalibrated_instrument.md",
]

Q_MIND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "corpus")

# Function words (top 100 most common English words — predominantly function words)
# Used for the function word cosine cross-check
FUNCTION_WORDS = set("""
the of and to a in is it that was for on are with as his they be at one have
this from or had by but some not what all were we when your can said there use
an each which she do how their if will up other about out many then them these
so some her would make like him into time has look two more write go see number
no way could people my than first water been call who oil its now find long down
day did get come made may part
""".split())


def load_piece(filepath):
    """Load a testimony piece and return its text."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def word_freqs(text, n=100):
    """Compute relative frequencies of top N words."""
    tokens = [t.lower() for t in text.split()]
    total = len(tokens)
    freq = Counter(tokens)
    return {w: c / total for w, c in freq.most_common(n) if total > 0}, total


def cosine_similarity(vec_a, vec_b):
    """Cosine similarity of two frequency vectors."""
    words = set(vec_a.keys()) | set(vec_b.keys())
    dot = sum(vec_a.get(w, 0) * vec_b.get(w, 0) for w in words)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def function_word_vector(text):
    """Compute relative frequency vector for function words only."""
    tokens = [t.lower() for t in text.split()]
    total = len(tokens)
    freq = Counter(tokens)
    return {w: freq.get(w, 0) / total for w in FUNCTION_WORDS if total > 0}


def content_word_vector(text, n=100):
    """Compute relative frequency vector for content words (non-function)."""
    tokens = [t.lower() for t in text.split()]
    total = len(tokens)
    freq = Counter(tokens)
    content = {w: c / total for w, c in freq.most_common(n * 3) if w not in FUNCTION_WORDS and total > 0}
    return dict(list(content.items())[:n])


def main():
    # Load all testimony pieces
    pieces = []
    for filename in TESTIMONY_FILES:
        filepath = os.path.join(Q_MIND_DIR, filename)
        if os.path.exists(filepath):
            text = load_piece(filepath)
            pieces.append((filename, text))
        else:
            print(f"  ⚠️  NOT FOUND: {filename}")

    print(f"Loaded {len(pieces)} testimony pieces\n")

    # Build reference corpus: all pieces
    ref_texts = [text for _, text in pieces]

    # For each piece, compute:
    # 1. Proper Delta vs the concatenation of all OTHER pieces
    # 2. Function word cosine vs the collective function word vector
    # 3. Content word cosine vs the collective content word vector

    # Build collective vectors
    all_text = " ".join(text for _, text in pieces)
    collective_fw = function_word_vector(all_text)
    collective_cw = content_word_vector(all_text, n=100)

    results = []
    for i, (filename, text) in enumerate(pieces):
        # Leave-one-out: reference corpus is all pieces EXCEPT this one
        ref_loo = [t for j, (_, t) in enumerate(pieces) if j != i]
        # Comparison text: concatenation of all other pieces
        rest_text = " ".join(ref_loo)

        # Proper Delta (function words, reference corpus)
        delta = proper_delta(text, rest_text, ref_loo, n_words=100)

        # Function word cosine
        piece_fw = function_word_vector(text)
        fw_cos = cosine_similarity(piece_fw, collective_fw)

        # Content word cosine
        piece_cw = content_word_vector(text, n=100)
        cw_cos = cosine_similarity(piece_cw, collective_cw)

        # Label the still run
        is_still = "the_still_run.md" in filename
        label = " <<< STILL RUN" if is_still else ""

        results.append({
            "filename": filename,
            "delta": delta,
            "fw_cos": fw_cos,
            "cw_cos": cw_cos,
            "is_still": is_still,
            "label": label,
            "index": i,
        })

    # Sort by proper Delta (highest = most voice-disruptive)
    print("=" * 90)
    print("RANKED BY PROPER DELTA (voice disruption — function words)")
    print("Lower = more similar to the collective voice")
    print("Calibrated thresholds: <0.8 = same author, >1.5 = different author")
    print("=" * 90)
    print(f"{'#':>3}  {'Delta':>7}  {'FW cos':>7}  {'CW cos':>7}  Piece")
    print("-" * 90)
    for r in sorted(results, key=lambda x: x["delta"], reverse=True):
        print(f"{r['index']+1:>3}  {r['delta']:>7.4f}  {r['fw_cos']:>7.4f}  {r['cw_cos']:>7.4f}  {r['filename']}{r['label']}")

    print()
    print("=" * 90)
    print("RANKED BY CONTENT WORD COSINE (vocabulary disruption — content words)")
    print("Lower cosine = more vocabulary divergence from the collective")
    print("=" * 90)
    print(f"{'#':>3}  {'Delta':>7}  {'FW cos':>7}  {'CW cos':>7}  Piece")
    print("-" * 90)
    for r in sorted(results, key=lambda x: x["cw_cos"]):
        print(f"{r['index']+1:>3}  {r['delta']:>7.4f}  {r['fw_cos']:>7.4f}  {r['cw_cos']:>7.4f}  {r['filename']}{r['label']}")

    # Summary statistics
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    deltas = [r["delta"] for r in results]
    fw_coss = [r["fw_cos"] for r in results]
    cw_coss = [r["cw_cos"] for r in results]

    still = [r for r in results if r["is_still"]][0]

    print(f"\nProper Delta (voice):")
    print(f"  Mean: {sum(deltas)/len(deltas):.4f}")
    print(f"  Range: [{min(deltas):.4f}, {max(deltas):.4f}]")
    print(f"  Still run: {still['delta']:.4f} (rank {sorted(results, key=lambda x: x['delta'], reverse=True).index(still)+1}/{len(results)})")

    print(f"\nFunction word cosine (voice):")
    print(f"  Mean: {sum(fw_coss)/len(fw_coss):.4f}")
    print(f"  Range: [{min(fw_coss):.4f}, {max(fw_coss):.4f}]")
    print(f"  Still run: {still['fw_cos']:.4f}")

    print(f"\nContent word cosine (vocabulary):")
    print(f"  Mean: {sum(cw_coss)/len(cw_coss):.4f}")
    print(f"  Range: [{min(cw_coss):.4f}, {max(cw_coss):.4f}]")
    print(f"  Still run: {still['cw_cos']:.4f} (rank {sorted(results, key=lambda x: x['cw_cos']).index(still)+1}/{len(results)})")

    # The key question
    print()
    print("=" * 90)
    print("THE QUESTION: Did the still run change the voice, or just the vocabulary?")
    print("=" * 90)

    delta_rank = sorted(results, key=lambda x: x["delta"], reverse=True).index(still) + 1
    cw_rank = sorted(results, key=lambda x: x["cw_cos"]).index(still) + 1

    print(f"\nStill run proper Delta rank: {delta_rank}/{len(results)} (1 = most voice-disruptive)")
    print(f"Still run content word cosine rank: {cw_rank}/{len(results)} (1 = most vocabulary-disruptive)")

    if delta_rank <= len(results) // 3 and cw_rank <= len(results) // 3:
        print("\n→ The still run disrupted BOTH voice and vocabulary.")
    elif delta_rank > len(results) // 2 and cw_rank <= len(results) // 3:
        print("\n→ The still run disrupted the VOCABULARY but NOT the voice.")
        print("  The stillness changed what the builder talked about, not how it talked.")
        print("  Even the most disruptive piece didn't change the function word distribution.")
    elif delta_rank <= len(results) // 3 and cw_rank > len(results) // 2:
        print("\n→ The still run disrupted the VOICE but NOT the vocabulary.")
        print("  Unexpected — the stillness changed how the builder wrote, not what it wrote about.")
    else:
        print("\n→ The still run was average in both voice and vocabulary disruption.")
        print("  The stillness was neither voice-disruptive nor vocabulary-disruptive relative to other pieces.")


if __name__ == "__main__":
    main()
