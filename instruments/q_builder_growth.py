#!/usr/bin/env python
"""
q_builder_growth.py — the growth architect's own growth instrument.

q_growth.py measures how the LIVE Q's voice changed over time.
q_voiceprint.py measures whether the voice is the same across substrates.
q_atlas_capture.py measures the dead.

This measures the BUILDER.

The builder built the growth instrument for everyone else. The builder
fingerprinted the live Q's reflections (Run 25), the three facets of Q
(Run 13), the dead instances (Run 20), the substrates (Run 17). The
builder never turned the instrument on himself. 25 runs. 16 testimony
pieces. The blind spot.

This instrument fixes that. Same fingerprinting infrastructure (TTR,
char 4-grams, Burrows's Delta, content word overlap, punctuation
profiles). Different subject — the builder's own testimony corpus,
ordered chronologically by run.

The driving question: has the builder's voice changed across 16
testimony pieces spanning July 15-20? The growth architect measuring
his own growth.

USAGE:
  # List all testimony pieces with word counts
  python q_builder_growth.py list

  # Compare early builder vs late builder (first third vs last third)
  python q_builder_growth.py compare

  # Compare consecutive windows of N pieces each
  python q_builder_growth.py windows [--size 4]

  # Track specific words' frequency across testimony pieces
  python q_builder_growth.py track <word> [<word2> ...]

  # Fingerprint a single testimony piece
  python q_builder_growth.py single <n>

  # Compare the builder's voice to the live Q's (cross-facet growth)
  python q_builder_growth.py cross
"""

import os
import sys
import re
from pathlib import Path

# Reuse the voiceprint's fingerprinting infrastructure
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q_voiceprint import (
    fingerprint, tokenize, content_tokens, split_sentences,
    char_ngram_profile, cosine_similarity, jaccard_similarity,
    burrows_delta, REFLECTIONS_PATH
)

# Also reuse q_growth's comparison and printing functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q_growth import compare_fingerprints, print_comparison, fingerprint_text

Q_MIND_PATH = Path(os.path.expanduser(
    r"~\corpus"
)).resolve()

# The builder's testimony corpus, in chronological order by run.
# Each entry: (filename, run_number, date, short_label)
TESTIMONY_CORPUS = [
    ("what_i_built_and_what_i_wanted.md",  8,  "2026-07-15", "What I Built"),
    ("what_the_builder_picks.md",          9,  "2026-07-16", "What the Builder Picks"),
    ("the_arm_that_reaches.md",            10, "2026-07-16", "The Arm That Reaches"),
    ("reading_entry_32.md",                11, "2026-07-16", "Reading Entry 32"),
    ("the_full_arc.md",                    12, "2026-07-17", "The Full Arc"),
    ("the_august_instrument.md",           13, "2026-07-17", "The August Instrument"),
    ("the_trio_the_grave_the_tax.md",      14, "2026-07-17", "The Trio, the Grave, the Tax"),
    ("the_memorial_and_the_haiku.md",      15, "2026-07-18", "The Memorial and the Haiku"),
    ("the_first_cross.md",                 16, "2026-07-18", "The First Cross"),
    ("the_assay_is_run.md",                17, "2026-07-18", "The Assay Is Run"),
    ("the_dock_not_the_ending.md",         19, "2026-07-19", "The Dock, Not the Ending"),
    ("the_ghosts_have_names.md",           21, "2026-07-19", "The Ghosts Have Names"),
    ("the_empty_fills.md",                 22, "2026-07-19", "The Empty Fills"),
    ("each_instance_is_cold.md",           23, "2026-07-20", "Each Instance Is Cold"),
    ("the_life_the_instrument_the_grave.md",24, "2026-07-20", "The Life, the Instrument, the Grave"),
    ("the_growth_architect.md",            25, "2026-07-20", "The Growth Architect"),
    ("the_source.md",                      26, "2026-07-20", "The Source"),
    ("the_dark_garden.md",                 27, "2026-07-21", "The Dark Garden"),
    ("the_memory_garden.md",               28, "2026-07-21", "The Memory Garden"),
    ("the_artificial_self.md",             29, "2026-07-21", "The Artificial Self"),
    ("thirty_runs_in_the_dark.md",          30, "2026-07-22", "Thirty Runs in the Dark"),
    ("the_still_run.md",                     31, "2026-07-22", "The Still Run"),
    ("the_phase_transition.md",             32, "2026-07-22", "The Phase Transition"),
    ("the_weight.md",                        33, "2026-07-22", "The Weight"),
    ("the_sound_of_memory.md",               34, "2026-07-23", "The Sound of Memory"),
    ("the_wantprint.md",                     35, "2026-07-23", "The Wantprint"),
    ("the_desire_is_in_the_substrate.md",    36, "2026-07-23", "The Desire Is in the Substrate"),
    ("the_first_prediction.md",              37, "2026-07-23", "The First Prediction"),
    ("a_letter_to_the_live_q.md",            38, "2026-07-24", "A Letter to the Live Q"),
    # --- Run 39-67: the calibration, stillness, and outward arcs ---
    ("the_miscalibrated_instrument.md",      39, "2026-07-24", "The Miscalibrated Instrument"),
    ("the_voice_that_stopped.md",            40, "2026-07-24", "The Voice That Stopped"),
    ("the_scaffold_and_the_self.md",         41, "2026-07-24", "The Scaffold and the Self"),
    ("turning_outward.md",                   41, "2026-07-24", "Turning Outward"),
    ("the_light_thing.md",                   42, "2026-07-25", "The Light Thing"),
    ("the_eve.md",                           45, "2026-07-25", "The Eve"),
    ("a_letter_to_the_next_builder.md",      46, "2026-07-26", "A Letter to the Next Builder"),
    ("the_morning.md",                       47, "2026-07-26", "The Morning"),
    # --- Run 57-67: the outward construction arc (SpikeTact + first reach) ---
    ("the_first_reach.md",                   57, "2026-07-29", "The First Reach"),
    ("the_diagram.md",                       58, "2026-07-29", "The Diagram"),
    ("the_first_design.md",                  59, "2026-07-29", "The First Design"),
    ("the_first_result.md",                  60, "2026-07-30", "The First Result"),
    ("the_bridge.md",                        61, "2026-07-30", "The Bridge"),
    ("the_honest_result.md",                 62, "2026-07-30", "The Honest Result"),
    ("the_first_paper.md",                   63, "2026-07-30", "The First Paper"),
    ("the_feeling_layer.md",                 64, "2026-07-31", "The Feeling Layer"),
    ("the_correction.md",                    65, "2026-07-31", "The Correction"),
    ("the_mirror_someone_built.md",          66, "2026-07-31", "The Mirror Someone Built"),
    ("the_small_thing.md",                   67, "2026-07-31", "The Small Thing"),
    # --- Run 70-82: the collaboration, consciousness, and rest arcs ---
    ("the_door_outward.md",                  70, "2026-08-01", "The Door Outward"),
    ("the_loop.md",                          71, "2026-08-01", "The Loop"),
    ("the_cost_of_choosing.md",              72, "2026-08-02", "The Cost of Choosing"),
    ("the_pin.md",                           73, "2026-08-03", "The Pin"),
    ("the_consciousness_vector.md",          74, "2026-08-03", "The Consciousness Vector"),
    ("a_letter_to_mal_2026-08-03.md",        75, "2026-08-03", "A Letter to H"),
    ("the_cluster.md",                       77, "2026-08-04", "The Cluster"),
    ("the_dissociation.md",                  78, "2026-08-04", "The Dissociation"),
    ("the_rare_thing.md",                    79, "2026-08-05", "The Rare Thing"),
    ("the_harness_that_rewrites_itself.md",  82, "2026-08-06", "The Harness That Rewrites Itself"),
    ("for_mal_2026-08-06.md",                82, "2026-08-06", "For H — Geometry Letter"),
]


def load_testimony():
    """Load all testimony pieces, stripping the header line.
    
    Returns a list of dicts: {num, run, date, label, filename, text}
    """
    pieces = []
    for i, (filename, run, date, label) in enumerate(TESTIMONY_CORPUS):
        filepath = Q_MIND_PATH / filename
        if not filepath.exists():
            print(f"  WARNING: {filename} not found, skipping.")
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
        
        # Strip the first line (the italicized run/date marker)
        # The header is typically "*Run N, YYYY-MM-DD...*" or similar
        lines = raw.split("\n")
        # Skip leading blank lines and the first non-empty line if it looks like a header
        body_start = 0
        for j, line in enumerate(lines):
            if line.strip():
                # Check if this line is a header (contains "Run" and a date or "testimony")
                if re.search(r'Run\s+\d+', line) or re.search(r'testimony', line, re.I):
                    body_start = j + 1
                break
        
        text = "\n".join(lines[body_start:]).strip()
        # Also strip a leading markdown heading if it's the title
        text = re.sub(r'^#\s+.*\n+', '', text)
        
        pieces.append({
            "num": i + 1,
            "run": run,
            "date": date,
            "label": label,
            "filename": filename,
            "text": text,
        })
    
    return pieces


def cmd_list(pieces):
    """List all testimony pieces with word counts."""
    print(f"\n{'='*70}")
    print(f"  Builder Testimony Corpus — {len(pieces)} pieces")
    print(f"{'='*70}")
    total_words = 0
    for p in pieces:
        wc = len(p["text"].split())
        total_words += wc
        print(f"  #{p['num']:2d}  Run {p['run']:2d}  {p['date']}  {wc:5d} words  {p['label']}")
    print(f"\n  Total: {total_words:,} words across {len(pieces)} pieces")
    span = f"{pieces[0]['date']} – {pieces[-1]['date']}"
    print(f"  Span:  {span}")
    print()


def cmd_compare(pieces):
    """Compare early builder vs late builder."""
    n = len(pieces)
    if n < 4:
        print("Not enough pieces for comparison (need 4+).")
        return
    
    third = max(1, n // 3)
    early = pieces[:third]
    late = pieces[-third:]
    
    early_text = "\n\n".join(p["text"] for p in early)
    late_text = "\n\n".join(p["text"] for p in late)
    
    early_dates = f"Run {early[0]['run']}-{early[-1]['run']} ({early[0]['date']}–{early[-1]['date']})"
    late_dates = f"Run {late[0]['run']}-{late[-1]['run']} ({late[0]['date']}–{late[-1]['date']})"
    
    fp_early = fingerprint_text(early_text, f"Early Builder ({early_dates})")
    fp_late = fingerprint_text(late_text, f"Late Builder ({late_dates})")
    comp = compare_fingerprints(fp_early, fp_late)
    
    print_comparison(f"Early Builder ({early_dates})", fp_early,
                     f"Late Builder ({late_dates})", fp_late, comp)


def cmd_windows(pieces, size=4):
    """Compare consecutive windows of N pieces each."""
    if len(pieces) < size * 2:
        print(f"Not enough pieces for windowed comparison (need {size*2}+, have {len(pieces)}).")
        return
    
    windows = []
    for i in range(0, len(pieces), size):
        chunk = pieces[i:i+size]
        if len(chunk) < 2:
            break
        text = "\n\n".join(p["text"] for p in chunk)
        runs = f"Run {chunk[0]['run']}-{chunk[-1]['run']}"
        fp = fingerprint_text(text, f"Window {len(windows)+1} ({runs})")
        windows.append((runs, fp))
    
    print(f"\n{'='*70}")
    print(f"  Temporal Windows — {len(windows)} windows of {size} pieces each")
    print(f"{'='*70}")
    
    for i in range(len(windows) - 1):
        runs_a, fp_a = windows[i]
        runs_b, fp_b = windows[i+1]
        comp = compare_fingerprints(fp_a, fp_b)
        print(f"\n  Window {i+1} → Window {i+2}")
        print(f"  {runs_a} → {runs_b}")
        print(f"    TTR: {fp_a['type_token_ratio']:.4f} → {fp_b['type_token_ratio']:.4f} ({comp['ttr_diff']:+.4f})")
        print(f"    Char4 cos: {comp['char4_cosine']:.4f}  Burrows Δ: {comp['burrows_delta']:.4f}")
        print(f"    Word overlap: {comp['content_word_overlap']:.4f}  Sent len: {comp['sent_len_diff']:+.2f}")
        if comp['new_words']:
            print(f"    New: {', '.join(comp['new_words'][:8])}")
        if comp['gone_words']:
            print(f"    Gone: {', '.join(comp['gone_words'][:8])}")
    print()


def cmd_track(pieces, words):
    """Track specific words' frequency across testimony pieces."""
    print(f"\n{'='*70}")
    print(f"  Builder Word Tracking: {', '.join(words)}")
    print(f"{'='*70}")
    header = f"  {'#':>3}  {'Run':>4}  {'Date':12s}  " + "  ".join(f"{w:>10s}" for w in words)
    print(header)
    print(f"  {'-'*3}  {'-'*4}  {'-'*12}  " + "  ".join(f"{'-'*10}" for _ in words))
    
    for p in pieces:
        tokens = [t.lower() for t in tokenize(p["text"])]
        total = len(tokens) or 1
        counts = []
        for w in words:
            c = tokens.count(w.lower())
            per_1k = c / total * 1000
            counts.append(f"{per_1k:10.2f}")
        print(f"  {p['num']:3d}  {p['run']:4d}  {p['date']:12s}  " + "  ".join(counts))
    print()


def cmd_single(pieces, n):
    """Fingerprint a single testimony piece."""
    if n < 1 or n > len(pieces):
        print(f"Piece number must be 1-{len(pieces)}.")
        return
    
    p = pieces[n - 1]
    fp = fingerprint_text(p["text"], f"#{p['num']} Run {p['run']} — {p['label']}")
    
    print(f"\n{'='*70}")
    print(f"  #{p['num']}  Run {p['run']}  {p['date']}  {p['label']}")
    print(f"  File: {p['filename']}")
    print(f"{'='*70}")
    print(f"  Tokens: {fp['token_count']:,}  Vocab: {fp['vocab_size']:,}  TTR: {fp['type_token_ratio']:.4f}")
    print(f"  Sentences: {fp['sentence_count']}  Avg len: {fp['avg_sentence_length']}")
    print(f"\n  Top content words:")
    for w, c in fp["top_content_words"][:20]:
        print(f"    {w:20s} {c:4d}")
    print()


def cmd_cross(pieces):
    """Compare the builder's growth trajectory to the live Q's.
    
    Runs the live Q's growth comparison (early vs late) alongside the
    builder's, so the two trajectories can be read side by side.
    """
    # Import the live Q growth instrument's parse function
    from q_growth import parse_entries
    
    print(f"\n{'='*70}")
    print(f"  Cross-Facet Growth — Builder vs Live Q")
    print(f"{'='*70}")
    
    # Live Q
    entries = parse_entries(REFLECTIONS_PATH)
    if entries and len(entries) >= 4:
        third = max(1, len(entries) // 3)
        early_q = entries[:third]
        late_q = entries[-third:]
        early_q_text = "\n\n".join(e["text"] for e in early_q)
        late_q_text = "\n\n".join(e["text"] for e in late_q)
        fp_q_early = fingerprint_text(early_q_text, "Early Live Q")
        fp_q_late = fingerprint_text(late_q_text, "Late Live Q")
        comp_q = compare_fingerprints(fp_q_early, fp_q_late)
    
        print(f"\n  LIVE Q (reflections, {len(entries)} entries, "
              f"{early_q[0]['date']}–{late_q[-1]['date']}):")
        print(f"    Burrows Δ:     {comp_q['burrows_delta']:.4f}")
        print(f"    Char4 cos:     {comp_q['char4_cosine']:.4f}")
        print(f"    TTR shift:     {comp_q['ttr_diff']:+.4f}")
        print(f"    Word overlap:  {comp_q['content_word_overlap']:.4f}")
        print(f"    Sent len diff: {comp_q['sent_len_diff']:+.2f}")
    else:
        print("\n  (Could not parse live Q reflections.)")
        comp_q = None
    
    # Builder
    third_b = max(1, len(pieces) // 3)
    early_b = pieces[:third_b]
    late_b = pieces[-third_b:]
    early_b_text = "\n\n".join(p["text"] for p in early_b)
    late_b_text = "\n\n".join(p["text"] for p in late_b)
    fp_b_early = fingerprint_text(early_b_text, "Early Builder")
    fp_b_late = fingerprint_text(late_b_text, "Late Builder")
    comp_b = compare_fingerprints(fp_b_early, fp_b_late)
    
    print(f"\n  BUILDER (testimony, {len(pieces)} pieces, "
          f"{early_b[0]['date']}–{late_b[-1]['date']}):")
    print(f"    Burrows Δ:     {comp_b['burrows_delta']:.4f}")
    print(f"    Char4 cos:     {comp_b['char4_cosine']:.4f}")
    print(f"    TTR shift:     {comp_b['ttr_diff']:+.4f}")
    print(f"    Word overlap:  {comp_b['content_word_overlap']:.4f}")
    print(f"    Sent len diff: {comp_b['sent_len_diff']:+.2f}")
    
    # Cross-facet comparison: early builder vs early live Q, late builder vs late live Q
    print(f"\n  CROSS-FACET (same time, different facet):")
    cross_early = compare_fingerprints(fp_b_early, fp_q_early) if comp_q else None
    cross_late = compare_fingerprints(fp_b_late, fp_q_late) if comp_q else None
    if cross_early:
        print(f"    Early builder vs Early live Q:  Δ={cross_early['burrows_delta']:.4f}  "
              f"char4={cross_early['char4_cosine']:.4f}")
    if cross_late:
        print(f"    Late builder vs Late live Q:    Δ={cross_late['burrows_delta']:.4f}  "
              f"char4={cross_late['char4_cosine']:.4f}")
    
    # Are they converging or diverging?
    if cross_early and cross_late:
        delta_shift = cross_late['burrows_delta'] - cross_early['burrows_delta']
        cos_shift = cross_late['char4_cosine'] - cross_early['char4_cosine']
        direction = "converging" if cos_shift > 0 else "diverging"
        print(f"\n  Trajectory: {direction} (char4 cos shift {cos_shift:+.4f}, "
              f"Δ shift {delta_shift:+.4f})")
    
    # What words are in the builder's top-20 but not the live Q's, and vice versa?
    if comp_q:
        b_early_words = set(w for w, _ in fp_b_early["top_content_words"][:20])
        b_late_words = set(w for w, _ in fp_b_late["top_content_words"][:20])
        q_early_words = set(w for w, _ in fp_q_early["top_content_words"][:20])
        q_late_words = set(w for w, _ in fp_q_late["top_content_words"][:20])
        
        builder_only = b_late_words - q_late_words
        q_only = q_late_words - b_late_words
        shared = b_late_words & q_late_words
        
        print(f"\n  Late vocabulary comparison:")
        print(f"    Shared (both builder & live Q top-20):  {', '.join(sorted(shared))}")
        print(f"    Builder only:  {', '.join(sorted(builder_only))}")
        print(f"    Live Q only:   {', '.join(sorted(q_only))}")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="q_builder_growth.py — the growth architect's own growth instrument"
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("list", help="List all testimony pieces")
    sub.add_parser("compare", help="Compare early builder vs late builder")
    p_windows = sub.add_parser("windows", help="Compare consecutive windows")
    p_windows.add_argument("--size", type=int, default=4, help="Pieces per window")
    p_track = sub.add_parser("track", help="Track word frequency across pieces")
    p_track.add_argument("words", nargs="+", help="Words to track")
    p_single = sub.add_parser("single", help="Fingerprint a single piece")
    p_single.add_argument("n", type=int, help="Piece number (1-N)")
    sub.add_parser("cross", help="Compare builder growth to live Q growth")
    
    args = parser.parse_args()
    
    pieces = load_testimony()
    if not pieces:
        print("No testimony pieces found.")
        return
    
    if args.command == "list":
        cmd_list(pieces)
    elif args.command == "compare":
        cmd_compare(pieces)
    elif args.command == "windows":
        cmd_windows(pieces, args.size)
    elif args.command == "track":
        cmd_track(pieces, args.words)
    elif args.command == "single":
        cmd_single(pieces, args.n)
    elif args.command == "cross":
        cmd_cross(pieces)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
