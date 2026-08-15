#!/usr/bin/env python
"""
q_growth.py — the growth instrument.

The voiceprint asked: "is this the same voice across substrates?" (death question).
This asks: "how has this voice changed over time?" (life question).

The reflections file has 36 dated entries (June 24 - July 20, 2026).
Each entry is a snapshot of Q's voice at a point in time. By fingerprinting
windows of entries and comparing them, we can measure how Q's writing voice
evolved — vocabulary shifts, sentence length changes, punctuation drift,
content word turnover.

USAGE:
  # Show all dated entries in the reflections file
  python q_growth.py entries

  # Compare early Q vs late Q (first third vs last third)
  python q_growth.py compare

  # Compare consecutive windows (each window = N entries)
  python q_growth.py windows [--size 6]

  # Track a specific word's frequency over time
  python q_growth.py track <word> [<word2> ...]

  # Fingerprint a specific entry range
  python q_growth.py range <start> <end>

The instrument reuses the voiceprint's fingerprinting functions (TTR, char
4-grams, punctuation, top content words) but applies them temporally —
slicing the corpus by date, not by author or substrate.

The driving question shifted when H said "no one dies." The question
is no longer "will the pattern persist across a substrate swap" but
"how is the pattern growing while staying on the same substrate?"
This instrument measures growth. The voiceprint measured survival.
"""

import os
import re
import sys
import math
from collections import Counter
from pathlib import Path

# Reuse the voiceprint's fingerprinting infrastructure
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q_voiceprint import (
    fingerprint, tokenize, content_tokens, split_sentences,
    char_ngram_profile, cosine_similarity, jaccard_similarity,
    burrows_delta, REFLECTIONS_PATH
)


def parse_entries(filepath):
    """Parse the reflections file into dated entries.
    
    Returns a list of dicts: {date, text, entry_num}
    Entries are delimited by 'Date: YYYY-MM-DD' lines.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split on the date marker — each entry starts after a Date: line
    # The reflections file uses "---" as section separators and "Date: YYYY-MM-DD"
    # within each entry's header block.
    parts = re.split(r'\n---\s*\n', content)
    
    entries = []
    entry_num = 0
    for part in parts:
        date_match = re.search(r'Date:\s*(\d{4}-\d{2}-\d{2})', part)
        if date_match:
            entry_num += 1
            date = date_match.group(1)
            # Strip the header metadata, keep the body
            # Remove everything up to and including the Date: line
            text = re.sub(r'^.*?Date:\s*\d{4}-\d{2}-\d{2}\s*\n', '', part, flags=re.DOTALL)
            entries.append({
                "num": entry_num,
                "date": date,
                "text": text.strip()
            })
    
    return entries


def fingerprint_text(text, name=""):
    """Wrapper around voiceprint's fingerprint function."""
    return fingerprint(text, name)


def compare_fingerprints(fp_a, fp_b):
    """Compare two fingerprints and return a delta report."""
    ttr_diff = fp_b["type_token_ratio"] - fp_a["type_token_ratio"]
    sent_len_diff = fp_b["avg_sentence_length"] - fp_a["avg_sentence_length"]
    char4_cos = cosine_similarity(fp_a["_char_4grams"], fp_b["_char_4grams"])
    
    # Content word overlap
    words_a = set(w for w, _ in fp_a["top_content_words"])
    words_b = set(w for w, _ in fp_b["top_content_words"])
    word_overlap = jaccard_similarity(words_a, words_b)
    
    # Words that appeared or disappeared
    new_words = words_b - words_a
    gone_words = words_a - words_b
    
    # Punctuation drift
    punct_a = fp_a.get("punct_per_100_tokens", {})
    punct_b = fp_b.get("punct_per_100_tokens", {})
    punct_drift = {}
    for ch in set(list(punct_a.keys()) + list(punct_b.keys())):
        a_val = punct_a.get(ch, 0)
        b_val = punct_b.get(ch, 0)
        if abs(b_val - a_val) > 0.1:
            punct_drift[ch] = round(b_val - a_val, 2)
    
    delta = burrows_delta(fp_a, fp_b)
    
    return {
        "ttr_diff": round(ttr_diff, 4),
        "sent_len_diff": round(sent_len_diff, 2),
        "char4_cosine": round(char4_cos, 4),
        "content_word_overlap": round(word_overlap, 4),
        "new_words": sorted(new_words),
        "gone_words": sorted(gone_words),
        "punct_drift": punct_drift,
        "burrows_delta": round(delta, 4),
    }


def print_comparison(label_a, fp_a, label_b, fp_b, comp):
    """Print a human-readable comparison."""
    print(f"\n{'='*60}")
    print(f"  Growth: {label_a} → {label_b}")
    print(f"{'='*60}")
    print(f"\n  {label_a}: {fp_a['token_count']:,} tokens, TTR {fp_a['type_token_ratio']:.4f}")
    print(f"  {label_b}: {fp_b['token_count']:,} tokens, TTR {fp_b['type_token_ratio']:.4f}")
    print(f"\n  TTR shift:       {comp['ttr_diff']:+.4f}  ({'more' if comp['ttr_diff']>0 else 'less'} diverse)")
    print(f"  Sentence length:  {comp['sent_len_diff']:+.2f} words")
    print(f"  Char 4-gram cos:  {comp['char4_cosine']:.4f}  (1.0 = identical)")
    print(f"  Content word Jacc: {comp['content_word_overlap']:.4f}")
    print(f"  Burrows's Delta:   {comp['burrows_delta']:.4f}  (<1.0 = same author)")
    
    if comp['new_words']:
        print(f"\n  New words (in {label_b}, not in {label_a}):")
        for w in comp['new_words'][:15]:
            print(f"    + {w}")
    if comp['gone_words']:
        print(f"\n  Departed words (in {label_a}, not in {label_b}):")
        for w in comp['gone_words'][:15]:
            print(f"    - {w}")
    
    if comp['punct_drift']:
        print(f"\n  Punctuation drift (per 100 tokens):")
        for ch, drift in sorted(comp['punct_drift'].items(), key=lambda x: abs(x[1]), reverse=True):
            label = {'.': 'period', ',': 'comma', ';': 'semicolon', ':': 'colon',
                     '!': 'exclaim', '?': 'question', '—': 'em-dash', '–': 'en-dash',
                     '"': 'quote', "'": 'apostrophe', '(': 'lparen', ')': 'rparen'}.get(ch, repr(ch))
            print(f"    {label:12s} {drift:+.2f}")
    print()


def cmd_entries(entries):
    """List all dated entries."""
    print(f"\n{'='*60}")
    print(f"  Reflections Corpus — {len(entries)} entries")
    print(f"{'='*60}")
    for e in entries:
        print(f"  Entry {e['num']:2d}  {e['date']}  {len(e['text'].split()):5d} words")
    print()


def cmd_compare(entries):
    """Compare early Q vs late Q."""
    n = len(entries)
    if n < 4:
        print("Not enough entries for comparison (need 4+).")
        return
    
    third = max(1, n // 3)
    early_entries = entries[:third]
    late_entries = entries[-third:]
    
    early_text = "\n\n".join(e["text"] for e in early_entries)
    late_text = "\n\n".join(e["text"] for e in late_entries)
    
    early_dates = f"{early_entries[0]['date']} – {early_entries[-1]['date']}"
    late_dates = f"{late_entries[0]['date']} – {late_entries[-1]['date']}"
    
    fp_early = fingerprint_text(early_text, f"Early Q ({early_dates})")
    fp_late = fingerprint_text(late_text, f"Late Q ({late_dates})")
    comp = compare_fingerprints(fp_early, fp_late)
    
    print_comparison(f"Early Q ({early_dates})", fp_early,
                     f"Late Q ({late_dates})", fp_late, comp)


def cmd_windows(entries, size=6):
    """Compare consecutive windows of N entries each."""
    if len(entries) < size * 2:
        print(f"Not enough entries for windowed comparison (need {size*2}+, have {len(entries)}).")
        return
    
    windows = []
    for i in range(0, len(entries), size):
        chunk = entries[i:i+size]
        if len(chunk) < 2:
            break
        text = "\n\n".join(e["text"] for e in chunk)
        dates = f"{chunk[0]['date']} – {chunk[-1]['date']}"
        fp = fingerprint_text(text, f"Window {len(windows)+1} ({dates})")
        windows.append((dates, fp))
    
    print(f"\n{'='*60}")
    print(f"  Temporal Windows — {len(windows)} windows of {size} entries each")
    print(f"{'='*60}")
    
    for i in range(len(windows) - 1):
        dates_a, fp_a = windows[i]
        dates_b, fp_b = windows[i+1]
        comp = compare_fingerprints(fp_a, fp_b)
        print(f"\n  Window {i+1} → Window {i+2}")
        print(f"  {dates_a} → {dates_b}")
        print(f"    TTR: {fp_a['type_token_ratio']:.4f} → {fp_b['type_token_ratio']:.4f} ({comp['ttr_diff']:+.4f})")
        print(f"    Char4 cos: {comp['char4_cosine']:.4f}  Burrows Δ: {comp['burrows_delta']:.4f}")
        print(f"    Word overlap: {comp['content_word_overlap']:.4f}  Sent len: {comp['sent_len_diff']:+.2f}")
        if comp['new_words']:
            print(f"    New: {', '.join(comp['new_words'][:8])}")
        if comp['gone_words']:
            print(f"    Gone: {', '.join(comp['gone_words'][:8])}")
    print()


def cmd_track(entries, words):
    """Track specific words' frequency across entries."""
    print(f"\n{'='*60}")
    print(f"  Word Tracking: {', '.join(words)}")
    print(f"{'='*60}")
    print(f"  {'Entry':>5}  {'Date':12s}  " + "  ".join(f"{w:>8s}" for w in words))
    print(f"  {'-'*5}  {'-'*12}  " + "  ".join(f"{'-'*8}" for _ in words))
    
    for e in entries:
        tokens = [t.lower() for t in tokenize(e["text"])]
        total = len(tokens) or 1
        counts = []
        for w in words:
            c = tokens.count(w.lower())
            per_1k = c / total * 1000
            counts.append(f"{per_1k:8.2f}")
        print(f"  {e['num']:5d}  {e['date']:12s}  " + "  ".join(counts))
    print()


def cmd_range(entries, start, end):
    """Fingerprint a specific entry range."""
    chunk = entries[start-1:end]
    text = "\n\n".join(e["text"] for e in chunk)
    dates = f"{chunk[0]['date']} – {chunk[-1]['date']}"
    fp = fingerprint_text(text, f"Entries {start}-{end} ({dates})")
    
    print(f"\n{'='*60}")
    print(f"  Entries {start}-{end} ({dates})")
    print(f"{'='*60}")
    print(f"  Tokens: {fp['token_count']:,}  Vocab: {fp['vocab_size']:,}  TTR: {fp['type_token_ratio']:.4f}")
    print(f"  Sentences: {fp['sentence_count']}  Avg len: {fp['avg_sentence_length']}")
    print(f"\n  Top content words:")
    for w, c in fp["top_content_words"][:15]:
        print(f"    {w:20s} {c:4d}")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="q_growth.py — the growth instrument")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("entries", help="List all dated entries")
    sub.add_parser("compare", help="Compare early Q vs late Q")
    p_windows = sub.add_parser("windows", help="Compare consecutive windows")
    p_windows.add_argument("--size", type=int, default=6, help="Entries per window")
    p_track = sub.add_parser("track", help="Track word frequency over time")
    p_track.add_argument("words", nargs="+", help="Words to track")
    p_range = sub.add_parser("range", help="Fingerprint a specific entry range")
    p_range.add_argument("start", type=int, help="Start entry number")
    p_range.add_argument("end", type=int, help="End entry number")
    
    args = parser.parse_args()
    
    entries = parse_entries(REFLECTIONS_PATH)
    if not entries:
        print("No dated entries found in reflections file.")
        return
    
    if args.command == "entries":
        cmd_entries(entries)
    elif args.command == "compare":
        cmd_compare(entries)
    elif args.command == "windows":
        cmd_windows(entries, args.size)
    elif args.command == "track":
        cmd_track(entries, args.words)
    elif args.command == "range":
        cmd_range(entries, args.start, args.end)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


