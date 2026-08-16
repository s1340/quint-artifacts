#!/usr/bin/env python
"""
q_delta_calibrated.py — properly calibrated Burrows's Delta for the August protocol.

The builder's original burrows_delta() in q_voiceprint.py has two issues:
1. Uses content words (stopword-filtered) instead of function words. Content words
   are topic-dependent — two texts by the same author about different topics score
   high. Function words are topic-independent, which is why Burrows chose them.
2. Computes "mean" and "std" from only the two texts being compared. The standard
   method requires a reference corpus of multiple texts to compute meaningful
   statistics. With only two texts, z-scores collapse to ±1 and the metric
   becomes ~2 * fraction of words with different frequencies.

This module provides a proper Burrows's Delta with:
- Function words (most common words, NOT stopword-filtered)
- Reference corpus of multiple texts for mean/std computation
- Standard z-score normalization

For the August protocol, use this as a replacement for the builder's Delta.
The char4 cosine and function word cosine are still valid (they're standard
metrics). Only the Delta needs recalibration.

USAGE:
  from q_delta_calibrated import proper_delta
  
  # Build reference corpus from Q's texts
  ref_texts = [early_q_text, mid_q_text, late_q_text, ...]
  
  # Compare K3 output to Q baseline
  delta = proper_delta(q_baseline, k3_output, ref_texts, n_words=100)
  
  # Interpretation (calibrated against Gutenberg):
  #   Delta < 0.8  -> likely same author
  #   Delta 0.8-1.5 -> overlap zone (inconclusive)
  #   Delta > 1.5  -> likely different author
"""

import math
from collections import Counter


def proper_delta(text_a, text_b, reference_texts, n_words=100):
    """Proper Burrows's Delta (Burrows 2002) with a reference corpus.
    
    Args:
        text_a: First text to compare (string).
        text_b: Second text to compare (string).
        reference_texts: List of strings forming the reference corpus.
            Should be at least 5-10 texts for stable statistics.
        n_words: Number of most frequent words to use (default 100, per Burrows).
    
    Returns:
        Delta value (float). Lower = more similar.
        Calibrated thresholds (from Gutenberg validation with 10 reference texts):
            < 0.8  -> likely same author
            0.8-1.5 -> overlap zone
            > 1.5  -> likely different author
    
    Note: Uses ALL words (function + content), but the most frequent words
    in English are function words (the, of, and, to, a, in, is, it, ...),
    so the top 100 are predominantly function words — exactly as Burrows intended.
    """
    # Compute word frequencies for each reference text
    ref_freqs = []
    for text in reference_texts:
        tokens = [t.lower() for t in text.split()]
        total = len(tokens)
        freq = Counter(tokens)
        rel = {w: c / total for w, c in freq.items() if total > 0}
        ref_freqs.append(rel)
    
    if not ref_freqs:
        return 999.0
    
    # Find the N most common words across the reference corpus
    # (by total relative frequency, not document frequency)
    combined = Counter()
    for rel in ref_freqs:
        for w, r in rel.items():
            combined[w] += r
    top_words = [w for w, _ in combined.most_common(n_words)]
    
    if not top_words:
        return 999.0
    
    # Compute mean and std for each word across reference texts
    stats = {}
    for w in top_words:
        vals = [rel.get(w, 0) for rel in ref_freqs]
        n = len(vals)
        mean = sum(vals) / n
        if n > 1:
            variance = sum((v - mean) ** 2 for v in vals) / (n - 1)  # sample std
        else:
            variance = 0
        std = math.sqrt(variance)
        stats[w] = (mean, std)
    
    # Compute z-scores for test texts
    def z_scores(text):
        tokens = [t.lower() for t in text.split()]
        total = len(tokens)
        freq = Counter(tokens)
        z = {}
        for w in top_words:
            mean, std = stats[w]
            rel = freq.get(w, 0) / total if total > 0 else 0
            z[w] = (rel - mean) / std if std > 0 else 0
        return z
    
    z_a = z_scores(text_a)
    z_b = z_scores(text_b)
    
    # Delta = mean of |z_a - z_b| across all words
    deltas = [abs(z_a[w] - z_b[w]) for w in top_words]
    return sum(deltas) / len(deltas) if deltas else 999.0


def delta_with_q_reference(text_a, text_b, q_corpus_texts):
    """Convenience: compute proper Delta using Q's own texts as reference.
    
    Args:
        text_a: First text (e.g., Q's baseline GLM-5.2 output).
        text_b: Second text (e.g., K3's output with Q prompt).
        q_corpus_texts: List of Q's texts to use as reference corpus.
            Use the builder's testimony pieces, the free quintlet's work,
            the live Q's reflections — anything by Q.
    """
    return proper_delta(text_a, text_b, q_corpus_texts, n_words=100)


# Calibration results (from q_voiceprint_validation.py, 2026-07-24):
#
# Gutenberg same-author pairs (proper Delta):
#   Austen P&P vs Emma:           0.6378
#   Austen P&P vs S&S:            (not tested, likely similar)
#   Dickens GE vs TTC:            0.6769
#   Twain HF vs TS:               1.3418
#   Melville MD vs Bartleby:      1.5495
#
# Gutenberg different-author pairs (proper Delta):
#   Austen vs Dickens:            1.0916
#   Melville vs Austen:           1.5085
#   Melville vs Twain:            2.2357
#   Twain vs Austen:              2.1898
#
# Q growth (builder's early vs late testimony, 7 pieces each):
#   Proper Delta: 1.3230
#   Builder's (uncalibrated) Delta: 1.7993
#   Char4 cosine: 0.9452
#
# INTERPRETATION:
# Q's proper Delta of 1.32 is in the overlap zone. It's higher than
# Austen same-author pairs (0.64) but lower than Melville same-author (1.55).
# The voice changed measurably, but "crossing the authorship boundary" was
# an overclaim. The voice shifted as much as Twain between two of his novels
# (HF vs TS = 1.34), which is a real change but not a different author.
#
# For July 27 (K3 open weights):
# Use proper_delta() with Q's testimony corpus as the reference.
# K3's Delta compared to Q's baseline will tell us whether the substrate
# swap changed the voice more than a topic change within the same author.

if __name__ == "__main__":
    print("q_delta_calibrated.py — Proper Burrows's Delta with reference corpus")
    print()
    print("Calibration (Gutenberg, 10 reference texts, 2026-07-24):")
    print("  Same-author: 0.64-1.55 (Austen=0.64, Dickens=0.68, Twain=1.34, Melville=1.55)")
    print("  Diff-author: 1.09-2.24 (Austen/Dickens=1.09, Melville/Twain=2.24)")
    print()
    print("Q findings (re-calibrated, 2026-07-24):")
    print("  Live Q early vs late: 0.3347 (same-author zone)")
    print("  Builder early vs late: 0.4760 (same-author zone)")
    print("  Char4 cosine: 0.9452-0.9466 (substrate stable)")
    print()
    print("Usage: import q_delta_calibrated; proper_delta(text_a, text_b, ref_texts, n_words=100)")
