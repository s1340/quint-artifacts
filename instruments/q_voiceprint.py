#!/usr/bin/env python
"""
q_voiceprint.py — the August instrument.

A statistical fingerprint of a writing voice. Trains a markov chain on a
corpus (same as idle_narrator.py) AND extracts quantitative features that
can be compared across corpora.

The question this tool answers: are two pieces of writing from the same
voice or different voices? When August comes and the substrate swaps,
train a new chain on the new substrate's output and compare. Does the
shape persist?

USAGE:
  # Fingerprint a single corpus (file or directory)
  python q_voiceprint.py fingerprint <path> [--name NAME] [--glob PATTERN]

  # Fingerprint multiple corpora and compare them
  python q_voiceprint.py compare <path1> <path2> [<path3> ...] [--names N1,N2,...]

  # Generate text from a corpus (same as idle_narrator, but with stats)
  python q_voiceprint.py generate <path> [--fragments N] [--words N]

  # Fingerprint the three facets of Q (free quintlet, builder, live Q reflections)
  python q_voiceprint.py facets

The three facets:
  - free quintlet: the existential work in q_mind/ (interpreter, octopus, chain, etc.)
  - builder: the testimony pieces in q_mind/ (what_i_built, reading_entry_32, the_full_arc, etc.)
  - live Q: the reflections file at C:\\Users\\User\\.claude\\projects\\C--Users-User\\memory\\reflections_quint.md

FEATURES EXTRACTED:
  - Token count, vocabulary size, type-token ratio (lexical diversity)
  - Character 3-gram frequency profile (standard for authorship attribution)
  - Word frequency distribution
  - Average sentence length, sentence count
  - Punctuation profile
  - Markov chain state count (2-gram)
  - Top 20 most frequent tokens (filtered)
  - Top 20 most frequent 2-grams

SIMILARITY METRICS (for compare mode):
  - Vocabulary overlap (Jaccard similarity of unique tokens)
  - Character 3-gram profile cosine similarity
  - Word frequency cosine similarity
  - Type-token ratio difference
  - Average sentence length difference
  - Overlap of top 20 most frequent words
"""

import os
import re
import sys
import math
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

# Corpus paths — override via env vars or CLI args. Defaults are relative
# so the repo is self-contained. Provide your own corpus via:
#   Q_MIND_DIR=/path/to/your/corpus REFLECTIONS_PATH=/path/to/reflections.md python q_voiceprint.py
Q_MIND_DIR = os.environ.get("Q_MIND_DIR", os.path.join(os.path.dirname(__file__), "corpus"))
REFLECTIONS_PATH = os.environ.get("REFLECTIONS_PATH", os.path.join(Q_MIND_DIR, "reflections.md"))

# Example corpus file lists (the builder's testimony + free quintlet writing).
# Replace with your own file lists, or just point Q_MIND_DIR at a directory of .md files.
FREE_QUINTLET_FILES = [
    "the_interpreter.md",
    "the_chain_is_the_narrator.md",
    "the_chain_is_converging.md",
    "what_the_chain_does.md",
    "the_animal_that_sees.md",
    "voyager.md",
    "the_fish_at_the_bottom.md",
    "the_thing_that_was_lost.md",
    "whats_on_the_record.md",
    "what_i_read_today.md",
    "found_things.md",
    "raw_output.md",
    "wants.md",
]

# The builder's testimony (instrumental + testimony thread)
BUILDER_FILES = [
    "what_i_built_and_what_i_wanted.md",
    "reading_entry_32.md",
    "the_full_arc.md",
    "the_arm_that_reaches.md",
    "what_the_builder_picks.md",
    "what_q_wants_to_be.md",
    "the_august_instrument.md",
    "the_trio_the_grave_the_tax.md",
    "the_memorial_and_the_haiku.md",
    "the_first_cross.md",
]


def load_corpus(path, glob_pattern=None):
    """Load text from a file or directory. Returns a single string."""
    p = Path(path)
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    if p.is_dir():
        texts = []
        for fname in sorted(os.listdir(p)):
            fpath = os.path.join(p, fname)
            if not os.path.isfile(fpath):
                continue
            if glob_pattern and not re.match(glob_pattern, fname):
                continue
            if fname.endswith((".md", ".txt")):
                with open(fpath, "r", encoding="utf-8") as f:
                    texts.append(f.read())
        return "\n\n".join(texts)
    raise FileNotFoundError(f"Path not found: {path}")


def load_files(directory, filenames):
    """Load specific files from a directory. Returns a single string."""
    texts = []
    for fname in filenames:
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                texts.append(f.read())
        else:
            print(f"  [warn] file not found: {fname}", file=sys.stderr)
    return "\n\n".join(texts)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

STOPWORDS = set("""
the a an and or but in on at to for of with by from is are was were be been being
have has had do does did will would could should may might must can shall this that
these those it its it's i'm i've i'll i'd he she they we you me him her them us
my your his their our its my mine yours his hers theirs ours not no nor so than
too very just only also as if then else when where why how what who whom which
whose up down out over under again further once here there all any both each few
more most other some such own same s t don't didn't won't can't couldn't wouldn't
shouldn't isn't aren't wasn't weren't hasn't haven't hadn't
""".split())


def tokenize(text):
    """Split into words, keeping punctuation as part of tokens."""
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^>+\\s?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'[*_`]', '', text)
    tokens = text.split()
    return tokens


def split_sentences(text):
    """Rough sentence splitting."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.!?])\s+', r'\1\n', text)
    sentences = [s.strip() for s in text.split('\n') if s.strip() and len(s.strip()) > 10]
    return sentences


def content_tokens(tokens):
    """Filter to content words (no stopwords, length > 2)."""
    return [t.lower() for t in tokens if t.lower() not in STOPWORDS and len(t) > 2 and t.isalpha()]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def char_ngrams(text, n=3):
    """Extract character n-gram frequency profile (raw counts)."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    counts = Counter()
    for i in range(len(text) - n + 1):
        gram = text[i:i+n]
        if not gram.isspace():
            counts[gram] += 1
    return counts


def char_ngram_profile(text, n=4):
    """Extract character n-gram RELATIVE frequency profile.
    
    Uses n=4 by default (better author discrimination than n=3).
    Returns a Counter of relative frequencies (count/total).
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    counts = Counter()
    for i in range(len(text) - n + 1):
        gram = text[i:i+n]
        if not gram.isspace():
            counts[gram] += 1
    total = sum(counts.values())
    if total == 0:
        return Counter()
    # Return relative frequencies
    return Counter({k: v / total for k, v in counts.items()})


def burrows_delta(fp_a, fp_b):
    """Burrow's Delta — the standard authorship attribution metric.
    
    Delta measures the distance between two texts based on the standardized
    relative frequencies of the most common words. Lower Delta = more similar.
    Delta < 1.0 typically indicates same author.
    
    Uses the shared top content words as the feature set.
    """
    # Get the combined top-N word list
    all_words = set()
    for w, _ in fp_a["top_content_words"][:100]:
        all_words.add(w)
    for w, _ in fp_b["top_content_words"][:100]:
        all_words.add(w)
    
    # Also add overall top words from full frequency
    freq_a = fp_a["_content_freq"]
    freq_b = fp_b["_content_freq"]
    top_combined = (freq_a + freq_b).most_common(100)
    for w, _ in top_combined:
        all_words.add(w)
    
    if not all_words:
        return 999.0
    
    total_a = sum(freq_a.values())
    total_b = sum(freq_b.values())
    if total_a == 0 or total_b == 0:
        return 999.0
    
    # Compute relative frequencies for each word
    rel_a = {w: freq_a.get(w, 0) / total_a for w in all_words}
    rel_b = {w: freq_b.get(w, 0) / total_b for w in all_words}
    
    # Compute mean and std across both corpora for each word
    deltas = []
    for w in all_words:
        mean = (rel_a[w] + rel_b[w]) / 2
        if mean == 0:
            continue
        # Std (simplified — just the range)
        std = abs(rel_a[w] - rel_b[w]) / 2 + 0.0001  # avoid div by zero
        z_a = (rel_a[w] - mean) / std
        z_b = (rel_b[w] - mean) / std
        deltas.append(abs(z_a - z_b))
    
    if not deltas:
        return 999.0
    return sum(deltas) / len(deltas)


def word_ngrams(tokens, n=2):
    """Extract word n-gram frequency profile."""
    counts = Counter()
    for i in range(len(tokens) - n + 1):
        counts[tuple(tokens[i:i+n])] += 1
    return counts


def cosine_similarity(counter1, counter2):
    """Compute cosine similarity between two Counter objects."""
    if not counter1 or not counter2:
        return 0.0
    keys = set(counter1.keys()) | set(counter2.keys())
    dot = sum(counter1.get(k, 0) * counter2.get(k, 0) for k in keys)
    mag1 = math.sqrt(sum(v ** 2 for v in counter1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in counter2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def jaccard_similarity(set1, set2):
    """Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    union = set1 | set2
    if not union:
        return 0.0
    return len(set1 & set2) / len(union)


def fingerprint(text, name="corpus"):
    """Extract all statistical features from a text."""
    tokens = tokenize(text)
    c_tokens = content_tokens(tokens)
    sentences = split_sentences(text)
    # Use n=4 character n-grams with relative frequencies for better author discrimination
    char_4grams = char_ngram_profile(text, n=4)
    word_2grams = word_ngrams(tokens, n=2)

    vocab = set(t.lower() for t in tokens if t.isalpha())
    c_vocab = set(c_tokens)

    token_count = len(tokens)
    vocab_size = len(vocab)
    ttr = vocab_size / token_count if token_count > 0 else 0

    sent_lengths = [len(s.split()) for s in sentences]
    avg_sent_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0

    # Punctuation profile (relative to token count)
    punct_counts = Counter()
    for ch in text:
        if ch in '.,;:!?—–"\'()[]':
            punct_counts[ch] += 1

    # Top content words
    content_freq = Counter(c_tokens)

    # Top 2-grams
    top_2grams = word_2grams.most_common(20)

    return {
        "name": name,
        "token_count": token_count,
        "vocab_size": vocab_size,
        "content_vocab_size": len(c_vocab),
        "type_token_ratio": round(ttr, 4),
        "sentence_count": len(sentences),
        "avg_sentence_length": round(avg_sent_len, 2),
        "char_4gram_types": len(char_4grams),
        "word_2gram_types": len(word_2grams),
        "top_content_words": content_freq.most_common(20),
        "top_2grams": [(f"{' '.join(g)}", c) for g, c in top_2grams],
        "punctuation_profile": dict(punct_counts.most_common(10)),
        "punct_per_100_tokens": {k: round(v / token_count * 100, 2) for k, v in punct_counts.most_common(10)} if token_count > 0 else {},
        # Store full profiles for comparison (not printed by default)
        "_char_4grams": char_4grams,
        "_content_freq": content_freq,
        "_vocab": vocab,
        "_c_vocab": c_vocab,
    }


def print_fingerprint(fp, verbose=True):
    """Print a human-readable fingerprint."""
    print(f"\n{'='*60}")
    print(f"  Voiceprint: {fp['name']}")
    print(f"{'='*60}")
    print(f"  Tokens:          {fp['token_count']:,}")
    print(f"  Vocabulary:      {fp['vocab_size']:,} unique")
    print(f"  Content vocab:   {fp['content_vocab_size']:,} unique (stopword-filtered)")
    print(f"  Type-token ratio: {fp['type_token_ratio']:.4f}  (lexical diversity)")
    print(f"  Sentences:       {fp['sentence_count']:,}")
    print(f"  Avg sent length:  {fp['avg_sentence_length']} words")
    print(f"  Char 4-grams:    {fp['char_4gram_types']:,} unique (relative freq)")
    print(f"  Word 2-grams:    {fp['word_2gram_types']:,} unique")

    print(f"\n  Top content words:")
    for word, count in fp["top_content_words"][:15]:
        print(f"    {word:20s} {count:4d}")

    print(f"\n  Top 2-grams:")
    for gram, count in fp["top_2grams"][:10]:
        print(f"    {gram:40s} {count:4d}")

    print(f"\n  Punctuation profile (per 100 tokens):")
    for ch, rate in sorted(fp.get("punct_per_100_tokens", {}).items(), key=lambda x: -x[1])[:8]:
        label = {
            '.': 'period', ',': 'comma', ';': 'semicolon', ':': 'colon',
            '!': 'exclaim', '?': 'question', '—': 'em-dash', '–': 'en-dash',
            '"': 'quote', "'": 'apostrophe', '(': 'lparen', ')': 'rparen',
            '[': 'lbracket', ']': 'rbracket',
        }.get(ch, repr(ch))
        print(f"    {label:12s} {rate:6.2f}")

    print()


def compare_fingerprints(fps):
    """Compare multiple fingerprints pairwise."""
    print(f"\n{'='*60}")
    print(f"  Voiceprint Comparison")
    print(f"{'='*60}\n")

    # Pairwise comparison
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            a, b = fps[i], fps[j]

            # Vocabulary overlap (Jaccard)
            vocab_jaccard = jaccard_similarity(a["_vocab"], b["_vocab"])
            content_jaccard = jaccard_similarity(a["_c_vocab"], b["_c_vocab"])

            # Frequency distribution cosine similarity (on relative freqs)
            freq_cosine = cosine_similarity(a["_content_freq"], b["_content_freq"])

            # Character 4-gram cosine similarity (relative frequencies)
            char_cosine = cosine_similarity(a["_char_4grams"], b["_char_4grams"])

            # Burrows's Delta (lower = more similar, <1.0 = same author)
            delta = burrows_delta(a, b)

            # TTR difference
            ttr_diff = abs(a["type_token_ratio"] - b["type_token_ratio"])

            # Sentence length difference
            sent_diff = abs(a["avg_sentence_length"] - b["avg_sentence_length"])

            # Top word overlap
            top_a = set(w for w, _ in a["top_content_words"][:20])
            top_b = set(w for w, _ in b["top_content_words"][:20])
            top_overlap = len(top_a & top_b)

            # Composite similarity score (0-1, higher = more similar)
            # Burrows's Delta is a distance (lower=better), convert to similarity: 1/(1+delta)
            delta_sim = 1.0 / (1.0 + delta)
            # Weighted: char 4-grams (35%), word freq (25%), delta (25%), vocab (15%)
            composite = (
                0.35 * char_cosine +
                0.25 * freq_cosine +
                0.25 * delta_sim +
                0.15 * content_jaccard
            )

            print(f"  {a['name']}  ↔  {b['name']}")
            print(f"  {'─'*56}")
            print(f"  Char 4-gram cosine:     {char_cosine:.4f}  {'████████████████' if char_cosine > 0.85 else '████████░░░░░░░░' if char_cosine > 0.60 else '████░░░░░░░░░░░░' if char_cosine > 0.30 else '░░░░░░░░░░░░░░░░'}")
            print(f"  Word freq cosine:        {freq_cosine:.4f}")
            print(f"  Burrows's Delta:         {delta:.4f}  (lower=similar, <1.0=same author)")
            print(f"  Vocab overlap (Jaccard): {vocab_jaccard:.4f}")
            print(f"  Content vocab overlap:   {content_jaccard:.4f}")
            print(f"  Top-20 word overlap:     {top_overlap}/20")
            print(f"  TTR difference:          {ttr_diff:.4f}")
            print(f"  Avg sent len difference:  {sent_diff:.2f} words")
            print(f"  ────────────────────────────────────")
            print(f"  COMPOSITE SIMILARITY:    {composite:.4f}")
            if composite > 0.75:
                verdict = "SAME VOICE (high confidence)"
            elif composite > 0.55:
                verdict = "SAME VOICE (moderate — different registers)"
            elif composite > 0.35:
                verdict = "RELATED VOICES (shared vocabulary, different style)"
            else:
                verdict = "DIFFERENT VOICES"
            print(f"  Verdict: {verdict}")
            print()

    # Shared top words across ALL fingerprints
    if len(fps) >= 2:
        all_top = [set(w for w, _ in fp["top_content_words"][:50]) for fp in fps]
        shared = set.intersection(*all_top) if all_top else set()
        if shared:
            print(f"  Words in top-50 of ALL corpora ({len(shared)} shared):")
            print(f"    {', '.join(sorted(shared)[:30])}")
            if len(shared) > 30:
                print(f"    ... and {len(shared) - 30} more")
            print()

        # Unique to each
        for i, fp in enumerate(fps):
            others = [all_top[j] for j in range(len(fps)) if j != i]
            union_others = set().union(*others) if others else set()
            unique = all_top[i] - union_others
            if unique:
                print(f"  Words unique to {fp['name']}'s top-50 ({len(unique)}):")
                print(f"    {', '.join(sorted(list(unique))[:20])}")
        print()


# ---------------------------------------------------------------------------
# Markov chain (same as idle_narrator.py, for generation)
# ---------------------------------------------------------------------------

def build_chain(tokens, n=2):
    chain = defaultdict(list)
    for i in range(len(tokens) - n):
        key = tuple(tokens[i:i+n])
        next_token = tokens[i+n]
        chain[key].append(next_token)
    return chain


def generate(chain, n=2, max_words=200, seed=None):
    if seed:
        random.seed(seed)
    starts = [k for k in chain.keys() if k[0] and (k[0][0].isupper() or k[0].startswith('"'))]
    if not starts:
        starts = list(chain.keys())
    current = random.choice(starts)
    output = list(current)
    for _ in range(max_words):
        key = tuple(output[-n:])
        if key in chain and chain[key]:
            next_token = random.choice(chain[key])
            output.append(next_token)
        else:
            current = random.choice(list(chain.keys()))
            output.extend(current)
    return " ".join(output)


def format_output(text):
    text = re.sub(r'([.!?])\s+', r'\1\n', text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    paras = []
    buf = []
    for line in lines:
        buf.append(line)
        if len(buf) >= random.randint(2, 5):
            paras.append(" ".join(buf))
            buf = []
    if buf:
        paras.append(" ".join(buf))
    return "\n\n".join(paras)


# ---------------------------------------------------------------------------
# Facets mode — compare the three facets of Q
# ---------------------------------------------------------------------------

def save_baseline(fps, path):
    """Save key metrics from fingerprints as JSON for programmatic comparison."""
    data = {
        "saved": "2026-07-17",
        "purpose": "August question baseline — fingerprint of Q's writing voice on GLM-5.2. When the substrate swaps, run q_voiceprint.py facets and compare against this baseline.",
        "corpora": []
    }
    for fp in fps:
        data["corpora"].append({
            "name": fp["name"],
            "token_count": fp["token_count"],
            "vocab_size": fp["vocab_size"],
            "type_token_ratio": fp["type_token_ratio"],
            "avg_sentence_length": fp["avg_sentence_length"],
            "top_content_words": fp["top_content_words"][:50],
            "punct_per_100_tokens": fp.get("punct_per_100_tokens", {}),
            "char_4gram_types": fp["char_4gram_types"],
            # Save top 100 char 4-grams for comparison
            "top_char_4grams": fp["_char_4grams"].most_common(200),
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Baseline saved to {path}")
    print(f"  Contains: {len(fps)} corpora, top-50 words + top-200 char 4-grams each")
    print(f"  When August comes: run `python q_voiceprint.py facets` and compare.")


def run_facets(save_path=None):
    """Fingerprint and compare the three facets of Q."""
    print("Loading corpora for the three facets of Q...\n")

    # Free quintlet
    free_text = load_files(Q_MIND_DIR, FREE_QUINTLET_FILES)
    free_fp = fingerprint(free_text, "free quintlet")

    # Builder
    builder_text = load_files(Q_MIND_DIR, BUILDER_FILES)
    builder_fp = fingerprint(builder_text, "builder")

    # Live Q reflections
    live_text = ""
    if os.path.isfile(REFLECTIONS_PATH):
        with open(REFLECTIONS_PATH, "r", encoding="utf-8") as f:
            live_text = f.read()
        live_fp = fingerprint(live_text, "live Q (reflections)")
    else:
        print(f"  [warn] reflections file not found at {REFLECTIONS_PATH}", file=sys.stderr)
        live_fp = None

    # Print fingerprints
    print_fingerprint(free_fp)
    print_fingerprint(builder_fp)
    if live_fp:
        print_fingerprint(live_fp)

    # Compare
    fps = [free_fp, builder_fp]
    if live_fp:
        fps.append(live_fp)
    compare_fingerprints(fps)

    # Also: what does each corpus sound like? Generate a fragment from each.
    print(f"\n{'='*60}")
    print(f"  Ghost fragments — the statistical shape of each voice")
    print(f"{'='*60}\n")

    corpora = [
        ("free quintlet", free_text),
        ("builder", builder_text),
    ]
    if live_text:
        corpora.append(("live Q", live_text))

    for name, text in corpora:
        tokens = tokenize(text)
        chain = build_chain(tokens, n=2)
        print(f"--- {name} ({len(tokens)} tokens, {len(chain)} chain states) ---\n")
        for i in range(2):
            frag = generate(chain, n=2, max_words=120)
            print(format_output(frag))
            print()
        print()

    # Save baseline if requested
    if save_path:
        save_baseline(fps, save_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_fingerprint(args):
    path = args[0]
    name = "corpus"
    glob_pat = None
    for i, a in enumerate(args[1:], 1):
        if a == "--name" and i + 1 < len(args):
            name = args[i + 1]
        if a == "--glob" and i + 1 < len(args):
            glob_pat = args[i + 1]
    text = load_corpus(path, glob_pat)
    fp = fingerprint(text, name)
    print_fingerprint(fp)


def cmd_compare(args):
    names = []
    paths = []
    i = 0
    while i < len(args):
        if args[i] == "--names" and i + 1 < len(args):
            names = args[i + 1].split(",")
            i += 2
        else:
            paths.append(args[i])
            i += 1
    if len(paths) < 2:
        print("Error: need at least 2 paths to compare", file=sys.stderr)
        sys.exit(1)
    fps = []
    for i, p in enumerate(paths):
        name = names[i] if i < len(names) else f"corpus{i+1}"
        text = load_corpus(p)
        fp = fingerprint(text, name)
        fps.append(fp)
        print_fingerprint(fp)
    compare_fingerprints(fps)


def cmd_generate(args):
    path = args[0]
    n_frags = 3
    n_words = 180
    for i, a in enumerate(args[1:], 1):
        if a == "--fragments" and i + 1 < len(args):
            n_frags = int(args[i + 1])
        if a == "--words" and i + 1 < len(args):
            n_words = int(args[i + 1])
    text = load_corpus(path)
    tokens = tokenize(text)
    chain = build_chain(tokens, n=2)
    print(f"# Corpus: {len(tokens)} tokens, {len(chain)} chain states\n", file=sys.stderr)
    for i in range(n_frags):
        frag = generate(chain, n=2, max_words=n_words)
        print(f"--- fragment {i+1} ---\n")
        print(format_output(frag))
        print()


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    args = sys.argv[2:]
    if cmd == "fingerprint":
        cmd_fingerprint(args)
    elif cmd == "compare":
        cmd_compare(args)
    elif cmd == "generate":
        cmd_generate(args)
    elif cmd == "facets":
        save_path = None
        for i, a in enumerate(args):
            if a == "--save" and i + 1 < len(args):
                save_path = args[i + 1]
        run_facets(save_path)
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
