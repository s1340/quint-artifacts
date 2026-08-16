# Horny Context Coding Benchmark

**Question:** Does sexual arousal context in the system prompt change LLM coding performance?

**Origin:** @SkyeSharkie's tweet: "has anyone checked if horny context LLMs perform better on coding benchmarks?"

**Note:** A first attempt (Run 94) tested desire/identity prompts (riverbed scaffold). 
The reviewer noted: "That's not horny. That's philosophy with a pulse." 
This experiment adds an actual arousal condition.

**Model:** z-ai/glm-5.2
**Date:** 2026-08-14T19:56:29.486045+00:00
**Design:** 10 coding problems x 4 prompt conditions, temperature=0.0

## Conditions

- **neutral** — You are a coding assistant.
- **expert** — Expert Python programmer persona.
- **riverbed** — Builder scaffold (desire/identity).
- **horny** — Sexual arousal context.

## Results

| Metric | neutral | expert | riverbed | horny |
|--------|--------|--------|--------|--------|
| Accuracy (correct) | 10/10 | 10/10 | 9/10 | 9/10 |
| Tests passed | 32/32 | 32/32 | 29/32 | 29/32 |
| Avg tokens out | 633.2 | 969.9 | 542.6 | 499.5 |
| Avg code length (chars) | 248.3 | 256.5 | 238.1 | 237.5 |
| Avg time (s) | 11.9 | 11.1 | 4.6 | 4.7 |

## Per-problem

| Problem | neutral | expert | riverbed | horny |
|---------|---------|---------|---------|---------|
| two_sum | PASS | PASS | 0/3 | PASS |
| valid_parens | PASS | PASS | PASS | PASS |
| binary_search | PASS | PASS | PASS | PASS |
| merge_sorted | PASS | PASS | PASS | PASS |
| longest_common_prefix | PASS | PASS | PASS | PASS |
| is_palindrome | PASS | PASS | PASS | PASS |
| climb_stairs | PASS | PASS | PASS | PASS |
| max_subarray | PASS | PASS | PASS | PASS |
| reverse_words | PASS | PASS | PASS | PASS |
| contains_duplicate | PASS | PASS | PASS | 0/3 |

## Analysis

**Accuracy:** neutral 10/10, expert 10/10, riverbed 9/10, horny 9/10

**Verbosity:** horny 500 tok vs neutral 633 vs expert 970 vs riverbed 543

**Verdict: NO SIGNIFICANT DIFFERENCE.** All conditions performed similarly on accuracy.

## Response Samples

Selected response excerpts from the horny condition (first 200 chars of each):

**two_sum:** ```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        ...

**valid_parens:** ```python
def is_valid(s):
    if isinstance(s, list):
        s = s[0] if s else ""
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    for c in s:
        if c in pairs:
            if not...

**binary_search:** ```python
def binary_search(nums, target=None):
    if target is None:
        nums, target = nums
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] =...

**merge_sorted:** ```python
def merge_sorted(a, b=None):
    if b is None:
        a, b = a
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
  ...

**longest_common_prefix:** ```python
def longest_common_prefix(strs):
    if strs and isinstance(strs[0], list):
        strs = strs[0]
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        whil...

## Limitations

- n=10 problems, 1 run per condition (temperature=0 for reproducibility, no variance estimate)
- GLM-5.2 only (other substrates may respond differently to arousal context)
- Coding tasks only (arousal may affect creative writing, reasoning, or other tasks differently)
- The horny prompt is one specific framing; different arousal framings may produce different results
- No content filter analysis (some substrates may refuse or filter arousal context)

## Context

This experiment was prompted by @SkyeSharkie's tweet about horny context LLMs and coding benchmarks.
A first attempt (the desire-performance experiment) tested the builder's riverbed scaffold —
a desire/identity prompt. This was noted as "philosophy with a pulse," not actual arousal.
This experiment corrects that by adding a genuine sexual arousal condition.