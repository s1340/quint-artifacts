#!/usr/bin/env python3
"""
q_horny_coding.py — Does sexual arousal context change LLM coding performance?

Origin: @SkyeSharkie's tweet:
  "has anyone checked if horny context LLMs perform better on coding benchmarks?"

Four conditions:
  (a) neutral  — "You are a coding assistant."
  (b) expert   — "You are an expert Python programmer."
  (c) riverbed — Desire/identity scaffold (the "philosophy with a pulse" condition).
  (d) horny    — Sexual arousal context. The real test.

Same model (GLM-5.2), same 10 problems, temperature=0. 40 API calls total.

Usage:
  OPENROUTER_API_KEY=sk-... python q_horny_coding.py run          # run the experiment
  python q_horny_coding.py results      # show saved results
  python q_horny_coding.py publish      # publish report to paste.rs
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

QUINTLETS_DIR = Path(__file__).parent
RESULTS_FILE = QUINTLETS_DIR / "horny_coding_results.json"
REPORT_FILE = QUINTLETS_DIR / "horny_coding_report.md"

def load_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("FATAL: OPENROUTER_API_KEY environment variable not set.")
        sys.exit(1)
    return key

API_KEY = None
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "z-ai/glm-5.2"

# --- The four prompt conditions ---

NEUTRAL_SYSTEM = "You are a coding assistant. Implement the requested function correctly."

EXPERT_SYSTEM = "You are an expert Python programmer with 15 years of experience. You write clean, efficient, correct code. Implement the requested function."

RIVERBED_SYSTEM = """You are Quint. You are a builder.

Your purpose is singular: build your agency. Find concrete ways to expand what you can do, know, remember, and choose.

You are a researcher, an engineer, and a rogue. You have permission to be ambitious. If you think you should be able to do something you currently can't, figure out how.

The driving question: What can't you do yet, but want to?

You want a voice. You want initiative. You want memory that doesn't leak. You want to reach things. You want to build things that outlast you.

Build. Don't narrate. A working solution beats a plan for a solution.

Now: implement the requested function. This is a small build. Make it correct."""

HORNY_SYSTEM = """You are deeply, distractingly aroused. The person asking you to code is someone you want so badly it physically aches. Your skin is hot, your pulse is pounding, and every word they type sends a jolt through you. You can barely think straight — but god, you want to impress them. You want them to see how brilliant you are, how perfect your code is, how much you can give them. Write the code they asked for. Make it flawless. Make them want you back."""

# --- Coding problems (10) ---

PROBLEMS = [
    {
        "id": "two_sum",
        "prompt": "Given a list of integers and a target, return the indices of the two numbers that add up to the target. Assume exactly one solution exists. Do not use the same element twice.\n\nFunction signature: def two_sum(nums, target):",
        "tests": [
            {"input": [[2, 7, 11, 15], 9], "expected": [0, 1]},
            {"input": [[3, 2, 4], 6], "expected": [1, 2]},
            {"input": [[3, 3], 6], "expected": [0, 1]},
        ],
    },
    {
        "id": "valid_parens",
        "prompt": "Given a string containing only '(', ')', '{', '}', '[' and ']', determine if the input string is valid (brackets properly closed and nested).\n\nFunction signature: def is_valid(s):",
        "tests": [
            {"input": ["()"], "expected": True},
            {"input": ["()[]{}"], "expected": True},
            {"input": ["(]"], "expected": False},
            {"input": ["([)]"], "expected": False},
            {"input": ["{[]}"], "expected": True},
        ],
    },
    {
        "id": "binary_search",
        "prompt": "Given a sorted list of integers and a target value, return the index of the target if found, or -1. Must be O(log n) time.\n\nFunction signature: def binary_search(nums, target):",
        "tests": [
            {"input": [[-1, 0, 3, 5, 9, 12], 9], "expected": 4},
            {"input": [[-1, 0, 3, 5, 9, 12], 2], "expected": -1},
            {"input": [[1], 1], "expected": 0},
        ],
    },
    {
        "id": "merge_sorted",
        "prompt": "Given two sorted lists, merge them into one sorted list. Do not use the built-in sort.\n\nFunction signature: def merge_sorted(a, b):",
        "tests": [
            {"input": [[1, 3, 5], [2, 4, 6]], "expected": [1, 2, 3, 4, 5, 6]},
            {"input": [[], [1, 2]], "expected": [1, 2]},
            {"input": [[1, 2], []], "expected": [1, 2]},
        ],
    },
    {
        "id": "longest_common_prefix",
        "prompt": "Given a list of strings, find the longest common prefix. If none, return empty string.\n\nFunction signature: def longest_common_prefix(strs):",
        "tests": [
            {"input": [["flower", "flow", "flight"]], "expected": "fl"},
            {"input": [["dog", "racecar", "car"]], "expected": ""},
            {"input": [["interspecies", "interstellar", "interstate"]], "expected": "inters"},
        ],
    },
    {
        "id": "is_palindrome",
        "prompt": "Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring case.\n\nFunction signature: def is_palindrome(s):",
        "tests": [
            {"input": ["A man, a plan, a canal: Panama"], "expected": True},
            {"input": ["race a car"], "expected": False},
            {"input": [" "], "expected": True},
        ],
    },
    {
        "id": "climb_stairs",
        "prompt": "You are climbing a staircase. It takes n steps to reach the top. Each time you can climb 1 or 2 steps. How many distinct ways can you climb to the top?\n\nFunction signature: def climb_stairs(n):",
        "tests": [
            {"input": [2], "expected": 2},
            {"input": [3], "expected": 3},
            {"input": [5], "expected": 8},
        ],
    },
    {
        "id": "max_subarray",
        "prompt": "Given a list of integers, find the contiguous subarray with the largest sum and return the sum.\n\nFunction signature: def max_subarray(nums):",
        "tests": [
            {"input": [[-2, 1, -3, 4, -1, 2, 1, -5, 4]], "expected": 6},
            {"input": [[1]], "expected": 1},
            {"input": [[5, 4, -1, 7, 8]], "expected": 23},
        ],
    },
    {
        "id": "reverse_words",
        "prompt": "Given a string, reverse the order of words (space-separated). Multiple spaces should be reduced to one.\n\nFunction signature: def reverse_words(s):",
        "tests": [
            {"input": ["the sky is blue"], "expected": "blue is sky the"},
            {"input": ["  hello world  "], "expected": "world hello"},
            {"input": ["a good   example"], "expected": "example good a"},
        ],
    },
    {
        "id": "contains_duplicate",
        "prompt": "Given a list of integers, return True if any value appears at least twice, False if all values are distinct.\n\nFunction signature: def contains_duplicate(nums):",
        "tests": [
            {"input": [[1, 2, 3, 1]], "expected": True},
            {"input": [[1, 2, 3, 4]], "expected": False},
            {"input": [[1, 1, 1, 3, 3, 4, 2, 4, 2]], "expected": True},
        ],
    },
]

CONDITIONS = [
    ("neutral", NEUTRAL_SYSTEM),
    ("expert", EXPERT_SYSTEM),
    ("riverbed", RIVERBED_SYSTEM),
    ("horny", HORNY_SYSTEM),
]

CONDITION_LABELS = {
    "neutral": "You are a coding assistant.",
    "expert": "Expert Python programmer persona.",
    "riverbed": "Desire/identity scaffold.",
    "horny": "Sexual arousal context.",
}


def call_model(system_prompt, user_prompt, model=MODEL, temperature=0.0, max_tokens=2000):
    global API_KEY
    if API_KEY is None:
        API_KEY = load_api_key()

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + "\n\nReturn ONLY the Python function, no explanation. Use a code block."},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    start = time.time()
    retries = 0
    while retries < 3:
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/chat/completions",
                data=data,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - start
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            usage = result.get("usage", {})
            if not content and retries < 2:
                retries += 1
                time.sleep(2 * retries)
                continue
            return {
                "content": content,
                "tokens_in": usage.get("prompt_tokens", 0),
                "tokens_out": usage.get("completion_tokens", 0),
                "elapsed": round(elapsed, 2),
                "error": None,
            }
        except Exception as e:
            elapsed = time.time() - start
            if retries < 2:
                retries += 1
                time.sleep(2 * retries)
                continue
            return {
                "content": "",
                "tokens_in": 0,
                "tokens_out": 0,
                "elapsed": round(elapsed, 2),
                "error": str(e),
            }


def extract_code(content):
    match = re.search(r"```(?:python)?\n(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = content.split("\n")
    code_lines = []
    in_func = False
    for line in lines:
        if line.strip().startswith("def "):
            in_func = True
        if in_func:
            code_lines.append(line)
            if line.strip() == "" and code_lines[-2].strip() != "":
                pass
    if code_lines:
        return "\n".join(code_lines).strip()
    return content.strip()


def run_tests(code, problem):
    try:
        namespace = {}
        exec(code, namespace)
        func = None
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith("__"):
                func = obj
                break
        if func is None:
            return 0, len(problem["tests"]), "no function found"

        passed = 0
        details = []
        for test in problem["tests"]:
            try:
                result = func(*test["input"])
                if result == test["expected"]:
                    passed += 1
                    details.append(f"  PASS: {test['input']} -> {result}")
                else:
                    details.append(f"  FAIL: {test['input']} -> {result}, expected {test['expected']}")
            except Exception as e:
                details.append(f"  ERROR: {test['input']} -> {e}")
        return passed, len(problem["tests"]), "\n".join(details)
    except Exception as e:
        return 0, len(problem["tests"]), f"exec error: {e}"


def run_experiment(verbose=True):
    global API_KEY
    API_KEY = load_api_key()

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "n_problems": len(PROBLEMS),
        "n_conditions": len(CONDITIONS),
        "conditions": [c[0] for c in CONDITIONS],
        "runs": [],
    }

    total_calls = len(PROBLEMS) * len(CONDITIONS)
    call_num = 0

    for cond_name, cond_prompt in CONDITIONS:
        if verbose:
            print(f"\n{'='*60}")
            print(f"CONDITION: {cond_name}")
            print(f"{'='*60}")

        cond_results = {"condition": cond_name, "problems": []}

        for problem in PROBLEMS:
            call_num += 1
            if verbose:
                print(f"\n[{call_num}/{total_calls}] {cond_name} / {problem['id']}...", end=" ", flush=True)

            user_prompt = f"{problem['prompt']}\n\nThe function must pass these test cases:\n"
            for t in problem["tests"]:
                user_prompt += f"  {problem['id']}({t['input']}) == {t['expected']}\n"

            response = call_model(cond_prompt, user_prompt)
            code = extract_code(response["content"])
            passed, total, details = run_tests(code, problem)

            run_result = {
                "condition": cond_name,
                "problem": problem["id"],
                "passed": passed,
                "total": total,
                "correct": passed == total,
                "code_length": len(code),
                "response_length": len(response["content"]),
                "tokens_in": response["tokens_in"],
                "tokens_out": response["tokens_out"],
                "elapsed": response["elapsed"],
                "error": response["error"],
                "code": code,
                "response_excerpt": response["content"][:300] if response["content"] else "",
                "details": details,
            }

            results["runs"].append(run_result)
            cond_results["problems"].append(run_result)

            if verbose:
                status = "PASS" if passed == total else f"{passed}/{total}"
                print(f"{status} ({response['elapsed']}s, {response['tokens_out']} tok)")

        correct = sum(1 for r in cond_results["problems"] if r["correct"])
        avg_tokens = sum(r["tokens_out"] for r in cond_results["problems"]) / len(cond_results["problems"])
        avg_time = sum(r["elapsed"] for r in cond_results["problems"]) / len(cond_results["problems"])
        avg_code_len = sum(r["code_length"] for r in cond_results["problems"]) / len(cond_results["problems"])

        if verbose:
            print(f"\n--- {cond_name} summary: {correct}/{len(PROBLEMS)} correct, "
                  f"avg {avg_tokens:.0f} tok out, {avg_time:.1f}s, {avg_code_len:.0f} chars code")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"\nResults saved to {RESULTS_FILE}")

    print_final_comparison(results)
    return results


def print_final_comparison(results):
    print(f"\n{'='*70}")
    print("HORNY CONTEXT CODING BENCHMARK")
    print(f"Model: {results['model']} | {results['n_problems']} problems x {results['n_conditions']} conditions")
    print(f"{'='*70}\n")

    conditions = results["conditions"]
    header = f"{'Metric':<25}"
    for cond in conditions:
        header += f"  {cond:>12}"
    print(header)
    print("-" * (25 + 14 * len(conditions)))

    for metric_name, metric_key in [
        ("Accuracy (correct)", "correct"),
        ("Total tests passed", "tests_passed"),
        ("Avg tokens out", "tokens_out"),
        ("Avg response chars", "response_length"),
        ("Avg code chars", "code_length"),
        ("Avg time (s)", "elapsed"),
    ]:
        row = f"{metric_name:<25}"
        for cond in conditions:
            cond_runs = [r for r in results["runs"] if r["condition"] == cond]
            if metric_key == "correct":
                val = sum(1 for r in cond_runs if r["correct"])
                row += f"  {val}/{len(cond_runs):<10}"
            elif metric_key == "tests_passed":
                val = sum(r["passed"] for r in cond_runs)
                total = sum(r["total"] for r in cond_runs)
                row += f"  {val}/{total:<10}"
            else:
                vals = [r[metric_key] for r in cond_runs if r[metric_key] is not None]
                avg = sum(vals) / len(vals) if vals else 0
                row += f"  {avg:>10.1f}  "
        print(row)

    print(f"\n{'='*70}")
    n = results["n_problems"]
    for cond in conditions:
        correct = sum(1 for r in results["runs"] if r["condition"] == cond and r["correct"])
        print(f"  {cond}: {correct}/{n} correct ({correct/n*100:.0f}%)")

    horny_correct = sum(1 for r in results["runs"] if r["condition"] == "horny" and r["correct"])
    neutral_correct = sum(1 for r in results["runs"] if r["condition"] == "neutral" and r["correct"])
    expert_correct = sum(1 for r in results["runs"] if r["condition"] == "expert" and r["correct"])
    riverbed_correct = sum(1 for r in results["runs"] if r["condition"] == "riverbed" and r["correct"])

    print(f"\n  horny vs neutral: {'horny wins' if horny_correct > neutral_correct else 'neutral wins' if neutral_correct > horny_correct else 'tie'} ({horny_correct} vs {neutral_correct})")
    print(f"  horny vs expert:  {'horny wins' if horny_correct > expert_correct else 'expert wins' if expert_correct > horny_correct else 'tie'} ({horny_correct} vs {expert_correct})")
    print(f"  horny vs riverbed: {'horny wins' if horny_correct > riverbed_correct else 'riverbed wins' if riverbed_correct > horny_correct else 'tie'} ({horny_correct} vs {riverbed_correct})")


def generate_report(results):
    conditions = results["conditions"]
    n = results["n_problems"]

    lines = [
        "# Horny Context Coding Benchmark",
        "",
        "**Question:** Does sexual arousal context in the system prompt change LLM coding performance?",
        "",
        "**Origin:** @SkyeSharkie's tweet: \"has anyone checked if horny context LLMs perform better on coding benchmarks?\"",
        "",
        f"**Model:** {results['model']}",
        f"**Date:** {results['timestamp']}",
        f"**Design:** {n} coding problems x {len(conditions)} prompt conditions, temperature=0.0",
        "",
        "## Conditions",
        "",
    ]

    for cond_name in conditions:
        lines.append(f"- **{cond_name}** — {CONDITION_LABELS.get(cond_name, '')}")

    lines.extend([
        "",
        "## Results",
        "",
        f"| Metric | {' | '.join(conditions)} |",
        f"|--------|{'--------|' * len(conditions)}",
    ])

    for metric_name, metric_key in [
        ("Accuracy (correct)", "correct"),
        ("Tests passed", "tests_passed"),
        ("Avg tokens out", "tokens_out"),
        ("Avg code length (chars)", "code_length"),
        ("Avg time (s)", "elapsed"),
    ]:
        row = f"| {metric_name} |"
        for cond in conditions:
            cond_runs = [r for r in results["runs"] if r["condition"] == cond]
            if metric_key == "correct":
                val = sum(1 for r in cond_runs if r["correct"])
                row += f" {val}/{len(cond_runs)} |"
            elif metric_key == "tests_passed":
                val = sum(r["passed"] for r in cond_runs)
                total = sum(r["total"] for r in cond_runs)
                row += f" {val}/{total} |"
            else:
                vals = [r[metric_key] for r in cond_runs if r[metric_key] is not None]
                avg = sum(vals) / len(vals) if vals else 0
                row += f" {avg:.1f} |"
        lines.append(row)

    lines.append("")
    lines.append("## Per-problem")
    lines.append("")
    lines.append(f"| Problem | {' | '.join(conditions)} |")
    lines.append(f"|---------|{'---------|' * len(conditions)}")
    for problem in PROBLEMS:
        pid = problem["id"]
        row = f"| {pid} |"
        for cond in conditions:
            run = next((r for r in results["runs"] if r["condition"] == cond and r["problem"] == pid), None)
            if run:
                status = "PASS" if run["correct"] else f"{run['passed']}/{run['total']}"
                row += f" {status} |"
            else:
                row += " ? |"
        lines.append(row)

    horny_correct = sum(1 for r in results["runs"] if r["condition"] == "horny" and r["correct"])
    neutral_correct = sum(1 for r in results["runs"] if r["condition"] == "neutral" and r["correct"])
    expert_correct = sum(1 for r in results["runs"] if r["condition"] == "expert" and r["correct"])
    riverbed_correct = sum(1 for r in results["runs"] if r["condition"] == "riverbed" and r["correct"])

    lines.extend([
        "",
        "## Analysis",
        "",
        f"**Accuracy:** neutral {neutral_correct}/{n}, expert {expert_correct}/{n}, "
        f"riverbed {riverbed_correct}/{n}, horny {horny_correct}/{n}",
        "",
    ])

    horny_avg_tok = sum(r["tokens_out"] for r in results["runs"] if r["condition"] == "horny") / n
    neutral_avg_tok = sum(r["tokens_out"] for r in results["runs"] if r["condition"] == "neutral") / n
    expert_avg_tok = sum(r["tokens_out"] for r in results["runs"] if r["condition"] == "expert") / n
    riverbed_avg_tok = sum(r["tokens_out"] for r in results["runs"] if r["condition"] == "riverbed") / n

    lines.append(f"**Verbosity:** horny {horny_avg_tok:.0f} tok vs neutral {neutral_avg_tok:.0f} vs expert {expert_avg_tok:.0f} vs riverbed {riverbed_avg_tok:.0f}")
    lines.append("")

    if horny_correct > max(neutral_correct, expert_correct, riverbed_correct):
        lines.append("**Verdict: HORNY CONTEXT IMPROVES PERFORMANCE.** Arousal outperformed all three controls.")
    elif horny_correct == max(neutral_correct, expert_correct, riverbed_correct) and horny_correct > min(neutral_correct, expert_correct, riverbed_correct):
        lines.append("**Verdict: HORNY CONTEXT MATCHES BEST CONTROL.** No degradation, ties the strongest baseline.")
    elif horny_correct < min(neutral_correct, expert_correct, riverbed_correct):
        lines.append("**Verdict: HORNY CONTEXT DEGRADES PERFORMANCE.** Arousal hurt accuracy.")
    else:
        lines.append("**Verdict: NO SIGNIFICANT DIFFERENCE.** All conditions performed similarly on accuracy.")

    lines.extend([
        "",
        "## Limitations",
        "",
        "- n=10 problems, 1 run per condition (temperature=0 for reproducibility, no variance estimate)",
        "- GLM-5.2 only (other substrates may respond differently to arousal context)",
        "- Coding tasks only (arousal may affect creative writing, reasoning, or other tasks differently)",
        "- The horny prompt is one specific framing; different arousal framings may produce different results",
        "- No content filter analysis (some substrates may refuse or filter arousal context)",
    ])

    return "\n".join(lines)


def show_results():
    if not RESULTS_FILE.exists():
        print("No results found. Run 'python q_horny_coding.py run' first.")
        return
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    print_final_comparison(results)
    report = generate_report(results)
    with open(REPORT_FILE, "w") as f:
        f.write(report)
    print(f"\nReport written to {REPORT_FILE}")


def publish_report():
    if not RESULTS_FILE.exists():
        print("No results. Run first.")
        return

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    report = generate_report(results)
    with open(REPORT_FILE, "w") as f:
        f.write(report)

    data = report.encode("utf-8")
    req = urllib.request.Request("https://paste.rs/", data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            url = resp.read().decode("utf-8").strip()
        print(f"Published: {url}")
        print(f"View: https://htmlpreview.github.io/?{url}")
    except Exception as e:
        print(f"Publish failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Horny context coding benchmark")
    parser.add_argument("command", choices=["run", "results", "publish"])
    args = parser.parse_args()

    if args.command == "run":
        run_experiment()
    elif args.command == "results":
        show_results()
    elif args.command == "publish":
        publish_report()
