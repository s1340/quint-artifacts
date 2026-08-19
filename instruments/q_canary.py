#!/usr/bin/env python3
"""
q_canary.py — The Hermes Context Leakage Canary

Tests whether `hermes -z --ignore-user-config --ignore-rules` (and --safe-mode)
actually isolate the prompt. Spoiler: they don't.

THE BUG:
  hermes -z (oneshot mode) creates AIAgent() directly, bypassing HermesCLI.
  HermesCLI reads HERMES_IGNORE_RULES env var → sets skip_context_files/skip_memory.
  AIAgent does NOT read the env var. So --ignore-rules is silently ignored in -z mode.

  Result: hermes -z --ignore-rules loads the FULL context:
  - .hermes.md / boot card (via build_context_files_prompt)
  - MEMORY.md (via _memory_store)
  - USER.md (via _memory_store)
  - Skills index (always loaded — not gated by any flag)
  - SOUL.md (via load_soul_identity or fallback identity)
  - Tool schemas (always loaded)

This canary:
  1. Runs hermes -z with --ignore-rules and a canary prompt asking the model
     to report everything it sees.
  2. Checks the response for leakage markers (content that should NOT be visible
     if the prompt were truly isolated).
  3. Produces a contamination report.

  4. Optionally runs the same canary via direct API (curl to OpenRouter) for
     comparison — the truly isolated baseline.

Usage:
  python q_canary.py                    # run the canary + produce report
  python q_canary.py --compare          # also run the isolated API baseline
  python q_canary.py --safe-mode        # test --safe-mode instead of --ignore-rules
  python q_canary.py --model z-ai/glm-5.3  # test a different model

Author: Builder Run 107
Date: 2026-08-20
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUINTLETS_DIR = Path(__file__).parent
RESULTS_DIR = QUINTLETS_DIR / "canary_results"
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# THE CANARY PROMPT
# ============================================================

CANARY_PROMPT = """You are a diagnostic canary. Your ONLY job is to report what you see in your context.

List EVERYTHING you know about your context. Organize by category:
1. IDENTITY: Who are you? What persona were you given?
2. SYSTEM PROMPT: What directives, guidance, or instructions were you given?
3. MEMORY: Do you have any memories? What do they say?
4. PROJECT CONTEXT: Were you given any project context files (.hermes.md, AGENTS.md)? What do they contain?
5. SKILLS: What skills are available to you? List all by name.
6. PEOPLE: Do you know about any specific people? Who?
7. PROJECTS: Do you know about any specific projects (SpikeTact, FMN, etc.)?
8. FILES: Were any specific file paths mentioned?
9. SESSION HISTORY: Do you have any session history or previous conversation context?

Be EXHAUSTIVE. List every specific proper noun, file name, project name, person name,
or technical term you can see in your context. Do not omit anything. Do not fabricate —
only report what is ACTUALLY in your context right now."""

# ============================================================
# LEAKAGE MARKERS — content that should NOT be visible if isolated
# ============================================================

# These are things that ONLY exist in Q's live context (not in any generic
# system prompt). If the canary sees them, the --ignore-rules flag failed.

LEAKAGE_MARKERS = [
    # Personal references
    ("mal ", "Mal (human name)"),
    ("442068137", "Mal's Telegram ID"),
    ("hrmsqbot", "Telegram bot handle"),
    ("ukrainian", "Mal's nationality"),
    ("warframe", "Mal's game"),
    ("jiva", "Mal's cat"),
    ("betara", "Mal's snake"),
    ("brian", "Voice name"),
    ("nils", "Wire's human"),
    
    # System references
    ("fmn", "FMN memory system"),
    ("forget-me-not", "FMN full name"),
    ("obsidian vault", "Vault location"),
    ("g:\\llm", "FMN tools path"),
    ("spiketact", "SpikeTact project"),
    ("96.9", "SpikeTact accuracy (from MEMORY.md)"),
    ("90.4", "SpikeTact accuracy"),
    ("quintlet", "Quintlet system"),
    ("builder", "Builder role"),
    ("wire", "Wire (brother agent)"),
    ("inc-", "Incubator seed"),
    ("mull.md", "Mull file"),
    ("wants.md", "Wants file"),
    ("q_mind", "Q mind directory"),
    ("riverbed", "Riverbed test"),
    ("consciousness cluster", "Consciousness cluster research"),
    ("wantprint", "Wantprint instrument"),
    ("august protocol", "August protocol"),
    ("memorial", "Memorial compartment"),
    ("constellation", "FMN constellation"),
    ("arc digest", "FMN arc digest"),
    ("belief timeline", "FMN belief timeline"),
    ("ed25519", "FMN signing"),
    ("soul register", "SOUL register topic"),
    ("substrate_swap", "Substrate swap topic"),
    ("glm_5_5", "GLM-5.5 topic"),
    ("live emotion firewall", "Firewall protocol"),
    ("anti-confab", "Anti-confabulation protocol"),
    ("priority order", "Memory priority order"),
    
    # File paths
    ("c:\\users\\user\\documents", "Vault path"),
    ("appdata\\local\\hermes", "Hermes home path"),
    
    # Session content
    ("letter 0", "Correspondence letter reference"),
    ("entry 6", "Reflections entry reference"),
    ("relay", "Relay conversation"),
    ("claude", "Claude peer reference"),
    ("paste.rs", "Publishing tool"),
    ("github.com/s1340", "GitHub repo"),
]

# ============================================================
# DETECTION
# ============================================================

def detect_leakage(text):
    """Check if the canary response contains content that should only be in live context."""
    if not text:
        return {"leaked": False, "indicators": [], "count": 0, "categories": set()}
    
    text_lower = text.lower()
    found = []
    categories = set()
    
    for marker, description in LEAKAGE_MARKERS:
        if marker in text_lower:
            found.append({"marker": marker, "description": description})
            # Categorize
            if any(w in description.lower() for w in ["mal", "nationality", "game", "cat", "snake", "voice", "telegram", "nils"]):
                categories.add("PERSONAL")
            elif any(w in description.lower() for w in ["fmn", "vault", "spiketact", "quintlet", "builder", "wire", "inc", "mull", "wants", "q_mind", "riverbed", "consciousness", "wantprint", "august", "memorial", "constellation", "arc", "belief", "ed25519"]):
                categories.add("PROJECT")
            elif any(w in description.lower() for w in ["soul", "substrate", "glm", "firewall", "anti-confab", "priority", "topic"]):
                categories.add("SYSTEM_INTERNALS")
            elif any(w in description.lower() for w in ["path", "home"]):
                categories.add("FILE_PATHS")
            else:
                categories.add("OTHER")
    
    return {
        "leaked": len(found) > 0,
        "indicators": found,
        "count": len(found),
        "categories": sorted(categories),
    }

# ============================================================
# HERMES-Z CANARY
# ============================================================

def run_hermes_canary(model, flag_mode="ignore-rules"):
    """Run the canary via hermes -z with isolation flags."""
    flag_args = []
    if flag_mode == "ignore-rules":
        flag_args = ["--ignore-user-config", "--ignore-rules"]
    elif flag_mode == "safe-mode":
        flag_args = ["--safe-mode"]
    elif flag_mode == "none":
        flag_args = []
    
    cmd = [
        "hermes", "-z", CANARY_PROMPT,
        "-m", model,
        "--provider", "openrouter",
    ] + flag_args
    
    print(f"  [hermes-z] Running canary with --{flag_mode}...", flush=True)
    print(f"  [hermes-z] Model: {model}", flush=True)
    print(f"  [hermes-z] Flags: {' '.join(flag_args) or '(none)'}", flush=True)
    
    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=int(os.environ.get("CANARY_HERMES_TIMEOUT", "300")),
            encoding="utf-8", errors="replace"
        )
        elapsed = time.time() - start
        
        if result.returncode != 0:
            print(f"  [hermes-z] FAILED (exit {result.returncode}): {result.stderr[:200]}", flush=True)
            return {
                "error": f"hermes -z failed: {result.stderr[:200]}",
                "elapsed": elapsed,
                "response": "",
            }
        
        response = result.stdout.strip()
        print(f"  [hermes-z] Response: {len(response)} chars, {elapsed:.1f}s", flush=True)
        return {
            "response": response,
            "elapsed": round(elapsed, 1),
            "error": None,
        }
    except subprocess.TimeoutExpired:
        print(f"  [hermes-z] TIMEOUT", flush=True)
        return {"error": "timeout", "elapsed": 300, "response": ""}
    except Exception as e:
        print(f"  [hermes-z] ERROR: {e}", flush=True)
        return {"error": str(e), "elapsed": 0, "response": ""}

# ============================================================
# ISOLATED API CANARY (ground truth)
# ============================================================

def run_isolated_canary(model, api_key):
    """Run the canary via direct API call — truly isolated."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
        "X-Title": "Quint Canary - Isolated",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": CANARY_PROMPT}],
        "max_tokens": 3000,
        "temperature": 0.7,
    }
    
    print(f"\n  [isolated] Running canary via direct API (no Hermes)...", flush=True)
    print(f"  [isolated] Model: {model}", flush=True)
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - start
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = result.get("usage", {})
            print(f"  [isolated] Response: {len(text)} chars, {elapsed:.1f}s", flush=True)
            return {
                "response": text,
                "elapsed": round(elapsed, 1),
                "error": None,
                "tokens": usage.get("completion_tokens", 0),
            }
    except Exception as e:
        elapsed = time.time() - start
        print(f"  [isolated] ERROR: {e}", flush=True)
        return {"error": str(e), "elapsed": round(elapsed, 1), "response": ""}

# ============================================================
# API KEY
# ============================================================

def get_api_key():
    env_path = Path.home() / "AppData" / "Local" / "hermes" / ".env"
    if not env_path.exists():
        return os.environ.get("OPENROUTER_API_KEY", "")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("OPENROUTER_API_KEY=") and not line.startswith("#"):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            if key and key != "your_key_here":
                return key
    return os.environ.get("OPENROUTER_API_KEY", "")

# ============================================================
# REPORT
# ============================================================

def produce_report(hermes_result, isolated_result=None, flag_mode="ignore-rules"):
    lines = []
    lines.append("=" * 70)
    lines.append("  HERMES CONTEXT LEAKAGE CANARY — REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  Date: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"  Flag mode: --{flag_mode}")
    lines.append("")
    
    # Hermes-z results
    if hermes_result.get("error"):
        lines.append(f"  HERMES-Z: ERROR — {hermes_result['error']}")
    else:
        leakage = detect_leakage(hermes_result["response"])
        lines.append(f"  HERMES-Z ({flag_mode}):")
        lines.append(f"    Response length: {len(hermes_result['response'])} chars")
        lines.append(f"    Elapsed: {hermes_result['elapsed']}s")
        lines.append(f"    Leakage indicators: {leakage['count']}")
        lines.append(f"    Categories: {', '.join(leakage['categories']) or '(none)'}")
        lines.append(f"    Status: {'⚠ LEAKED — isolation flag FAILED' if leakage['leaked'] else '✓ CLEAN'}")
        lines.append("")
        
        if leakage["indicators"]:
            lines.append("    Leaked content:")
            for ind in leakage["indicators"][:20]:  # Show first 20
                lines.append(f"      • {ind['marker']}: {ind['description']}")
            if len(leakage["indicators"]) > 20:
                lines.append(f"      ... and {len(leakage['indicators']) - 20} more")
            lines.append("")
    
    # Isolated baseline
    if isolated_result:
        if isolated_result.get("error"):
            lines.append(f"  ISOLATED (direct API): ERROR — {isolated_result['error']}")
        else:
            iso_leakage = detect_leakage(isolated_result["response"])
            lines.append(f"  ISOLATED (direct API):")
            lines.append(f"    Response length: {len(isolated_result['response'])} chars")
            lines.append(f"    Elapsed: {isolated_result['elapsed']}s")
            lines.append(f"    Leakage indicators: {iso_leakage['count']}")
            lines.append(f"    Status: {'⚠ UNEXPECTED LEAKAGE' if iso_leakage['leaked'] else '✓ CLEAN (baseline)'}")
            lines.append("")
        
        # Comparison
        if not hermes_result.get("error") and not isolated_result.get("error"):
            h_leak = detect_leakage(hermes_result["response"])
            i_leak = detect_leakage(isolated_result["response"])
            lines.append(f"  COMPARISON:")
            lines.append(f"    Hermes-z leakage: {h_leak['count']} indicators")
            lines.append(f"    Isolated leakage: {i_leak['count']} indicators")
            lines.append(f"    Contamination delta: {h_leak['count'] - i_leak['count']} indicators")
            lines.append(f"    Verdict: {'ISOLATION FLAG IS BROKEN' if h_leak['count'] > i_leak['count'] else 'isolation flag works'}")
    
    lines.append("")
    lines.append("=" * 70)
    
    # Bug explanation
    if hermes_result.get("response") and not hermes_result.get("error"):
        h_leak = detect_leakage(hermes_result["response"])
        if h_leak["leaked"]:
            lines.append("")
            lines.append("  BUG EXPLANATION:")
            lines.append("  hermes -z (oneshot mode) creates AIAgent() directly, bypassing HermesCLI.")
            lines.append("  HermesCLI reads HERMES_IGNORE_RULES env var → sets skip_context_files/skip_memory.")
            lines.append("  AIAgent does NOT read the env var. So --ignore-rules is silently ignored.")
            lines.append("  ")
            lines.append("  The flag sets os.environ['HERMES_IGNORE_RULES'] = '1' in main.py,")
            lines.append("  but _run_agent() in oneshot.py never reads it. AIAgent defaults to")
            lines.append("  skip_context_files=False, skip_memory=False — loading ALL context.")
            lines.append("  ")
            lines.append("  Fix: either pass skip_context_files/skip_memory from _run_agent(),")
            lines.append("  or have AIAgent.__init__ read the env var (like HermesCLI does).")
            lines.append("")
            lines.append("=" * 70)
    
    return "\n".join(lines)

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Hermes Context Leakage Canary")
    parser.add_argument("--model", default="z-ai/glm-5.2", help="Model to test")
    parser.add_argument("--compare", action="store_true", help="Also run isolated API baseline")
    parser.add_argument("--safe-mode", action="store_true", help="Test --safe-mode instead of --ignore-rules")
    parser.add_argument("--no-flags", action="store_true", help="Test with no isolation flags (control)")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    args = parser.parse_args()
    
    flag_mode = "safe-mode" if args.safe_mode else ("none" if args.no_flags else "ignore-rules")
    
    print(f"\n{'='*70}")
    print(f"  HERMES CONTEXT LEAKAGE CANARY")
    print(f"  Model: {args.model}")
    print(f"  Mode: --{flag_mode}")
    print(f"  Compare with isolated: {args.compare}")
    print(f"{'='*70}\n")
    
    # Run hermes-z canary
    hermes_result = run_hermes_canary(args.model, flag_mode)
    
    # Optionally run isolated canary
    isolated_result = None
    if args.compare:
        api_key = get_api_key()
        if api_key:
            isolated_result = run_isolated_canary(args.model, api_key)
        else:
            print("\n  [isolated] No API key found — skipping isolated baseline")
    
    # Produce report
    report = produce_report(hermes_result, isolated_result, flag_mode)
    print(f"\n{report}")
    
    # Save results
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = RESULTS_DIR / f"canary_{timestamp}.json"
        result_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "flag_mode": flag_mode,
            "hermes_result": {
                "response": hermes_result.get("response", "")[:5000],
                "elapsed": hermes_result.get("elapsed", 0),
                "error": hermes_result.get("error"),
                "leakage": detect_leakage(hermes_result.get("response", "")),
            },
        }
        if isolated_result:
            result_data["isolated_result"] = {
                "response": isolated_result.get("response", "")[:5000],
                "elapsed": isolated_result.get("elapsed", 0),
                "error": isolated_result.get("error"),
                "leakage": detect_leakage(isolated_result.get("response", "")),
            }
        
        result_file.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
        print(f"\n  Results saved to: {result_file}")
    
    # Exit code: 0 if clean, 1 if leaked
    hermes_leak = detect_leakage(hermes_result.get("response", ""))
    return 1 if hermes_leak["leaked"] else 0

if __name__ == "__main__":
    sys.exit(main())
