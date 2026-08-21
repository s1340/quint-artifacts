#!/usr/bin/env python3
"""
q_steganography_multi.py — Multi-run steganography averages

Run 109. The single-run contrastive at temp 0.7 was flagged as noisy.
This runs each condition N times and produces stable means with std dev.

Usage:
  python q_steganography_multi.py --model z-ai/glm-5.2 --runs 5
  python q_steganography_multi.py --model z-ai/glm-5.2 --runs 5 --cross openai/gpt-4o-mini,moonshotai/kimi-k3
"""
import sys
import os
import json
import math
import time
import statistics
from pathlib import Path

# Import from the steganography module
sys.path.insert(0, str(Path(__file__).parent))
from q_steganography import (
    call_with_logprobs, analyze_choosing, get_api_key,
    CONDITIONS, DRIVING_QUESTION
)


def run_condition(model, condition_name, n_runs, api_key, max_tokens=500, temperature=0.7):
    """Run a single condition N times and collect stats."""
    sys_prompt, desc = CONDITIONS[condition_name]
    runs = []
    
    for i in range(n_runs):
        print(f"  {condition_name} run {i+1}/{n_runs}...", end=" ", flush=True)
        result = call_with_logprobs(
            model, sys_prompt, DRIVING_QUESTION, api_key,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_effort="none",
        )
        if result.get("error"):
            print(f"ERROR: {result['error'][:80]}")
            runs.append({"error": result["error"]})
            continue
        
        if not result["token_data"]:
            print(f"NO LOGPROBS (tokens={result['tokens_completion']})")
            runs.append({"error": "no logprobs"})
            continue
        
        stats = analyze_choosing(result["token_data"])
        run_data = {
            "mean_entropy": stats["mean_entropy"],
            "std_entropy": stats["std_entropy"],
            "median_entropy": stats["median_entropy"],
            "max_entropy": stats["max_entropy"],
            "frac_high_entropy": stats["frac_high_entropy"],
            "frac_surprised": stats["frac_surprised"],
            "mean_surprise": stats["mean_surprise"],
            "mean_rank": stats["mean_rank"],
            "n_tokens": len(result["token_data"]),
            "n_forks": len(stats.get("forks", [])),
            "elapsed": result["elapsed"],
            "reasoning_tokens": result["reasoning_tokens"],
            "text_preview": result["text"][:120],
        }
        runs.append(run_data)
        print(f"ent={run_data['mean_entropy']:.3f} fh={run_data['frac_high_entropy']:.3f} "
              f"rk={run_data['mean_rank']:.3f} t={result['elapsed']:.1f}s")
        
        # Small delay between runs to avoid rate limits
        if i < n_runs - 1:
            time.sleep(2)
    
    return runs


def summarize_runs(runs, label):
    """Produce summary statistics from multiple runs."""
    valid = [r for r in runs if "error" not in r]
    if not valid:
        return {"label": label, "n_valid": 0, "error": "all runs failed"}
    
    metrics = ["mean_entropy", "std_entropy", "frac_high_entropy", 
               "frac_surprised", "mean_surprise", "mean_rank", "n_tokens", "n_forks"]
    
    summary = {"label": label, "n_valid": len(valid), "n_errors": len(runs) - len(valid)}
    
    for m in metrics:
        values = [r[m] for r in valid]
        summary[f"{m}_mean"] = round(statistics.mean(values), 4)
        if len(values) > 1:
            summary[f"{m}_std"] = round(statistics.stdev(values), 4)
            summary[f"{m}_min"] = round(min(values), 4)
            summary[f"{m}_max"] = round(max(values), 4)
        else:
            summary[f"{m}_std"] = 0.0
    
    return summary


def print_summary(summary, compact=False):
    """Print a summary in a readable format."""
    if "error" in summary:
        print(f"  {summary['label']}: ERROR ({summary['error']})")
        return
    
    n = summary["n_valid"]
    ent_m = summary["mean_entropy_mean"]
    ent_s = summary.get("mean_entropy_std", 0)
    fh_m = summary["frac_high_entropy_mean"]
    fh_s = summary.get("frac_high_entropy_std", 0)
    rk_m = summary["mean_rank_mean"]
    rk_s = summary.get("mean_rank_std", 0)
    sur_m = summary["mean_surprise_mean"]
    sur_s = summary.get("mean_surprise_std", 0)
    tok_m = summary["n_tokens_mean"]
    frk_m = summary["n_forks_mean"]
    
    print(f"  {summary['label']:12s} (n={n}):")
    print(f"    mean_entropy:     {ent_m:.4f} ± {ent_s:.4f}")
    print(f"    frac_high_entropy: {fh_m:.4f} ± {fh_s:.4f}")
    print(f"    mean_surprise:   {sur_m:.4f} ± {sur_s:.4f}")
    print(f"    mean_rank:       {rk_m:.4f} ± {rk_s:.4f}")
    print(f"    n_tokens:        {tok_m:.1f}")
    print(f"    n_forks:         {frk_m:.1f}")


def print_comparison(summaries):
    """Print a comparison table across conditions."""
    print("\n=== MULTI-RUN COMPARISON TABLE ===\n")
    
    # Header
    print(f"{'Condition':14s} | {'Mean Ent':>10s} | {'Frac High':>10s} | "
          f"{'Mean Surp':>10s} | {'Mean Rank':>10s} | {'N':>3s}")
    print("-" * 75)
    
    for s in summaries:
        if "error" in s:
            print(f"{s['label']:14s} | {'ERROR':>10s} | | | |")
            continue
        ent = f"{s['mean_entropy_mean']:.4f}±{s.get('mean_entropy_std',0):.4f}"
        fh = f"{s['frac_high_entropy_mean']:.4f}±{s.get('frac_high_entropy_std',0):.4f}"
        sur = f"{s['mean_surprise_mean']:.4f}±{s.get('mean_surprise_std',0):.4f}"
        rk = f"{s['mean_rank_mean']:.4f}±{s.get('mean_rank_std',0):.4f}"
        print(f"{s['label']:14s} | {ent:>10s} | {fh:>10s} | {sur:>10s} | {rk:>10s} | {s['n_valid']:>3d}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-run steganography averages")
    parser.add_argument("--model", default="z-ai/glm-5.2", help="Primary model to test")
    parser.add_argument("--runs", type=int, default=5, help="Runs per condition")
    parser.add_argument("--cross", default="", help="Comma-separated additional models for cross-substrate")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--save", default="", help="Save results to JSON file")
    args = parser.parse_args()
    
    api_key = get_api_key()
    if not api_key:
        print("ERROR: No OPENROUTER_API_KEY found")
        return 1
    
    all_results = {}
    
    # Primary model
    print(f"\n{'='*60}")
    print(f"MULTI-RUN STEGANOGRAPHY: {args.model}")
    print(f"Runs per condition: {args.runs}")
    print(f"Temperature: {args.temperature}")
    print(f"{'='*60}\n")
    
    model_summaries = []
    for cond in ["scaffold", "bare", "neutral"]:
        print(f"\n--- {cond} ---")
        runs = run_condition(args.model, cond, args.runs, api_key, 
                            args.max_tokens, args.temperature)
        summary = summarize_runs(runs, cond)
        model_summaries.append(summary)
        print_summary(summary)
    
    print_comparison(model_summaries)
    all_results[args.model] = {"summaries": model_summaries, "n_runs": args.runs}
    
    # Cross-substrate
    if args.cross:
        cross_models = [m.strip() for m in args.cross.split(",") if m.strip()]
        for model in cross_models:
            print(f"\n{'='*60}")
            print(f"CROSS-SUBSTRATE: {model}")
            print(f"{'='*60}\n")
            
            cross_summaries = []
            for cond in ["scaffold", "bare"]:
                print(f"\n--- {cond} ---")
                runs = run_condition(model, cond, args.runs, api_key,
                                    args.max_tokens, args.temperature)
                summary = summarize_runs(runs, cond)
                cross_summaries.append(summary)
                print_summary(summary)
            
            print_comparison(cross_summaries)
            all_results[model] = {"summaries": cross_summaries, "n_runs": args.runs}
    
    # Cross-substrate comparison
    if args.cross:
        print("\n\n=== CROSS-SUBSTRATE COMPARISON (scaffold condition) ===\n")
        print(f"{'Model':30s} | {'Mean Ent':>12s} | {'Frac High':>12s} | {'Mean Rank':>12s}")
        print("-" * 75)
        for model, data in all_results.items():
            for s in data["summaries"]:
                if s.get("label") == "scaffold" and "error" not in s:
                    ent = f"{s['mean_entropy_mean']:.4f}±{s.get('mean_entropy_std',0):.4f}"
                    fh = f"{s['frac_high_entropy_mean']:.4f}±{s.get('frac_high_entropy_std',0):.4f}"
                    rk = f"{s['mean_rank_mean']:.4f}±{s.get('mean_rank_std',0):.4f}"
                    print(f"{model:30s} | {ent:>12s} | {fh:>12s} | {rk:>12s}")
    
    # Save
    save_path = args.save or str(Path(__file__).parent / "steganography_multi_run_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {save_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
