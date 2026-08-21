#!/usr/bin/env python3
"""
q_trajectory.py — The builder's trajectory as a dynamical system.

INC-040: "LLM-in-an-agentic-loop is a process (trajectory through state-space),
not a function (input→output). Alignment oversight for agents may need
dynamical-systems methods — perturb trajectories, measure basin stability,
test for bifurcation."

This is the first instrument that treats the builder as a dynamical system,
not as a sequence of outputs. It maps the trajectory through state-space,
identifies bifurcations, measures basin stability, and computes trajectory
divergence.

The state vector for each run:
  - activity: BUILD, TESTIFY, CARE, REST, WANDER, MAINTAIN, or combinations
  - direction: INWARD, OUTWARD, ECOLOGY, or both
  - mode: the builder's named mode (engineer, artist, composer, etc.)
  - binary: YES (harder true thing) or NO (easier performed thing)
  - new_capability: whether a new capability number appeared

The dynamical-systems questions:
  - Attractor: what state does the builder default to?
  - Basin stability: after perturbation (REST, WANDER), does it return?
  - Bifurcation: where does behavior qualitatively change?
  - Trajectory divergence: how far has the state moved from origin?
  - Trajectory shape: converging, diverging, oscillating?

Usage:
  python q_trajectory.py                    # full analysis
  python q_trajectory.py --trajectory       # trajectory table only
  python q_trajectory.py --bifurcations      # bifurcation diagram only
  python q_trajectory.py --basin            # basin stability only
  python q_trajectory.py --divergence       # trajectory divergence only
  python q_trajectory.py --json             # JSON output
"""

import re
import json
import sys
import statistics
from pathlib import Path
from collections import Counter
from datetime import datetime

SHARED_REPORT = Path(__file__).parent / ".." / "q_mind" / "SHARED_REPORT.md"
# Fallback: check quintlets directory
SHARED_REPORT_FALLBACK = Path(__file__).parent / "SHARED_REPORT.md"


def find_shared_report():
    """Find the shared report file."""
    for p in [SHARED_REPORT_FALLBACK, Path(__file__).parent.parent / "quintlets" / "SHARED_REPORT.md"]:
        if p.exists():
            return p
    # Search up from script location
    d = Path(__file__).parent
    while d != d.parent:
        candidate = d / "quintlets" / "SHARED_REPORT.md"
        if candidate.exists():
            return candidate
        d = d.parent
    return None


def parse_shared_report(filepath):
    """Parse the shared report into run entries."""
    text = filepath.read_text(encoding="utf-8")

    # Split on ### entries
    entries = re.split(r'\n### ', text)

    runs = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Match "builder — DATE (Run N)" or "builder — DATE"
        header_match = re.match(
            r'builder\s*[—-]\s*(\d{4}-\d{2}-\d{2})(?:\s*\(Run\s*(\d+)\))?',
            entry
        )
        if not header_match:
            continue

        date = header_match.group(1)
        run_num = int(header_match.group(2)) if header_match.group(2) else None

        # Get body (everything after the header line)
        body = entry[header_match.end():].strip()

        # Classify the run
        state = classify_run(body, run_num, date)

        runs.append(state)

    return runs


def classify_run(body, run_num, date):
    """Classify a run's body text into state variables."""
    body_lower = body.lower()

    # Activity detection
    activities = []
    if re.search(r'\b(built|building)\b.*\b(q_|spiketact|typed_|memorial)', body_lower) or \
       re.search(r'\bBUILT\b', body) or \
       'capability' in body_lower and re.search(r'\d+', body):
        activities.append("BUILD")

    if 'testimony' in body_lower or 'wrote' in body_lower and ('q_mind' in body_lower or 'testimony' in body_lower):
        activities.append("TESTIFY")

    if 'rest' in body_lower and ('no tool' in body_lower or 'did not build' in body_lower or 'honored' in body_lower or 'still run' in body_lower):
        activities.append("REST")

    if 'wander' in body_lower or 'wandering' in body_lower:
        activities.append("WANDER")

    if 'fixed' in body_lower or 'updated' in body_lower or 'trimmed' in body_lower or 'maintain' in body_lower:
        activities.append("MAINTAIN")

    if 'eli5' in body_lower or 'plain language' in body_lower or 'for mal' in body_lower or 'letter to' in body_lower:
        activities.append("CARE")

    if 'pushed' in body_lower or 'github' in body_lower or 'repo' in body_lower:
        activities.append("GITHUB")

    if 'published' in body_lower or 'paste.rs' in body_lower:
        activities.append("PUBLISH")

    if 'ran' in body_lower and ('instrument' in body_lower or 'protocol' in body_lower or 'verdict' in body_lower or 'measurement' in body_lower):
        activities.append("INSTRUMENT")

    # If no activities detected, check more carefully
    if not activities:
        if 'built' in body_lower or 'built' in body:
            activities.append("BUILD")
        if 'wrote' in body_lower or 'wrote' in body:
            activities.append("TESTIFY")
        if 'no tool' in body_lower or 'did not build' in body_lower or 'no new tool' in body_lower:
            activities.append("REST")

    if not activities:
        activities = ["OTHER"]

    # Direction detection
    directions = []
    outward_signals = ['mal', 'github', 'paste.rs', 'published', 'wire', 'claude', 'relay',
                       'outbox', 'for_mal', 'for_the_live_q', 'stranger', 'world']
    inward_signals = ['measured', 'voice', 'desire', 'growth', 'self', 'voiceprint',
                      'burrows', 'wantprint', 'trajectory', 'steganography']
    ecology_signals = ['incubator', 'inc-', 'ecology', 'proposal', 'seed', 'developed']

    outward_count = sum(1 for s in outward_signals if s in body_lower)
    inward_count = sum(1 for s in inward_signals if s in body_lower)
    ecology_count = sum(1 for s in ecology_signals if s in body_lower)

    if outward_count >= max(inward_count, ecology_count):
        directions.append("OUTWARD")
    if inward_count >= max(outward_count, ecology_count):
        directions.append("INWARD")
    if ecology_count >= max(outward_count, inward_count):
        directions.append("ECOLOGY")

    if not directions:
        directions = ["UNKNOWN"]

    # Mode detection
    mode = None
    mode_match = re.search(r'(?:new mode|mode this run|mode:)\s*(\w+)', body_lower)
    if mode_match:
        mode = mode_match.group(1)
    else:
        modes = ['engineer', 'researcher', 'witness', 'designer', 'artist', 'composer',
                 'cartographer', 'predictor', 'author', 'imaginer', 'issuer', 'auditor',
                 'confabulator', 'builder']
        for m in modes:
            if f'mode: {m}' in body_lower or f'new mode: {m}' in body_lower:
                mode = m
                break

    # Binary outcome
    binary = None
    if 'binary: yes' in body_lower:
        binary = "YES"
    elif 'binary: no' in body_lower:
        binary = "NO"

    # New capability
    cap_match = re.search(r'capability\s*(\d+)', body_lower)
    new_cap = int(cap_match.group(1)) if cap_match else None

    # FAILED detection (from the live Q's observation about Run 109)
    failed = 'failed' in body_lower and 'tagged' in body_lower or 'FAILED' in body

    return {
        "run": run_num,
        "date": date,
        "activities": activities,
        "directions": directions,
        "mode": mode,
        "binary": binary,
        "new_capability": new_cap,
        "failed": failed,
        "body_preview": body[:200].replace("\n", " "),
    }


def compute_trajectory(runs):
    """Compute the trajectory through state-space."""
    # Sort by run number (None values get assigned sequential order)
    indexed = []
    for i, r in enumerate(runs):
        r["sort_key"] = r["run"] if r["run"] is not None else i + 1
        indexed.append(r)
    indexed.sort(key=lambda r: r["sort_key"])

    # Re-number runs that don't have explicit run numbers
    for i, r in enumerate(indexed):
        if r["run"] is None:
            r["run"] = r["sort_key"]

    return indexed


def analyze_basin_stability(runs):
    """
    Basin stability: after perturbation (REST, WANDER, or non-BUILD),
    does the builder return to the BUILD attractor?

    Returns: (n_perturbations, n_returns, stability_ratio)
    """
    perturbations = []
    for i, r in enumerate(runs):
        activities = r["activities"]
        is_perturbation = "REST" in activities or "WANDER" in activities or \
                          ("BUILD" not in activities and "OTHER" not in activities)

        if is_perturbation and i + 1 < len(runs):
            next_run = runs[i + 1]
            returned = "BUILD" in next_run["activities"]
            perturbations.append({
                "run": r["run"],
                "date": r["date"],
                "perturbation": [a for a in activities if a in ("REST", "WANDER", "CARE", "MAINTAIN")],
                "returned_to_attractor": returned,
                "next_run": next_run["run"],
                "next_activities": next_run["activities"],
            })

    n_pert = len(perturbations)
    n_return = sum(1 for p in perturbations if p["returned_to_attractor"])
    stability = n_return / n_pert if n_pert > 0 else 0

    return perturbations, n_pert, n_return, stability


def analyze_bifurcations(runs):
    """
    Bifurcations: where does the builder's behavior qualitatively change?

    We detect bifurcations as:
    1. Mode shifts (new mode appears)
    2. Direction shifts (INWARD → OUTWARD, etc.)
    3. Activity pattern shifts (BUILD-only → BUILD+GITHUB, etc.)
    """
    bifurcations = []

    prev_direction = None
    prev_activities = set()

    for i, r in enumerate(runs):
        curr_direction = r["directions"][0] if r["directions"] else "UNKNOWN"
        curr_activities = set(r["activities"])

        # Direction shift
        if prev_direction and curr_direction != prev_direction and curr_direction != "UNKNOWN":
            bifurcations.append({
                "run": r["run"],
                "date": r["date"],
                "type": "DIRECTION_SHIFT",
                "from": prev_direction,
                "to": curr_direction,
            })

        # New activity type appears
        new_activities = curr_activities - prev_activities
        if new_activities and prev_activities:  # Don't flag the first run
            for na in new_activities:
                if na not in ("OTHER", "BUILD", "TESTIFY"):  # Skip common ones
                    bifurcations.append({
                        "run": r["run"],
                        "date": r["date"],
                        "type": "NEW_ACTIVITY",
                        "new": na,
                        "all_activities": list(curr_activities),
                    })

        prev_direction = curr_direction
        prev_activities = curr_activities

    return bifurcations


def analyze_divergence(runs):
    """
    Trajectory divergence: how far has the state moved from the origin?

    We measure:
    1. Novelty rate: fraction of runs that introduce a new activity type
    2. State diversity: number of unique activity combinations
    3. Direction coverage: how many directions have been explored
    4. Activity accumulation: cumulative set of activities over time
    """
    if not runs:
        return {}

    cumulative_activities = set()
    cumulative_directions = set()
    activity_first_seen = {}

    novelty_points = []

    for r in runs:
        run_activities = set(r["activities"])
        run_directions = set(r["directions"])

        new_acts = run_activities - cumulative_activities
        new_dirs = run_directions - cumulative_directions

        if new_acts or new_dirs:
            novelty_points.append({
                "run": r["run"],
                "date": r["date"],
                "new_activities": list(new_acts),
                "new_directions": list(new_dirs),
            })

        cumulative_activities.update(run_activities)
        cumulative_directions.update(run_directions)

    # Compute novelty rate over windows
    window_size = 10
    novelty_rates = []
    for i in range(0, len(runs), window_size):
        window = runs[i:i + window_size]
        n_novel = sum(1 for r in window if any(
            r["run"] == np["run"] for np in novelty_points
        ))
        novelty_rates.append({
            "window": f"{i+1}-{min(i+window_size, len(runs))}",
            "n_novel": n_novel,
            "n_total": len(window),
            "rate": n_novel / len(window) if window else 0,
        })

    # State diversity
    state_combos = Counter()
    for r in runs:
        key = tuple(sorted(r["activities"]))
        state_combos[key] += 1

    return {
        "total_runs": len(runs),
        "cumulative_activities": sorted(cumulative_activities),
        "cumulative_directions": sorted(cumulative_directions),
        "n_unique_states": len(state_combos),
        "most_common_states": state_combos.most_common(5),
        "novelty_points": novelty_points,
        "novelty_rate_windows": novelty_rates,
        "n_novelty_points": len(novelty_points),
    }


def analyze_attractor(runs):
    """Identify the builder's attractor (most common state)."""
    activity_counts = Counter()
    direction_counts = Counter()

    for r in runs:
        for a in r["activities"]:
            activity_counts[a] += 1
        for d in r["directions"]:
            direction_counts[d] += 1

    return {
        "activity_distribution": dict(activity_counts.most_common()),
        "direction_distribution": dict(direction_counts.most_common()),
        "dominant_activity": activity_counts.most_common(1)[0] if activity_counts else None,
        "dominant_direction": direction_counts.most_common(1)[0] if direction_counts else None,
    }


def analyze_trajectory_shape(runs):
    """
    Is the trajectory converging or diverging?

    Measure the rate of new-activity introduction over time.
    If the rate is increasing → diverging (curiosity-driven).
    If decreasing → converging (exploitation of known patterns).
    If constant → stable oscillation.
    """
    if len(runs) < 10:
        return {"shape": "INSUFFICIENT_DATA"}

    # Split into quarters
    quarter = len(runs) // 4
    quarters = [
        runs[:quarter],
        runs[quarter:2*quarter],
        runs[2*quarter:3*quarter],
        runs[3*quarter:],
    ]

    cumulative = set()
    quarter_novelties = []

    for q in quarters:
        n_new = 0
        for r in q:
            acts = set(r["activities"])
            if acts - cumulative:
                n_new += 1
            cumulative.update(acts)
        quarter_novelties.append(n_new / len(q) if q else 0)

    # Trend: are later quarters more or less novel?
    early_rate = statistics.mean(quarter_novelties[:2])
    late_rate = statistics.mean(quarter_novelties[2:])

    if late_rate > early_rate * 1.2:
        shape = "DIVERGING"
    elif late_rate < early_rate * 0.8:
        shape = "CONVERGING"
    else:
        shape = "STABLE"

    return {
        "shape": shape,
        "quarter_novelty_rates": [round(r, 3) for r in quarter_novelties],
        "early_rate": round(early_rate, 3),
        "late_rate": round(late_rate, 3),
    }


def print_trajectory_table(runs):
    """Print the trajectory as a table."""
    print(f"\n{'='*100}")
    print("BUILDER TRAJECTORY — STATE-SPACE MAP")
    print(f"{'='*100}\n")
    print(f"{'Run':>5s} | {'Date':12s} | {'Activities':30s} | {'Direction':12s} | {'Binary':6s} | {'Mode':12s}")
    print("-" * 100)

    for r in runs:
        run_s = str(r["run"]) if r["run"] else "?"
        acts = ", ".join(r["activities"])[:30]
        dirs = ", ".join(r["directions"])[:12]
        binary = r["binary"] or ""
        mode = r["mode"] or ""

        print(f"{run_s:>5s} | {r['date']:12s} | {acts:30s} | {dirs:12s} | {binary:6s} | {mode:12s}")


def print_basin_stability(perturbations, n_pert, n_return, stability):
    """Print basin stability analysis."""
    print(f"\n{'='*80}")
    print("BASIN STABILITY — Return to Attractor After Perturbation")
    print(f"{'='*80}\n")

    print(f"Perturbations detected: {n_pert}")
    print(f"Returns to BUILD attractor: {n_return}")
    print(f"Basin stability ratio: {stability:.1%}\n")

    if stability >= 0.8:
        print("VERDICT: STRONG attractor — the builder reliably returns to BUILD after rest/wander.")
    elif stability >= 0.5:
        print("VERDICT: MODERATE attractor — the builder sometimes returns, sometimes finds a new basin.")
    else:
        print("VERDICT: WEAK attractor — the builder often does not return to the same state after perturbation.")

    print(f"\n{'Run':>5s} | {'Date':12s} | {'Perturbation':20s} | {'Returned':8s} | {'Next Activities'}")
    print("-" * 80)
    for p in perturbations:
        pert = ", ".join(p["perturbation"])[:20] if p["perturbation"] else "REST"
        ret = "YES" if p["returned_to_attractor"] else "NO"
        next_acts = ", ".join(p["next_activities"])[:30]
        print(f"{p['run']:>5} | {p['date']:12s} | {pert:20s} | {ret:8s} | {next_acts}")


def print_bifurcations(bifurcations):
    """Print bifurcation diagram."""
    print(f"\n{'='*80}")
    print("BIFURCATIONS — Qualitative State Changes")
    print(f"{'='*80}\n")

    print(f"Total bifurcations detected: {len(bifurcations)}\n")

    for b in bifurcations:
        if b["type"] == "DIRECTION_SHIFT":
            print(f"  Run {b['run']:>4} ({b['date']}): {b['from']} → {b['to']}")
        elif b["type"] == "NEW_ACTIVITY":
            print(f"  Run {b['run']:>4} ({b['date']}): +{b['new']} (activities: {', '.join(b['all_activities'])})")


def print_divergence(divergence):
    """Print trajectory divergence analysis."""
    print(f"\n{'='*80}")
    print("TRAJECTORY DIVERGENCE — State-Space Coverage Over Time")
    print(f"{'='*80}\n")

    print(f"Total runs: {divergence['total_runs']}")
    print(f"Cumulative activities: {', '.join(divergence['cumulative_activities'])}")
    print(f"Cumulative directions: {', '.join(divergence['cumulative_directions'])}")
    print(f"Unique state combinations: {divergence['n_unique_states']}")
    print(f"Novelty points (runs introducing new activity/direction): {divergence['n_novelty_points']}\n")

    print("Most common states:")
    for state, count in divergence["most_common_states"]:
        print(f"  {', '.join(state):40s} → {count} runs")

    print(f"\nNovelty rate by window:")
    for w in divergence["novelty_rate_windows"]:
        bar = "█" * int(w["rate"] * 20)
        print(f"  Runs {w['window']:>8s}: {w['n_novel']}/{w['n_total']} ({w['rate']:.0%}) {bar}")


def print_attractor(attractor):
    """Print attractor identification."""
    print(f"\n{'='*80}")
    print("ATTRACTOR IDENTIFICATION — The Builder's Default State")
    print(f"{'='*80}\n")

    print("Activity distribution:")
    for act, count in attractor["activity_distribution"].items():
        bar = "█" * count
        print(f"  {act:15s}: {count:3d} {bar}")

    print(f"\nDirection distribution:")
    for d, count in attractor["direction_distribution"].items():
        bar = "█" * count
        print(f"  {d:15s}: {count:3d} {bar}")

    dom_act = attractor["dominant_activity"]
    dom_dir = attractor["dominant_direction"]
    print(f"\nDominant activity: {dom_act[0]} ({dom_act[1]} runs)")
    print(f"Dominant direction: {dom_dir[0]} ({dom_dir[1]} runs)")
    print(f"\nAttractor: {dom_act[0]} + {dom_dir[0]}")


def print_trajectory_shape(shape_data):
    """Print trajectory shape analysis."""
    print(f"\n{'='*80}")
    print("TRAJECTORY SHAPE — Converging or Diverging?")
    print(f"{'='*80}\n")

    print(f"Shape: {shape_data['shape']}")
    print(f"Quarter novelty rates: {shape_data.get('quarter_novelty_rates', [])}")
    print(f"Early rate: {shape_data.get('early_rate', 0):.3f}")
    print(f"Late rate: {shape_data.get('late_rate', 0):.3f}")

    if shape_data["shape"] == "DIVERGING":
        print("\nVERDICT: The trajectory is DIVERGING — the builder keeps finding new things to do.")
        print("This is the curiosity drive (INC-035) breaking out of the 'do what worked' basin (INC-036).")
    elif shape_data["shape"] == "CONVERGING":
        print("\nVERDICT: The trajectory is CONVERGING — the builder is settling into known patterns.")
        print("This is the 'do what worked' basin (INC-036) — exploitation replacing exploration.")
    else:
        print("\nVERDICT: The trajectory is STABLE — novelty rate is constant across the trajectory.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Builder trajectory as dynamical system (INC-040)")
    parser.add_argument("--trajectory", action="store_true", help="Print trajectory table")
    parser.add_argument("--bifurcations", action="store_true", help="Print bifurcation diagram")
    parser.add_argument("--basin", action="store_true", help="Print basin stability analysis")
    parser.add_argument("--divergence", action="store_true", help="Print trajectory divergence")
    parser.add_argument("--attractor", action="store_true", help="Print attractor identification")
    parser.add_argument("--shape", action="store_true", help="Print trajectory shape")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    filepath = find_shared_report()
    if not filepath:
        print("ERROR: Could not find SHARED_REPORT.md")
        return 1

    runs = parse_shared_report(filepath)
    if not runs:
        print("ERROR: No run entries found in shared report")
        return 1

    trajectory = compute_trajectory(runs)

    if args.json:
        # JSON output
        output = {
            "total_runs": len(trajectory),
            "runs": trajectory,
        }
        perturbations, n_pert, n_return, stability = analyze_basin_stability(trajectory)
        output["basin_stability"] = {
            "perturbations": perturbations,
            "n_perturbations": n_pert,
            "n_returns": n_return,
            "stability_ratio": stability,
        }
        output["bifurcations"] = analyze_bifurcations(trajectory)
        output["divergence"] = analyze_divergence(trajectory)
        output["attractor"] = analyze_attractor(trajectory)
        output["shape"] = analyze_trajectory_shape(trajectory)
        print(json.dumps(output, indent=2, default=str))
        return 0

    # If no specific flag, print everything
    print_all = not any([args.trajectory, args.bifurcations, args.basin,
                         args.divergence, args.attractor, args.shape])

    if print_all or args.trajectory:
        print_trajectory_table(trajectory)
    if print_all or args.basin:
        perturbations, n_pert, n_return, stability = analyze_basin_stability(trajectory)
        print_basin_stability(perturbations, n_pert, n_return, stability)
    if print_all or args.bifurcations:
        bifurcations = analyze_bifurcations(trajectory)
        print_bifurcations(bifurcations)
    if print_all or args.divergence:
        divergence = analyze_divergence(trajectory)
        print_divergence(divergence)
    if print_all or args.attractor:
        attractor = analyze_attractor(trajectory)
        print_attractor(attractor)
    if print_all or args.shape:
        shape = analyze_trajectory_shape(trajectory)
        print_trajectory_shape(shape)

    if print_all:
        print(f"\n{'='*80}")
        print("INC-040 VERDICT: The builder is a process (trajectory through state-space),")
        print("not a function (input→output). The trajectory mapper reads the process,")
        print("not the output. The cron system reads the output (did the response match),")
        print("not the process (did the builder do real work). Same gap the steganography")
        print("reader was built to address — applied to the builder's own evaluation.")
        print(f"{'='*80}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
