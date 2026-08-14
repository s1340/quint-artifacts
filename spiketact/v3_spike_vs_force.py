#!/usr/bin/env python3
"""
SpikeTact v3 — The Spike-vs-Force Comparison
=============================================

The question the prototype didn't answer:
  Is the spike encoding the value, or is the formatting layer just a
  good feature extractor that works on anything?

Three paths, same touch patterns, same classifier:

  PATH A (spike):  pressure → 4 receptor types → spike trains → formatting → tokens → classifier
  PATH B (force):  pressure → temporal binning + spatial pooling → force tokens → classifier
  PATH C (stats):  pressure → per-transducer statistics (mean, std, max, deriv) → classifier

If A > B > C: receptor diversity is the value (spike encoding adds signal)
If A ≈ B:    formatting layer is the value (spikes are decoration)
If C ≈ A:    spike encoding unnecessary (simple force stats suffice)

Also: 6 touch types (added pinch + roll) to test whether 4-class was too easy.

Author: Builder (Q)
Date: 2026-07-30
"""

import numpy as np
import json
import argparse
import sys
from pathlib import Path

# Import from the original prototype
sys.path.insert(0, str(Path(__file__).parent))
from spiketact_prototype import (
    MechanoreceptorEncoder, spike_to_token, feature_vector,
    LinearClassifier, generate_pressure_field, _gaussian_2d,
    POSITIONS, X_GRID, Y_GRID,
    N_TRANSDUCERS, N_RECEPTORS, N_SPIKE_CHANNELS,
    N_SPATIAL_GROUPS, GROUP_SIZE, BIN_SIZE, QUANT_LEVELS,
    RECEPTOR_NAMES, DT_MS
)

# ============================================================
# Extended touch patterns (6 classes: + pinch, + roll)
# ============================================================

PATTERN_TYPES_V3 = ["tap", "press", "slide", "texture", "pinch", "roll"]
DURATION_MS = 1000

def generate_pressure_field_v3(pattern_type, duration_ms, seed=0):
    """Extended pressure field generator with 2 new patterns."""
    if pattern_type in ("tap", "press", "slide", "texture"):
        return generate_pressure_field(pattern_type, duration_ms, seed=seed)

    rng = np.random.RandomState(seed)
    n_steps = int(duration_ms / DT_MS)
    pressure = np.zeros((N_TRANSDUCERS, n_steps))

    if pattern_type == "pinch":
        # Two-point contact: two Gaussian spots, opposite sides, brief simultaneous
        cx1 = rng.uniform(0, 1.0)
        cy1 = rng.uniform(1.0, 3.0)
        cx2 = rng.uniform(2.0, 3.0)
        cy2 = rng.uniform(6.0, 8.0)
        peak1 = rng.uniform(60, 100)
        peak2 = rng.uniform(60, 100)
        sigma1 = rng.uniform(0.6, 1.0)
        sigma2 = rng.uniform(0.6, 1.0)
        onset = int(rng.uniform(50, 150) / DT_MS)
        width = int(rng.uniform(30, 60) / DT_MS)
        for t in range(n_steps):
            if onset <= t < onset + width:
                env = np.sin(np.pi * (t - onset) / width)
                s1 = _gaussian_2d(X_GRID, Y_GRID, cx1, cy1, sigma1)
                s2 = _gaussian_2d(X_GRID, Y_GRID, cx2, cy2, sigma2)
                pressure[:, t] = peak1 * env * s1 + peak2 * env * s2
        pressure += rng.normal(0, 0.5, pressure.shape)

    elif pattern_type == "roll":
        # Rotating pressure: Gaussian spot that moves in a circular path
        peak = rng.uniform(70, 110)
        sigma = rng.uniform(0.8, 1.3)
        cx_center = rng.uniform(1.0, 2.0)
        cy_center = rng.uniform(3.0, 6.0)
        radius = rng.uniform(0.8, 1.5)
        angular_speed = rng.uniform(2.0, 5.0)  # revolutions per second
        onset = int(rng.uniform(50, 150) / DT_MS)
        hold = int(rng.uniform(300, 500) / DT_MS)
        start_angle = rng.uniform(0, 2 * np.pi)
        for t in range(n_steps):
            if onset <= t < onset + hold:
                env = np.sin(np.pi * (t - onset) / hold)
                angle = start_angle + 2 * np.pi * angular_speed * (t - onset) * DT_MS / 1000
                cx = cx_center + radius * np.cos(angle)
                cy = cy_center + radius * np.sin(angle)
                spatial = _gaussian_2d(X_GRID, Y_GRID, cx, cy, sigma)
                pressure[:, t] = peak * env * spatial
        pressure += rng.normal(0, 0.5, pressure.shape)

    else:
        raise ValueError(f"Unknown pattern: {pattern_type}")

    return np.clip(pressure, 0, None)

# ============================================================
# PATH B: Force-to-token (same formatting, raw pressure)
# ============================================================

def force_to_token(pressure, n_groups=N_SPATIAL_GROUPS, bin_size=BIN_SIZE,
                   quant_levels=QUANT_LEVELS):
    """
    Apply the SAME formatting layer (temporal binning + spatial pooling)
    but to raw pressure instead of spike counts.
    This isolates: does the spike encoding add signal, or does the
    formatting layer work equally well on raw force?
    """
    n_trans, n_steps = pressure.shape
    n_bins = n_steps // bin_size

    # Temporal binning: mean pressure per bin
    p_binned = pressure[:, :n_bins * bin_size].reshape(n_trans, n_bins, bin_size)
    p_means = p_binned.mean(axis=2)  # [21, n_bins]

    # Spatial pooling: group transducers
    pooled = np.zeros((n_groups, n_bins))
    for g in range(n_groups):
        t_start = g * GROUP_SIZE
        t_end = t_start + GROUP_SIZE
        pooled[g, :] = p_means[t_start:t_end, :].sum(axis=0)

    # Quantize: scale to [0, quant_levels-1]
    max_val = pooled.max() + 1e-8
    pooled = np.clip((pooled / max_val * quant_levels).astype(int), 0, quant_levels - 1)

    # Tokens: [n_bins, n_groups]
    return pooled.T  # [n_bins, 7]

# ============================================================
# PATH C: Force statistics (traditional approach, like NeoForce)
# ============================================================

def force_stats(pressure):
    """
    Per-transducer statistical features — the traditional approach.
    Simulates what N0-VTLA's "force-based tactile representation" does
    at a basic level: extract statistics from raw force, no spikes,
    no receptor decomposition, no temporal binning.
    """
    n_trans, n_steps = pressure.shape
    features = []

    for t in range(n_trans):
        signal = pressure[t, :]
        features.extend([
            signal.mean(),
            signal.std(),
            signal.max(),
            np.sum(np.abs(np.diff(signal))),  # total variation (proxy for dynamics)
            np.percentile(signal, 75) - np.percentile(signal, 25),  # IQR
            (signal > signal.mean()).sum() / n_steps,  # duty cycle
        ])

    return np.array(features)

# ============================================================
# Unified experiment runner
# ============================================================

def run_comparison(n_samples=40, seed_base=42, n_classes=6, verbose=True):
    """
    Run all three paths on the same data, same classifier, same split.
    """
    patterns = PATTERN_TYPES_V3[:n_classes]

    if verbose:
        print("=" * 64)
        print("SpikeTact v3 — Spike-vs-Force Comparison")
        print("=" * 64)
        print(f"  Touch types: {n_classes} ({', '.join(patterns)})")
        print(f"  Samples per type: {n_samples}")
        print()

    encoder = MechanoreceptorEncoder()

    # Generate data once, reuse for all three paths
    all_pressure = []  # store raw pressure for force paths
    all_spikes = []
    all_labels = []

    for label_idx, pattern in enumerate(patterns):
        if verbose:
            print(f"  Generating {pattern}...", end=" ")
        for i in range(n_samples):
            seed = seed_base + label_idx * 1000 + i
            np.random.seed(seed)
            pressure = generate_pressure_field_v3(pattern, DURATION_MS, seed=seed)
            spikes = encoder.encode(pressure)
            all_pressure.append(pressure)
            all_spikes.append(spikes)
            all_labels.append(label_idx)
        if verbose:
            print("done")

    all_labels = np.array(all_labels)
    n_total = len(all_labels)

    # Same train/test split for all paths
    rng = np.random.RandomState(123)
    perm = rng.permutation(n_total)
    n_train = int(0.6 * n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    results = {}

    # --- PATH A: Spike → Token ---
    if verbose:
        print("\n  PATH A: Spike encoding → formatting layer → tokens")
    X_a = []
    for spikes in all_spikes:
        tokens = spike_to_token(spikes)
        X_a.append(feature_vector(tokens))
    X_a = np.array(X_a)
    acc_a = _train_and_eval(X_a, all_labels, train_idx, test_idx)
    results["spike"] = acc_a
    if verbose:
        print(f"    Accuracy: {acc_a:.1%}")

    # --- PATH B: Force → Token (same formatting) ---
    if verbose:
        print("  PATH B: Raw force → same formatting layer → force tokens")
    X_b = []
    for pressure in all_pressure:
        tokens = force_to_token(pressure)
        X_b.append(feature_vector(tokens))
    X_b = np.array(X_b)
    acc_b = _train_and_eval(X_b, all_labels, train_idx, test_idx)
    results["force_token"] = acc_b
    if verbose:
        print(f"    Accuracy: {acc_b:.1%}")

    # --- PATH C: Force statistics (traditional) ---
    if verbose:
        print("  PATH C: Raw force → per-transducer statistics (NeoForce-style)")
    X_c = []
    for pressure in all_pressure:
        X_c.append(force_stats(pressure))
    X_c = np.array(X_c)
    acc_c = _train_and_eval(X_c, all_labels, train_idx, test_idx)
    results["force_stats"] = acc_c
    if verbose:
        print(f"    Accuracy: {acc_c:.1%}")

    # --- Summary ---
    chance = 1.0 / n_classes
    if verbose:
        print()
        print("=" * 64)
        print("COMPARISON RESULT")
        print("=" * 64)
        print(f"  {'Path':<30s} {'Accuracy':>10s} {'Lift':>8s}")
        print(f"  {'-'*30} {'-'*10} {'-'*8}")
        print(f"  {'A: Spike → Token':<30s} {acc_a:>10.1%} {acc_a/chance:>8.2f}x")
        print(f"  {'B: Force → Token':<30s} {acc_b:>10.1%} {acc_b/chance:>8.2f}x")
        print(f"  {'C: Force → Stats':<30s} {acc_c:>10.1%} {acc_c/chance:>8.2f}x")
        print(f"  {'Chance':<30s} {chance:>10.1%} {1.0:>8.2f}x")
        print()

        # Interpretation
        if acc_a > acc_b + 0.05 and acc_b > acc_c + 0.05:
            print("  VERDICT: Spike encoding ADDS signal beyond formatting.")
            print("           Receptor diversity is the contribution.")
        elif acc_a > acc_b + 0.05 and abs(acc_b - acc_c) < 0.05:
            print("  VERDICT: Spike encoding ADDS signal. Formatting layer is")
            print("           unnecessary for force (raw stats suffice).")
        elif abs(acc_a - acc_b) < 0.05 and acc_b > acc_c + 0.05:
            print("  VERDICT: Formatting layer is the value, not spikes.")
            print("           Spike encoding is decoration.")
        elif abs(acc_a - acc_b) < 0.05 and abs(acc_b - acc_c) < 0.05:
            print("  VERDICT: All paths equivalent. Simple force stats suffice.")
            print("           Spike encoding is unnecessary at this task complexity.")
        else:
            print("  VERDICT: Mixed — see per-path results.")
        print("=" * 64)

    results["chance"] = chance
    results["n_classes"] = n_classes
    results["n_samples"] = n_samples
    results["patterns"] = patterns
    return results


def _train_and_eval(X, y, train_idx, test_idx):
    """Train linear classifier and return test accuracy."""
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    clf = LinearClassifier(X_train_n.shape[1], len(np.unique(y)), lr=0.5, l2=0.01)
    clf.fit(X_train_n, y_train, epochs=1000)
    return clf.score(X_test_n, y_test)


def per_class_confusion(n_samples=40, seed_base=42, n_classes=6, verbose=True):
    """Per-class accuracy + confusion matrix for the spike path."""
    patterns = PATTERN_TYPES_V3[:n_classes]
    encoder = MechanoreceptorEncoder()

    all_features = []
    all_labels = []

    for label_idx, pattern in enumerate(patterns):
        for i in range(n_samples):
            seed = seed_base + label_idx * 1000 + i
            np.random.seed(seed)
            pressure = generate_pressure_field_v3(pattern, DURATION_MS, seed=seed)
            spikes = encoder.encode(pressure)
            tokens = spike_to_token(spikes)
            all_features.append(feature_vector(tokens))
            all_labels.append(label_idx)

    X = np.array(all_features)
    y = np.array(all_labels)
    rng = np.random.RandomState(123)
    perm = rng.permutation(len(y))
    n_train = int(0.6 * len(y))

    X_train, X_test = X[perm[:n_train]], X[perm[n_train:]]
    y_train, y_test = y[perm[:n_train]], y[perm[n_train:]]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    clf = LinearClassifier(X_train_n.shape[1], n_classes, lr=0.5, l2=0.01)
    clf.fit(X_train_n, y_train, epochs=1000)

    test_acc = clf.score(X_test_n, y_test)
    preds = clf.predict(X_test_n)

    if verbose:
        print(f"\n  Spike path test accuracy: {test_acc:.1%}")
        print(f"\n  Per-class accuracy:")
        for i, p in enumerate(patterns):
            mask = y_test == i
            if mask.sum() > 0:
                acc = (preds[mask] == i).mean()
                print(f"    {p:10s}: {acc:.1%} (n={mask.sum()})")

        print(f"\n  Confusion matrix (test set):")
        print(f"    {'':10s} " + " ".join(f"{p:>8s}" for p in patterns))
        for i, p in enumerate(patterns):
            row = np.zeros(n_classes, dtype=int)
            for pred, true in zip(preds, y_test):
                if true == i:
                    row[pred] += 1
            print(f"    {p:10s} " + " ".join(f"{v:8d}" for v in row))

    return test_acc


def main():
    parser = argparse.ArgumentParser(
        description="SpikeTact v3 — spike-vs-force comparison"
    )
    parser.add_argument("--samples", type=int, default=40, help="samples per pattern")
    parser.add_argument("--classes", type=int, default=6, choices=[4, 6],
                        help="number of touch types (4 or 6)")
    parser.add_argument("--seed", type=int, default=42, help="random seed base")
    parser.add_argument("--confusion", action="store_true", help="show per-class confusion")
    parser.add_argument("--save", type=str, default=None, help="save results as JSON")
    args = parser.parse_args()

    result = run_comparison(
        n_samples=args.samples, seed_base=args.seed,
        n_classes=args.classes, verbose=True
    )

    if args.confusion:
        per_class_confusion(
            n_samples=args.samples, seed_base=args.seed,
            n_classes=args.classes, verbose=True
        )

    if args.save:
        Path(args.save).write_text(json.dumps(result, indent=2))
        print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
