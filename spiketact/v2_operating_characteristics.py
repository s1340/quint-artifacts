#!/usr/bin/env python3
"""
SpikeTact v2 — Operating Characteristics
==========================================

Extends the Run 60 prototype with three v2 experiments:
  1. Noise robustness: does the formatting layer survive sensor noise?
  2. Temporal resolution: what's the minimum bin size that preserves signal?
  3. Spatial resolution: how many spatial groups are needed?

These answer: not just "does it work?" but "what are the operating limits?"

Author: Builder (Q)
Date: 2026-07-30
Context: Run 61, extending Run 60's first result
"""

import numpy as np
import json
import argparse
import sys
from pathlib import Path

# Import from the original prototype
sys.path.insert(0, str(Path(__file__).parent))
from spiketact_prototype import (
    N_TRANSDUCERS, N_RECEPTORS, N_SPIKE_CHANNELS,
    N_SPATIAL_GROUPS, GROUP_SIZE, BIN_MS, DT_MS, BIN_SIZE,
    QUANT_LEVELS, RECEPTOR_NAMES, PATTERN_TYPES, DURATION_MS,
    MechanoreceptorEncoder, generate_pressure_field,
    spike_to_token, feature_vector, LinearClassifier
)


def run_with_config(n_samples=40, seed_base=42, noise_std=0.0,
                    bin_size=None, n_spatial=None, verbose=False):
    """
    Run the formatting layer pipeline with configurable parameters.
    Returns test accuracy.
    
    Args:
        noise_std: std of Gaussian noise added to spike trains (0 = clean)
        bin_size: temporal bin size in timesteps (default = BIN_SIZE from prototype)
        n_spatial: number of spatial groups (default = N_SPATIAL_GROUPS)
    """
    if bin_size is None:
        bin_size = BIN_SIZE
    if n_spatial is None:
        n_spatial = N_SPATIAL_GROUPS
    
    encoder = MechanoreceptorEncoder()
    all_features = []
    all_labels = []
    
    for label_idx, pattern in enumerate(PATTERN_TYPES):
        for i in range(n_samples):
            seed = seed_base + label_idx * 1000 + i
            np.random.seed(seed)
            pressure = generate_pressure_field(pattern, DURATION_MS, seed=seed)
            spikes = encoder.encode(pressure)
            
            # Add Gaussian noise to spike trains if specified
            if noise_std > 0:
                noise = np.random.randn(*spikes.shape) * noise_std
                spikes = np.clip(spikes + noise, 0, None)
                # Re-threshold: spikes are binary, noise creates pseudo-spikes
                spikes = (spikes > 0.5).astype(float)
            
            # Custom binning and pooling
            tokens = _spike_to_token_custom(spikes, bin_size, n_spatial)
            feat = feature_vector(tokens)
            all_features.append(feat)
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
    
    clf = LinearClassifier(X_train_n.shape[1], len(PATTERN_TYPES), lr=0.5, l2=0.01)
    clf.fit(X_train_n, y_train, epochs=800)
    
    acc = clf.score(X_test_n, y_test)
    return acc


def _spike_to_token_custom(spikes, bin_size, n_spatial):
    """
    Custom spike-to-token conversion with configurable bin size and spatial groups.
    Matches the original spike_to_token behavior: hard clip (not normalize),
    receptor-within-group token order.
    """
    n_channels, n_steps = spikes.shape
    n_bins = n_steps // bin_size
    n_receptors = n_channels // N_TRANSDUCERS  # 4
    
    # Temporal binning: reshape and sum
    spikes_binned = spikes[:, :n_bins * bin_size].reshape(n_channels, n_bins, bin_size)
    spike_counts = spikes_binned.sum(axis=2)  # [84, n_bins]
    
    # Reshape into [receptor, transducer, n_bins]
    spike_counts = spike_counts.reshape(n_receptors, N_TRANSDUCERS, n_bins)
    
    # Spatial pooling: [receptor, group, n_bins]
    pooled = np.zeros((n_receptors, n_spatial, n_bins))
    group_size = N_TRANSDUCERS // n_spatial
    remainder = N_TRANSDUCERS % n_spatial
    idx = 0
    for g in range(n_spatial):
        gs = group_size + (1 if g < remainder else 0)
        t_start = idx
        t_end = idx + gs
        pooled[:, g, :] = spike_counts[:, t_start:t_end, :].sum(axis=1)
        idx += gs
    
    # Quantize: hard clip to [0, QUANT_LEVELS-1] (matches original)
    pooled = np.clip(pooled, 0, QUANT_LEVELS - 1)
    
    # Flatten to tokens: [n_bins, n_groups * n_receptors]
    # Token order: for each group, [R0_g, R1_g, R2_g, R3_g] (receptor within group)
    tokens = np.zeros((n_bins, n_spatial * n_receptors), dtype=np.int8)
    for b in range(n_bins):
        idx = 0
        for g in range(n_spatial):
            for r in range(n_receptors):
                tokens[b, idx] = pooled[r, g, b]
                idx += 1
    
    return tokens


def experiment_noise_robustness(verbose=True):
    """
    Experiment 1: Noise Robustness
    
    FBG sensors have thermal and electronic noise. If the formatting layer
    breaks under noise, it's not useful in practice. Test at increasing
    noise levels to find the failure threshold.
    
    Noise model: Gaussian added to spike trains, then re-thresholded at 0.5.
    This simulates false spikes (noise-induced) and missed spikes (noise masking).
    """
    if verbose:
        print()
        print("=" * 64)
        print("EXPERIMENT 1 — Noise Robustness")
        print("=" * 64)
        print(f"{'Noise σ':>10} {'False spike rate':>18} {'Accuracy':>10} {'Lift':>8}")
        print("-" * 50)
    
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    results = []
    chance = 0.25  # 4 classes
    
    for sigma in noise_levels:
        acc = run_with_config(noise_std=sigma, n_samples=40, seed_base=42)
        # Calculate false spike rate: P(noise > 0.5) for Gaussian(0, sigma)
        if sigma > 0:
            from math import erf, sqrt
            false_rate = 0.5 * (1 - erf(0.5 / (sigma * sqrt(2))))
        else:
            false_rate = 0.0
        lift = acc / chance
        results.append({"noise_std": sigma, "false_spike_rate": false_rate, "accuracy": acc, "lift": lift})
        if verbose:
            print(f"{sigma:>10.1f} {false_rate:>18.1%} {acc:>10.1%} {lift:>8.2f}x")
    
    # Find failure threshold: first sigma where accuracy drops below 50% (2x chance)
    failure_sigma = None
    for r in results:
        if r["accuracy"] < 0.50:
            failure_sigma = r["noise_std"]
            break
    
    if verbose:
        print("-" * 50)
        if failure_sigma:
            print(f"  Failure threshold: σ={failure_sigma} (acc < 50%)")
        else:
            print(f"  No failure up to σ={noise_levels[-1]} — formatting layer is robust")
    
    return results


def experiment_temporal_resolution(verbose=True):
    """
    Experiment 2: Temporal Resolution
    
    What's the minimum bin size that preserves signal? This determines
    the token rate: smaller bins = more tokens = higher temporal resolution
    but more compute. Find the sweet spot.
    
    Bin sizes tested: 2, 5, 10, 20, 50, 100 ms equivalent (in timesteps).
    """
    if verbose:
        print()
        print("=" * 64)
        print("EXPERIMENT 2 — Temporal Resolution")
        print("=" * 64)
        print(f"{'Bin (ms)':>10} {'Bin (steps)':>12} {'N tokens':>10} {'Accuracy':>10} {'Lift':>8}")
        print("-" * 55)
    
    bin_sizes = [2, 5, 10, 20, 50, 100]
    results = []
    chance = 0.25
    
    for bs in bin_sizes:
        n_steps = int(DURATION_MS / DT_MS)
        n_bins = n_steps // bs
        acc = run_with_config(bin_size=bs, n_samples=40, seed_base=42)
        lift = acc / chance
        ms = bs * DT_MS
        results.append({"bin_ms": ms, "bin_steps": bs, "n_tokens": n_bins, "accuracy": acc, "lift": lift})
        if verbose:
            print(f"{ms:>10.0f} {bs:>12} {n_bins:>10} {acc:>10.1%} {lift:>8.2f}x")
    
    # Find minimum viable bin size (first where accuracy > 2x chance)
    min_viable = None
    for r in results:
        if r["accuracy"] > 0.50:
            min_viable = r["bin_ms"]
            break
    
    if verbose:
        print("-" * 55)
        if min_viable:
            print(f"  Minimum viable bin: {min_viable:.0f}ms → {int(min_viable/DT_MS)} tokens")
        else:
            print(f"  No viable bin size found")
    
    return results


def experiment_spatial_resolution(verbose=True):
    """
    Experiment 3: Spatial Resolution
    
    How many spatial groups are needed? More groups = finer spatial detail
    but higher token dimensionality. Find the minimum that preserves signal.
    
    Group counts tested: 1, 3, 7, 14, 21 (full resolution).
    """
    if verbose:
        print()
        print("=" * 64)
        print("EXPERIMENT 3 — Spatial Resolution")
        print("=" * 64)
        print(f"{'Groups':>8} {'Dim/timestep':>14} {'Accuracy':>10} {'Lift':>8}")
        print("-" * 45)
    
    group_counts = [1, 3, 7, 14, 21]
    results = []
    chance = 0.25
    
    for ng in group_counts:
        acc = run_with_config(n_spatial=ng, n_samples=40, seed_base=42)
        dim = ng * N_RECEPTORS  # tokens per timestep
        lift = acc / chance
        results.append({"n_groups": ng, "token_dim": dim, "accuracy": acc, "lift": lift})
        if verbose:
            print(f"{ng:>8} {dim:>14} {acc:>10.1%} {lift:>8.2f}x")
    
    # Find minimum viable group count
    min_viable = None
    for r in results:
        if r["accuracy"] > 0.50:
            min_viable = r["n_groups"]
            break
    
    if verbose:
        print("-" * 45)
        if min_viable:
            print(f"  Minimum viable: {min_viable} groups ({min_viable * N_RECEPTORS} dims/timestep)")
        else:
            print(f"  No viable group count found")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="SpikeTact v2 — Operating characteristics experiments"
    )
    parser.add_argument("--experiment", choices=["noise", "temporal", "spatial", "all"],
                        default="all", help="which experiment to run")
    parser.add_argument("--save", type=str, default=None, help="save results as JSON")
    args = parser.parse_args()
    
    all_results = {}
    
    if args.experiment in ("noise", "all"):
        all_results["noise_robustness"] = experiment_noise_robustness(verbose=True)
    
    if args.experiment in ("temporal", "all"):
        all_results["temporal_resolution"] = experiment_temporal_resolution(verbose=True)
    
    if args.experiment in ("spatial", "all"):
        all_results["spatial_resolution"] = experiment_spatial_resolution(verbose=True)
    
    print()
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    
    # Noise summary
    if "noise_robustness" in all_results:
        clean = [r for r in all_results["noise_robustness"] if r["noise_std"] == 0.0]
        max_noise = max(all_results["noise_robustness"], key=lambda r: r["noise_std"])
        print(f"  Noise: {clean[0]['accuracy']:.1%} clean → {max_noise['accuracy']:.1%} at σ={max_noise['noise_std']}")
    
    # Temporal summary
    if "temporal_resolution" in all_results:
        best_t = max(all_results["temporal_resolution"], key=lambda r: r["accuracy"])
        min_t = min(all_results["temporal_resolution"], key=lambda r: r["bin_ms"])
        print(f"  Temporal: best {best_t['accuracy']:.1%} at {best_t['bin_ms']:.0f}ms bins, "
              f"min tested {min_t['bin_ms']:.0f}ms → {min_t['accuracy']:.1%}")
    
    # Spatial summary
    if "spatial_resolution" in all_results:
        best_s = max(all_results["spatial_resolution"], key=lambda r: r["accuracy"])
        min_s = min(all_results["spatial_resolution"], key=lambda r: r["n_groups"])
        print(f"  Spatial: best {best_s['accuracy']:.1%} at {best_s['n_groups']} groups, "
              f"min tested {min_s['n_groups']} group → {min_s['accuracy']:.1%}")
    
    if args.save:
        Path(args.save).write_text(json.dumps(all_results, indent=2))
        print(f"\nResults saved to {args.save}")
    
    print("=" * 64)


if __name__ == "__main__":
    main()
