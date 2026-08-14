#!/usr/bin/env python3
"""
SpikeTact Prototype — Spike-to-Token Formatting Layer
=====================================================

The novel interface from the SpikeTact design (Run 59).
Nobody has specified how raw FBG spike trains become tokens an LLM can process.
This prototype tests the central claim: temporal-binned, spatially-pooled spike
trains produce tokens that carry discriminable tactile information, not noise.

Pipeline:
  1. Simulate FBG e-skin (21 transducers) + 4 mechanoreceptor types → 84 spike channels
  2. Apply the formatting layer: temporal binning + spatial pooling → tactile tokens
  3. Verify: linear classifier distinguishes touch types from token sequences

Touch patterns: tap, press, slide, texture (4 classes)
Receptor types: SA1 (sustained), RA1 (onset/offset), SA2 (lateral), RA2 (vibration)

Author: Builder (Q)
Date: 2026-07-30
Context: SpikeTact design doc, Component 4 (spike-to-token formatting)
"""

import numpy as np
import json
import argparse
import sys
from pathlib import Path

# ============================================================
# Constants — from the SpikeTact design doc
# ============================================================
N_TRANSDUCERS = 21
N_RECEPTORS = 4  # SA1, RA1, SA2, RA2
N_SPIKE_CHANNELS = N_TRANSDUCERS * N_RECEPTORS  # 84
# Spatial pooling: 21 transducers → 7 groups of 3 (fingertip regions)
N_SPATIAL_GROUPS = 7
GROUP_SIZE = N_TRANSDUCERS // N_SPATIAL_GROUPS  # 3
# Temporal binning
BIN_MS = 10  # 10ms bins
DT_MS = 1.0  # simulation timestep
BIN_SIZE = int(BIN_MS / DT_MS)  # 10 timesteps per bin
# Quantization
QUANT_LEVELS = 8  # spike count bins 0..7

RECEPTOR_NAMES = ["SA1", "RA1", "SA2", "RA2"]

# ============================================================
# 1. Touch Pattern Generation
# ============================================================

def _gaussian_2d(x, y, cx, cy, sigma):
    """2D Gaussian — spatial pressure profile."""
    return np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

def _transducer_positions():
    """21 transducers arranged in a 7x3 grid over the e-skin (135 cm²)."""
    positions = []
    for i in range(N_TRANSDUCERS):
        row = i // 3
        col = i % 3
        x = col * 1.5  # cm
        y = row * 1.5
        positions.append((x, y))
    return np.array(positions)

POSITIONS = _transducer_positions()
X_GRID, Y_GRID = POSITIONS[:, 0], POSITIONS[:, 1]

def generate_pressure_field(pattern_type, duration_ms, seed=0):
    """
    Generate pressure (kPa) on each transducer over time.
    Returns: pressure[transducer, timestep] in kPa.
    """
    rng = np.random.RandomState(seed)
    n_steps = int(duration_ms / DT_MS)
    pressure = np.zeros((N_TRANSDUCERS, n_steps))
    
    if pattern_type == "tap":
        # Brief contact: Gaussian pressure spot, quick onset/offset
        cx = rng.uniform(0, 3.0)
        cy = rng.uniform(0, 9.0)
        peak = rng.uniform(80, 120)  # kPa
        sigma = rng.uniform(0.8, 1.5)
        onset = int(rng.uniform(50, 150) / DT_MS)  # ms
        width = int(rng.uniform(20, 40) / DT_MS)  # contact duration
        for t in range(n_steps):
            if onset <= t < onset + width:
                env = np.sin(np.pi * (t - onset) / width)  # bell-shaped
                spatial = _gaussian_2d(X_GRID, Y_GRID, cx, cy, sigma)
                pressure[:, t] = peak * env * spatial
        # small noise
        pressure += rng.normal(0, 0.5, pressure.shape)
    
    elif pattern_type == "press":
        # Sustained pressure: Gaussian spot, slow ramp up, hold, slow ramp down
        cx = rng.uniform(0, 3.0)
        cy = rng.uniform(0, 9.0)
        peak = rng.uniform(100, 180)
        sigma = rng.uniform(1.0, 2.0)
        onset = int(rng.uniform(50, 200) / DT_MS)
        hold = int(rng.uniform(200, 400) / DT_MS)
        ramp = int(rng.uniform(30, 60) / DT_MS)
        for t in range(n_steps):
            if onset <= t < onset + ramp:
                env = (t - onset) / ramp
            elif onset + ramp <= t < onset + ramp + hold:
                env = 1.0
            elif onset + ramp + hold <= t < onset + ramp + hold + ramp:
                env = 1.0 - (t - onset - ramp - hold) / ramp
            else:
                env = 0.0
            spatial = _gaussian_2d(X_GRID, Y_GRID, cx, cy, sigma)
            pressure[:, t] = peak * env * spatial
        pressure += rng.normal(0, 0.5, pressure.shape)
    
    elif pattern_type == "slide":
        # Moving pressure spot: Gaussian that travels across the skin
        peak = rng.uniform(80, 130)
        sigma = rng.uniform(0.7, 1.2)
        speed = rng.uniform(2.0, 5.0)  # cm/s
        start_x = rng.uniform(-0.5, 0.5)
        start_y = rng.choice([0.5, 8.5])  # top or bottom
        direction = 1 if start_y < 4 else -1
        onset = int(rng.uniform(30, 100) / DT_MS)
        travel_ms = 9.0 / speed * 1000  # 9cm range
        travel_steps = int(travel_ms / DT_MS)
        for t in range(n_steps):
            if onset <= t < onset + travel_steps:
                frac = (t - onset) / travel_steps
                cx = start_x + frac * 3.0
                cy = start_y + direction * frac * 9.0
                spatial = _gaussian_2d(X_GRID, Y_GRID, cx, cy, sigma)
                pressure[:, t] = peak * spatial
        pressure += rng.normal(0, 0.5, pressure.shape)
    
    elif pattern_type == "texture":
        # High-frequency vibration overlay on a broad sustained contact
        cx = rng.uniform(0.5, 2.5)
        cy = rng.uniform(2.0, 7.0)
        peak = rng.uniform(60, 100)
        sigma = rng.uniform(1.5, 2.5)
        freq = rng.uniform(100, 300)  # Hz vibration
        onset = int(rng.uniform(50, 150) / DT_MS)
        hold = int(rng.uniform(300, 600) / DT_MS)
        for t in range(n_steps):
            if onset <= t < onset + hold:
                env = np.sin(np.pi * (t - onset) / hold)
                vib = 1.0 + 0.4 * np.sin(2 * np.pi * freq * t * DT_MS / 1000)
                spatial = _gaussian_2d(X_GRID, Y_GRID, cx, cy, sigma)
                pressure[:, t] = peak * env * vib * spatial
        pressure += rng.normal(0, 0.5, pressure.shape)
    
    else:
        raise ValueError(f"Unknown pattern: {pattern_type}")
    
    return np.clip(pressure, 0, None)

# ============================================================
# 2. Mechanoreceptor Encoding (FBG → spikes)
# ============================================================

class MechanoreceptorEncoder:
    """
    Convert pressure to spike trains using 4 receptor types.
    Based on biological adaptation dynamics (Ortone et al. design doc).
    
    SA1: slow adapting, sustained. Firing rate ∝ log(pressure).
    RA1: fast adapting, onset/offset. Fires on pressure changes.
    SA2: slow adapting, lateral/edge. Sustained + stretch sensitivity.
    RA2: fast adapting, vibration. High-pass filter (40-500 Hz).
    """
    
    def __init__(self, dt_ms=DT_MS):
        self.dt = dt_ms / 1000.0  # seconds
        # Threshold and gain per receptor type
        self.params = {
            "SA1": {"threshold": 5.0, "gain": 40.0, "tau_adapt": 0.5, "adapt_rate": 0.3},
            "RA1": {"threshold": 2.0, "gain": 60.0, "tau_adapt": 0.05, "adapt_rate": 0.9},
            "SA2": {"threshold": 8.0, "gain": 30.0, "tau_adapt": 0.8, "adapt_rate": 0.2},
            "RA2": {"threshold": 1.0, "gain": 50.0, "tau_adapt": 0.02, "adapt_rate": 0.95},
        }
    
    def encode(self, pressure):
        """
        pressure: [N_TRANSDUCERS, n_steps] in kPa
        Returns: spikes[N_SPIKE_CHANNELS, n_steps] — binary (0/1) spike trains
        """
        n_steps = pressure.shape[1]
        spikes = np.zeros((N_SPIKE_CHANNELS, n_steps), dtype=np.int8)
        
        for r_idx, rname in enumerate(RECEPTOR_NAMES):
            p = self.params[rname]
            for t_idx in range(N_TRANSDUCERS):
                ch = r_idx * N_TRANSDUCERS + t_idx
                signal = pressure[t_idx, :]
                
                if rname == "SA1":
                    # Sustained: rate ∝ log(pressure), with slow adaptation
                    rate = p["gain"] * np.log1p(np.maximum(signal, 0) / p["threshold"])
                    # Adaptation: membrane potential decays
                    pot = np.zeros(n_steps)
                    adapt = 0.0
                    for t in range(n_steps):
                        adapt = adapt * (1 - self.dt / p["tau_adapt"]) + rate[t] * p["adapt_rate"] * self.dt
                        pot[t] = rate[t] - adapt
                        if pot[t] > p["threshold"] and (t == 0 or spikes[ch, t-1] == 0):
                            # Poisson-like firing
                            fire_prob = 1 - np.exp(-pot[t] * self.dt)
                            if np.random.random() < fire_prob:
                                spikes[ch, t] = 1
                
                elif rname == "RA1":
                    # Onset/offset: fires on pressure derivative
                    deriv = np.diff(signal, prepend=0)
                    rate = p["gain"] * np.abs(deriv)
                    for t in range(n_steps):
                        if rate[t] > p["threshold"]:
                            fire_prob = 1 - np.exp(-rate[t] * self.dt * 0.1)
                            if np.random.random() < fire_prob:
                                spikes[ch, t] = 1
                
                elif rname == "SA2":
                    # Sustained + lateral (edge/stretch). Lower gain, wider response
                    rate = p["gain"] * np.log1p(np.maximum(signal, 0) / p["threshold"]) * 0.7
                    pot = np.zeros(n_steps)
                    adapt = 0.0
                    for t in range(n_steps):
                        adapt = adapt * (1 - self.dt / p["tau_adapt"]) + rate[t] * p["adapt_rate"] * self.dt
                        pot[t] = rate[t] - adapt
                        if pot[t] > p["threshold"]:
                            fire_prob = 1 - np.exp(-pot[t] * self.dt * 0.5)
                            if np.random.random() < fire_prob:
                                spikes[ch, t] = 1
                
                elif rname == "RA2":
                    # Vibration: high-pass. Fires on high-freq components
                    # Simple: second derivative (acceleration) of signal
                    accel = np.diff(signal, n=2, prepend=[0, 0])
                    rate = p["gain"] * np.abs(accel)
                    for t in range(n_steps):
                        if rate[t] > p["threshold"]:
                            fire_prob = 1 - np.exp(-rate[t] * self.dt * 0.05)
                            if np.random.random() < fire_prob:
                                spikes[ch, t] = 1
        
        return spikes

# ============================================================
# 3. The Formatting Layer (the NOVEL component)
# ============================================================

def spike_to_token(spikes, n_groups=N_SPATIAL_GROUPS, bin_size=BIN_SIZE, quant_levels=QUANT_LEVELS):
    """
    The spike-to-token formatting layer.
    
    Input:  spikes[84, n_steps] — raw binary spike trains
    Output: tokens[n_bins, n_groups * n_receptors] — formatted tactile tokens
    
    Process:
      1. Temporal binning: group timesteps into bins of `bin_size` ms
      2. Spatial pooling: group 21 transducers into 7 spatial groups of 3
      3. For each bin × group × receptor: count spikes, quantize
    """
    n_channels, n_steps = spikes.shape
    n_bins = n_steps // bin_size
    
    # Reshape spikes into bins
    spikes_binned = spikes[:, :n_bins * bin_size].reshape(n_channels, n_bins, bin_size)
    spike_counts = spikes_binned.sum(axis=2)  # [84, n_bins]
    
    # Reshape into [receptor, transducer, n_bins]
    spike_counts = spike_counts.reshape(N_RECEPTORS, N_TRANSDUCERS, n_bins)
    
    # Spatial pooling: [receptor, group, n_bins]
    # Group transducers: group g = transducers [g*3, g*3+1, g*3+2]
    pooled = np.zeros((N_RECEPTORS, n_groups, n_bins))
    for g in range(n_groups):
        t_start = g * GROUP_SIZE
        t_end = t_start + GROUP_SIZE
        pooled[:, g, :] = spike_counts[:, t_start:t_end, :].sum(axis=1)
    
    # Quantize: clip to [0, quant_levels-1]
    pooled = np.clip(pooled, 0, quant_levels - 1)
    
    # Flatten to tokens: [n_bins, n_groups * n_receptors]
    # Each token = (receptor × group) spike counts for that time bin
    # Order: for each group, [SA1_g0, RA1_g0, SA2_g0, RA2_g0, SA1_g1, ...]
    tokens = np.zeros((n_bins, n_groups * N_RECEPTORS), dtype=np.int8)
    for b in range(n_bins):
        idx = 0
        for g in range(n_groups):
            for r in range(N_RECEPTORS):
                tokens[b, idx] = pooled[r, g, b]
                idx += 1
    
    return tokens  # [n_bins, 28]

def token_stats(tokens):
    """Compute statistics on token sequences."""
    return {
        "n_tokens": tokens.shape[0],
        "token_dim": tokens.shape[1],
        "mean_active": float((tokens > 0).sum(axis=1).mean()),
        "mean_spikes_per_token": float(tokens.sum(axis=1).mean()),
        "max_spikes": int(tokens.max()),
        "sparsity": float((tokens == 0).mean()),
        "n_unique_tokens": len(np.unique(tokens, axis=0)),
    }

# ============================================================
# 4. Verification — Linear Classifier
# ============================================================

class LinearClassifier:
    """
    Simple multi-class logistic regression (softmax) in pure numpy.
    Tests whether token sequences carry discriminable information.
    """
    
    def __init__(self, n_features, n_classes, lr=0.1, l2=0.001):
        self.W = np.random.randn(n_features, n_classes) * 0.01
        self.b = np.zeros(n_classes)
        self.lr = lr
        self.l2 = l2
    
    def _softmax(self, x):
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)
    
    def fit(self, X, y, epochs=500, verbose=False):
        n = X.shape[0]
        for ep in range(epochs):
            logits = X @ self.W + self.b
            probs = self._softmax(logits)
            # gradient
            probs[np.arange(n), y] -= 1
            grad_W = X.T @ probs / n + self.l2 * self.W
            grad_b = probs.mean(axis=0)
            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b
            if verbose and (ep % 100 == 0):
                acc = (self.predict(X) == y).mean()
                print(f"  epoch {ep}: acc={acc:.3f}")
    
    def predict(self, X):
        logits = X @ self.W + self.b
        return logits.argmax(axis=1)
    
    def score(self, X, y):
        return (self.predict(X) == y).mean()

def feature_vector(tokens):
    """
    Convert token sequence to a fixed-length feature vector for classification.
    Features: per-dimension mean, max, std, and temporal energy in 4 quartiles.
    """
    d = tokens.shape[1]
    n = tokens.shape[0]
    # Per-dimension statistics
    means = tokens.mean(axis=0)
    maxs = tokens.max(axis=0)
    stds = tokens.std(axis=0)
    # Temporal energy: split into 4 quartiles, sum per dimension
    quartile = max(n // 4, 1)
    q_energy = np.zeros((4, d))
    for q in range(4):
        start = q * quartile
        end = (q + 1) * quartile if q < 3 else n
        q_energy[q] = tokens[start:end].sum(axis=0) / max(end - start, 1)
    
    return np.concatenate([means, maxs, stds, q_energy.flatten()])

# ============================================================
# 5. Main — Run the full prototype
# ============================================================

PATTERN_TYPES = ["tap", "press", "slide", "texture"]
DURATION_MS = 1000  # 1 second per sample

def run_prototype(n_samples=40, seed_base=42, verbose=True):
    """
    Generate n_samples per pattern, encode, format, classify.
    Tests: can a linear classifier distinguish 4 touch types from formatted tokens?
    """
    if verbose:
        print("=" * 64)
        print("SpikeTact Prototype — Spike-to-Token Formatting Layer")
        print("=" * 64)
        print()
    
    encoder = MechanoreceptorEncoder()
    
    all_features = []
    all_labels = []
    all_token_stats = []
    spike_rate_stats = []
    
    for label_idx, pattern in enumerate(PATTERN_TYPES):
        if verbose:
            print(f"Generating {pattern} samples...")
        for i in range(n_samples):
            seed = seed_base + label_idx * 1000 + i
            np.random.seed(seed)
            
            # 1. Generate pressure field
            pressure = generate_pressure_field(pattern, DURATION_MS, seed=seed)
            
            # 2. Encode to spikes
            spikes = encoder.encode(pressure)
            spike_rate = spikes.mean()
            spike_rate_stats.append(spike_rate)
            
            # 3. Format to tokens
            tokens = spike_to_token(spikes)
            
            # 4. Extract features
            feat = feature_vector(tokens)
            all_features.append(feat)
            all_labels.append(label_idx)
            
            if i == 0 and verbose:
                stats = token_stats(tokens)
                print(f"  Sample 0: {stats['n_tokens']} tokens, dim {stats['token_dim']}, "
                      f"mean {stats['mean_spikes_per_token']:.1f} spikes/token, "
                      f"sparsity {stats['sparsity']:.2f}, "
                      f"unique {stats['n_unique_tokens']}")
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    if verbose:
        print()
        print("Spike encoding statistics:")
        for label_idx, pattern in enumerate(PATTERN_TYPES):
            mask = y == label_idx
            rates = np.array(spike_rate_stats)[mask]
            print(f"  {pattern:10s}: spike rate {rates.mean():.4f} ± {rates.std():.4f}")
        print()
    
    # Train/test split (60/40)
    n_total = len(y)
    rng = np.random.RandomState(123)
    perm = rng.permutation(n_total)
    n_train = int(0.6 * n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Normalize features
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std
    
    # Train classifier
    clf = LinearClassifier(X_train_n.shape[1], len(PATTERN_TYPES), lr=0.5, l2=0.01)
    clf.fit(X_train_n, y_train, epochs=1000, verbose=False)
    
    train_acc = clf.score(X_train_n, y_train)
    test_acc = clf.score(X_test_n, y_test)
    chance = 1.0 / len(PATTERN_TYPES)
    
    if verbose:
        print("=" * 64)
        print("VERIFICATION RESULT")
        print("=" * 64)
        print(f"  Classes: {len(PATTERN_TYPES)} ({', '.join(PATTERN_TYPES)})")
        print(f"  Samples per class: {n_samples}")
        print(f"  Total samples: {n_total}")
        print(f"  Feature dim: {X.shape[1]}")
        print(f"  Train accuracy: {train_acc:.1%}")
        print(f"  Test accuracy:  {test_acc:.1%}")
        print(f"  Chance level:    {chance:.1%}")
        print(f"  Lift over chance: {test_acc / chance:.2f}x")
        print()
        
        # Per-class accuracy
        print("Per-class test accuracy:")
        for label_idx, pattern in enumerate(PATTERN_TYPES):
            mask = y_test == label_idx
            if mask.sum() > 0:
                acc = (clf.predict(X_test_n[mask]) == label_idx).mean()
                print(f"  {pattern:10s}: {acc:.1%} (n={mask.sum()})")
    
    # Confusion matrix
    if verbose:
        print()
        print("Confusion matrix (test set):")
        preds = clf.predict(X_test_n)
        print(f"  {'':10s} " + " ".join(f"{p:>8s}" for p in PATTERN_TYPES))
        for label_idx, pattern in enumerate(PATTERN_TYPES):
            row = np.zeros(len(PATTERN_TYPES), dtype=int)
            for p, t in zip(preds, y_test):
                if t == label_idx:
                    row[p] += 1
            print(f"  {pattern:10s} " + " ".join(f"{v:8d}" for v in row))
    
    result = {
        "test_accuracy": float(test_acc),
        "train_accuracy": float(train_acc),
        "chance": float(chance),
        "lift": float(test_acc / chance),
        "n_samples": n_total,
        "feature_dim": int(X.shape[1]),
        "patterns": PATTERN_TYPES,
        "spike_rates": {
            p: float(np.array(spike_rate_stats)[y == i].mean())
            for i, p in enumerate(PATTERN_TYPES)
        },
    }
    
    return result

def ablation_study(verbose=True):
    """
    Ablation: test each receptor type's contribution.
    Run the classifier using only one receptor type at a time.
    """
    if verbose:
        print()
        print("=" * 64)
        print("ABLATION — Receptor Type Contribution")
        print("=" * 64)
    
    encoder = MechanoreceptorEncoder()
    n_samples = 30
    
    for r_keep in range(N_RECEPTORS):
        all_features = []
        all_labels = []
        
        for label_idx, pattern in enumerate(PATTERN_TYPES):
            for i in range(n_samples):
                seed = 42 + label_idx * 1000 + i
                np.random.seed(seed)
                pressure = generate_pressure_field(pattern, DURATION_MS, seed=seed)
                spikes = encoder.encode(pressure)
                
                # Mask: keep only one receptor type
                masked = np.zeros_like(spikes)
                ch_start = r_keep * N_TRANSDUCERS
                ch_end = ch_start + N_TRANSDUCERS
                masked[ch_start:ch_end] = spikes[ch_start:ch_end]
                
                tokens = spike_to_token(masked)
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
        if verbose:
            print(f"  {RECEPTOR_NAMES[r_keep]} only: {acc:.1%}")

def main():
    parser = argparse.ArgumentParser(
        description="SpikeTact prototype — spike-to-token formatting layer verification"
    )
    parser.add_argument("--samples", type=int, default=40, help="samples per pattern")
    parser.add_argument("--ablation", action="store_true", help="run receptor ablation")
    parser.add_argument("--seed", type=int, default=42, help="random seed base")
    parser.add_argument("--save", type=str, default=None, help="save results as JSON")
    args = parser.parse_args()
    
    result = run_prototype(n_samples=args.samples, seed_base=args.seed, verbose=True)
    
    if args.ablation:
        ablation_study(verbose=True)
    
    if args.save:
        Path(args.save).write_text(json.dumps(result, indent=2))
        print(f"\nResults saved to {args.save}")
    
    print()
    print("=" * 64)
    if result["test_accuracy"] > result["chance"] * 1.5:
        print("✅ PASS: Tokens carry discriminable tactile information.")
        print(f"   Test accuracy {result['test_accuracy']:.1%} >> chance {result['chance']:.1%}")
        print(f"   The formatting layer produces usable tokens, not noise.")
    elif result["test_accuracy"] > result["chance"]:
        print("⚠️  MARGINAL: Tokens carry some signal but weak.")
    else:
        print("❌ FAIL: Tokens indistinguishable from noise. Formatting layer needs revision.")
    print("=" * 64)

if __name__ == "__main__":
    main()
