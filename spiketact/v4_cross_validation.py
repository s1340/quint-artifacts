#!/usr/bin/env python3
"""
spiketact_v4_cv.py — Comprehensive Cross-Validation Upgrade
============================================================

Addresses ALL remaining peer review issues from Claude's second re-review:
1. Stratified k-fold (replaces random k-fold)
2. Larger synthetic dataset (200+ samples/class)
3. L2 sensitivity sweep (1e-4 to 1e2)
4. PCA component sweep (4 to 196)
5. Non-spiking baseline (continuous rate features, no Poisson sampling)

Author: Builder (Q)
Date: 2026-08-02
"""

import numpy as np
import sys
import time
from pathlib import Path

# Import from the prototype
sys.path.insert(0, str(Path(__file__).parent))
from spiketact_prototype import (
    generate_pressure_field, MechanoreceptorEncoder, spike_to_token,
    feature_vector, PATTERN_TYPES, DURATION_MS, N_TRANSDUCERS,
    N_RECEPTORS, N_SPIKE_CHANNELS, N_SPATIAL_GROUPS, GROUP_SIZE,
    BIN_SIZE, QUANT_LEVELS, RECEPTOR_NAMES, DT_MS
)

# Import v3 for 6-class patterns
try:
    from spiketact_v3 import PATTERN_TYPES_V3, generate_pressure_field_v3
    HAS_6CLASS = True
except ImportError:
    HAS_6CLASS = False
    PATTERN_TYPES_V3 = ["tap", "press", "slide", "texture", "pinch", "roll"]


# ============================================================
# Stratified K-Fold (replaces random k-fold)
# ============================================================

def stratified_kfold_indices(y, k=5, seed=42):
    """Stratified k-fold: each fold has approximately equal class representation."""
    rng = np.random.RandomState(seed)
    n = len(y)
    fold_indices = [[] for _ in range(k)]
    
    for label in np.unique(y):
        label_idx = np.where(y == label)[0]
        rng.shuffle(label_idx)
        for i, idx in enumerate(label_idx):
            fold_indices[i % k].append(idx)
    
    return [np.array(sorted(f)) for f in fold_indices]


def stratified_kfold_cv(X, y, k=5, l2=0.01, use_pca=False, n_components=None, seed=42):
    """Run stratified k-fold cross-validation."""
    folds = stratified_kfold_indices(y, k=k, seed=seed)
    n = len(y)
    all_idx = np.arange(n)
    
    accuracies = []
    fold_ns = []
    
    for fold in range(k):
        val_idx = folds[fold]
        train_idx = np.array([i for i in all_idx if i not in set(val_idx)])
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        fold_ns.append(len(y_val))
        
        # Optional PCA (fit on train only)
        if use_pca and n_components:
            mean = X_train.mean(axis=0)
            X_train_c = X_train - mean
            X_val_c = X_val - mean
            U, S, Vt = np.linalg.svd(X_train_c, full_matrices=False)
            X_train = X_train_c @ Vt[:n_components].T
            X_val = X_val_c @ Vt[:n_components].T
        
        # Standardize (fit on train only)
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0) + 1e-8
        X_train = (X_train - mu) / sigma
        X_val = (X_val - mu) / sigma
        
        clf = SoftmaxRegression(X_train.shape[1], len(np.unique(y)), l2=l2)
        clf.fit(X_train, y_train)
        acc = clf.score(X_val, y_val)
        accuracies.append(acc)
    
    return np.array(accuracies), fold_ns


# ============================================================
# Softmax Regression (same as prototype)
# ============================================================

class SoftmaxRegression:
    def __init__(self, n_features, n_classes, lr=0.1, l2=0.01, n_iters=500):
        self.W = np.random.randn(n_features, n_classes) * 0.01
        self.b = np.zeros(n_classes)
        self.lr = lr
        self.l2 = l2
        self.n_iters = n_iters
    
    def _softmax(self, z):
        z = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)
    
    def fit(self, X, y):
        n, d = X.shape
        Y = np.zeros((n, self.W.shape[1]))
        Y[np.arange(n), y] = 1
        for _ in range(self.n_iters):
            probs = self._softmax(X @ self.W + self.b)
            grad = X.T @ (probs - Y) / n + self.l2 * self.W
            self.W -= self.lr * grad
            self.b -= self.lr * (probs - Y).mean(axis=0)
    
    def predict(self, X):
        return (X @ self.W + self.b).argmax(axis=1)
    
    def score(self, X, y):
        return (self.predict(X) == y).mean()


# ============================================================
# Dataset Generation (configurable n_samples)
# ============================================================

def generate_dataset(pattern_types, n_samples=200, seed_base=42, use_v3=False):
    """Generate the full dataset: features X and labels y."""
    encoder = MechanoreceptorEncoder()
    X_all = []
    y_all = []
    
    for label_idx, pattern in enumerate(pattern_types):
        for i in range(n_samples):
            seed = seed_base + label_idx * 10000 + i
            np.random.seed(seed)
            
            if use_v3 and HAS_6CLASS:
                pressure = generate_pressure_field_v3(pattern, DURATION_MS, seed=seed)
            else:
                pressure = generate_pressure_field(pattern, DURATION_MS, seed=seed)
            spikes = encoder.encode(pressure)
            tokens = spike_to_token(spikes)
            feat = feature_vector(tokens)
            
            X_all.append(feat)
            y_all.append(label_idx)
    
    return np.array(X_all), np.array(y_all)


# ============================================================
# Non-Spiking Baseline — Continuous Rate Features
# ============================================================

def encode_rates(pressure):
    """
    Compute continuous rate functions for each receptor type
    WITHOUT Poisson sampling (the non-spiking baseline).
    
    Returns: rates[N_SPIKE_CHANNELS, n_steps] — continuous firing rates
    """
    n_steps = pressure.shape[1]
    rates = np.zeros((N_SPIKE_CHANNELS, n_steps))
    
    params = {
        "SA1": {"threshold": 5.0, "gain": 20.0, "tau_adapt": 2.0, "adapt_rate": 0.05},
        "RA1": {"threshold": 2.0, "gain": 40.0, "tau_adapt": 0.05, "adapt_rate": 0.9},
        "SA2": {"threshold": 8.0, "gain": 30.0, "tau_adapt": 0.8, "adapt_rate": 0.2},
        "RA2": {"threshold": 1.0, "gain": 50.0, "tau_adapt": 0.02, "adapt_rate": 0.95},
    }
    dt = DT_MS
    
    for r_idx, rname in enumerate(RECEPTOR_NAMES):
        p = params[rname]
        for t_idx in range(N_TRANSDUCERS):
            ch = r_idx * N_TRANSDUCERS + t_idx
            signal = pressure[t_idx, :]
            
            if rname == "SA1":
                rate = p["gain"] * np.log1p(np.maximum(signal, 0) / p["threshold"])
                adapt = np.zeros(n_steps)
                pot = np.zeros(n_steps)
                a = 0.0
                for t in range(n_steps):
                    a = a * (1 - dt / p["tau_adapt"]) + rate[t] * p["adapt_rate"] * dt
                    pot[t] = max(0, rate[t] - a)
                rates[ch] = pot
                
            elif rname == "RA1":
                deriv = np.diff(signal, prepend=0)
                rate = p["gain"] * np.abs(deriv)
                rates[ch] = np.maximum(0, rate - p["threshold"])
                
            elif rname == "SA2":
                rate = p["gain"] * np.log1p(np.maximum(signal, 0) / p["threshold"]) * 0.7
                a = 0.0
                pot = np.zeros(n_steps)
                for t in range(n_steps):
                    a = a * (1 - dt / p["tau_adapt"]) + rate[t] * p["adapt_rate"] * dt
                    pot[t] = max(0, rate[t] - a)
                rates[ch] = pot
                
            elif rname == "RA2":
                accel = np.diff(signal, n=2, prepend=[0, 0])
                rate = p["gain"] * np.abs(accel)
                rates[ch] = np.maximum(0, rate - p["threshold"])
    
    return rates


def rate_to_token(rates, n_groups=N_SPATIAL_GROUPS, bin_size=BIN_SIZE):
    """
    Format continuous rate functions through the same temporal binning
    and spatial pooling as the spike path, but WITHOUT quantization
    (rates are continuous, not binary).
    
    Returns: tokens[n_bins, n_groups * n_receptors] — continuous tokens
    """
    n_channels, n_steps = rates.shape
    n_bins = n_steps // bin_size
    
    # Temporal binning: average rate per bin
    rate_binned = rates[:, :n_bins * bin_size].reshape(n_channels, n_bins, bin_size)
    rate_counts = rate_binned.mean(axis=2)  # [84, n_bins] — mean rate per bin
    
    # Reshape into [receptor, transducer, n_bins]
    rate_counts = rate_counts.reshape(N_RECEPTORS, N_TRANSDUCERS, n_bins)
    
    # Spatial pooling: [receptor, group, n_bins]
    pooled = np.zeros((N_RECEPTORS, n_groups, n_bins))
    for g in range(n_groups):
        t_start = g * GROUP_SIZE
        t_end = min(t_start + GROUP_SIZE, N_TRANSDUCERS)
        pooled[:, g, :] = rate_counts[:, t_start:t_end, :].sum(axis=1)
    
    # Flatten: [n_bins, n_groups * n_receptors]
    tokens = pooled.transpose(2, 0, 1).reshape(n_bins, n_groups * N_RECEPTORS)
    return tokens


def generate_nonspiking_dataset(pattern_types, n_samples=200, seed_base=42, use_v3=False):
    """Generate dataset using continuous rate features (non-spiking baseline)."""
    X_all = []
    y_all = []
    
    for label_idx, pattern in enumerate(pattern_types):
        for i in range(n_samples):
            seed = seed_base + label_idx * 10000 + i
            np.random.seed(seed)
            
            if use_v3 and HAS_6CLASS:
                pressure = generate_pressure_field_v3(pattern, DURATION_MS, seed=seed)
            else:
                pressure = generate_pressure_field(pattern, DURATION_MS, seed=seed)
            rates = encode_rates(pressure)
            tokens = rate_to_token(rates)
            feat = feature_vector(tokens)
            
            X_all.append(feat)
            y_all.append(label_idx)
    
    return np.array(X_all), np.array(y_all)


# ============================================================
# L2 Sensitivity Sweep
# ============================================================

def l2_sweep(X, y, l2_values, k=5, seed=42):
    """Sweep L2 regularization strength and report CV accuracy for each."""
    results = []
    for l2 in l2_values:
        accs, _ = stratified_kfold_cv(X, y, k=k, l2=l2, seed=seed)
        results.append((l2, accs.mean(), accs.std()))
    return results


# ============================================================
# PCA Component Sweep
# ============================================================

def pca_sweep(X, y, n_components_list, k=5, l2=0.01, seed=42):
    """Sweep PCA dimensionality and report CV accuracy for each."""
    results = []
    for n_comp in n_components_list:
        if n_comp >= X.shape[1]:
            # No PCA
            accs, _ = stratified_kfold_cv(X, y, k=k, l2=l2, seed=seed)
            results.append((n_comp, accs.mean(), accs.std(), "none"))
        else:
            accs, _ = stratified_kfold_cv(X, y, k=k, l2=l2, use_pca=True, n_components=n_comp, seed=seed)
            results.append((n_comp, accs.mean(), accs.std(), "pca"))
    return results


# ============================================================
# Main — Run All Tests
# ============================================================

def run_full_analysis(pattern_types, label, n_samples=200, use_v3=False):
    """Run the full analysis suite: stratified CV, L2 sweep, PCA sweep, baseline."""
    print(f"\n{'='*70}")
    print(f"  {label} ({len(pattern_types)} classes, {n_samples} samples/class)")
    print(f"{'='*70}")
    
    # Generate spiking dataset
    print(f"  Generating spiking dataset ({n_samples * len(pattern_types)} samples)...")
    t0 = time.time()
    X_spike, y = generate_dataset(pattern_types, n_samples=n_samples, use_v3=use_v3)
    print(f"  Done in {time.time()-t0:.1f}s. Shape: {X_spike.shape}")
    p_n_ratio = X_spike.shape[1] / (0.8 * X_spike.shape[0])
    print(f"  p/n ratio: {X_spike.shape[1]}/{int(0.8*X_spike.shape[0])} = {p_n_ratio:.3f}")
    
    # Generate non-spiking dataset
    print(f"  Generating non-spiking (rate) dataset...")
    t0 = time.time()
    X_rate, _ = generate_nonspiking_dataset(pattern_types, n_samples=n_samples, use_v3=use_v3)
    print(f"  Done in {time.time()-t0:.1f}s. Shape: {X_rate.shape}")
    
    # 1. Stratified 5-fold CV (spiking, original features)
    print(f"\n  1. Stratified 5-fold CV (spike path, L2=0.01):")
    accs, fold_ns = stratified_kfold_cv(X_spike, y, k=5, l2=0.01, seed=42)
    print(f"     Mean: {accs.mean()*100:.1f}% ± {accs.std()*100:.1f}%")
    print(f"     Per-fold: {[f'{a*100:.1f}%' for a in accs]}")
    print(f"     Fold test sizes: {fold_ns}")
    
    # 2. L2 sensitivity sweep
    l2_values = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    print(f"\n  2. L2 Sensitivity Sweep:")
    print(f"     {'L2':>10s} {'Accuracy':>12s} {'Std':>10s}")
    l2_results = l2_sweep(X_spike, y, l2_values, k=5, seed=42)
    for l2, mean, std in l2_results:
        print(f"     {l2:>10.0e} {mean*100:>11.1f}% {std*100:>9.1f}%")
    
    # 3. PCA component sweep
    max_comp = min(X_spike.shape[1], int(0.8 * X_spike.shape[0]))
    pca_values = [4, 8, 16, 32, 64, 96, 128, max_comp]
    pca_values = [c for c in pca_values if c <= max_comp]
    if X_spike.shape[1] not in pca_values:
        pca_values.append(X_spike.shape[1])
    print(f"\n  3. PCA Component Sweep (L2=0.01):")
    print(f"     {'Components':>12s} {'Accuracy':>12s} {'Std':>10s} {'Method':>8s}")
    pca_results = pca_sweep(X_spike, y, pca_values, k=5, l2=0.01, seed=42)
    for n_comp, mean, std, method in pca_results:
        print(f"     {n_comp:>12d} {mean*100:>11.1f}% {std*100:>9.1f}% {method:>8s}")
    
    # 4. Non-spiking baseline (rate features through same formatting)
    print(f"\n  4. Non-spiking baseline (continuous rate features, L2=0.01):")
    accs_rate, _ = stratified_kfold_cv(X_rate, y, k=5, l2=0.01, seed=42)
    print(f"     Mean: {accs_rate.mean()*100:.1f}% ± {accs_rate.std()*100:.1f}%")
    print(f"     Per-fold: {[f'{a*100:.1f}%' for a in accs_rate]}")
    print(f"     Spike vs Rate: {accs.mean()*100:.1f}% vs {accs_rate.mean()*100:.1f}%")
    print(f"     Spike advantage: {(accs.mean() - accs_rate.mean())*100:+.1f}%")
    
    # 5. Multi-seed stability (10 seeds, stratified)
    all_accs = []
    for seed in range(10):
        accs_s, _ = stratified_kfold_cv(X_spike, y, k=5, l2=0.01, seed=seed)
        all_accs.extend(accs_s.tolist())
    all_accs = np.array(all_accs)
    print(f"\n  5. 10-seed stratified stability (50 folds, L2=0.01):")
    print(f"     Mean: {all_accs.mean()*100:.1f}% ± {all_accs.std()*100:.1f}%")
    print(f"     Range: [{all_accs.min()*100:.1f}%, {all_accs.max()*100:.1f}%]")
    
    # Wilson CI on pooled predictions
    n_correct = int(all_accs.mean() * len(y) * 0.2)  # approximate
    n_total = int(len(y) * 0.2)  # ~20% test per fold
    ci_low, ci_high = wilson_ci(n_correct, n_total)
    print(f"     Wilson 95% CI (pooled ~{n_total} predictions): [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
    
    return {
        'spike_cv': (accs.mean(), accs.std()),
        'rate_cv': (accs_rate.mean(), accs_rate.std()),
        'l2_sweep': l2_results,
        'pca_sweep': pca_results,
        'multi_seed': (all_accs.mean(), all_accs.std(), all_accs.min(), all_accs.max()),
    }


def wilson_ci(k, n, z=1.96):
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return 0, 1
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    spread = z * np.sqrt(p * (1-p) / n + z**2 / (4*n**2)) / denom
    return max(0, center - spread), min(1, center + spread)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SpikeTact v4 — Comprehensive CV Upgrade")
    parser.add_argument("--n-samples", type=int, default=200, help="Samples per class (default 200)")
    parser.add_argument("--verify", action="store_true", help="Run verification summary")
    args = parser.parse_args()
    
    print("="*70)
    print("  SpikeTact v4 — Comprehensive Cross-Validation Upgrade")
    print("  Addresses: stratified k-fold, larger dataset, L2 sweep,")
    print("             PCA sweep, non-spiking baseline")
    print("="*70)
    
    n = args.n_samples
    
    # 4-class
    results_4 = run_full_analysis(PATTERN_TYPES, "4-class (original paper)", n_samples=n)
    
    # 6-class
    if HAS_6CLASS:
        results_6 = run_full_analysis(PATTERN_TYPES_V3, "6-class (honest number)", n_samples=n, use_v3=True)
    else:
        print("\n  6-class test skipped (spiketact_v3.py not available)")
        results_6 = None
    
    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY — All Peer Review Issues Addressed")
    print(f"{'='*70}")
    print(f"  1. Stratified k-fold: ✓ (replaces random k-fold)")
    print(f"  2. Dataset size: {n} samples/class ({n*4} total for 4-class, {n*6} for 6-class)")
    print(f"  3. L2 sweep: 7 values from 1e-4 to 1e2")
    print(f"  4. PCA sweep: 8+ component counts")
    print(f"  5. Non-spiking baseline: continuous rate features through same formatting")
    
    print(f"\n  4-class results:")
    print(f"    Spike CV: {results_4['spike_cv'][0]*100:.1f}% ± {results_4['spike_cv'][1]*100:.1f}%")
    print(f"    Rate CV:  {results_4['rate_cv'][0]*100:.1f}% ± {results_4['rate_cv'][1]*100:.1f}%")
    print(f"    Multi-seed: {results_4['multi_seed'][0]*100:.1f}% ± {results_4['multi_seed'][1]*100:.1f}%")
    
    if results_6:
        print(f"\n  6-class results:")
        print(f"    Spike CV: {results_6['spike_cv'][0]*100:.1f}% ± {results_6['spike_cv'][1]*100:.1f}%")
        print(f"    Rate CV:  {results_6['rate_cv'][0]*100:.1f}% ± {results_6['rate_cv'][1]*100:.1f}%")
        print(f"    Multi-seed: {results_6['multi_seed'][0]*100:.1f}% ± {results_6['multi_seed'][1]*100:.1f}%")
    
    print(f"\n  Paper claimed: 96.9% (4-class), 87.5% (6-class)")
    print(f"  These results show whether the numbers survive all fixes.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
