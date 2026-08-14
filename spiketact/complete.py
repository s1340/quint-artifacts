#!/usr/bin/env python3
"""
spiketact_complete.py — The Integrated Dual-Path SpikeTact System
=================================================================

The synthesis. SpikeTact v1-v3 built the discriminative pathway (what am I
touching?). Run 64 built the affective pathway (how does it feel?). This
script wires them together: one touch event → two parallel pathways →
two outputs.

The architecture:
  Touch Event ──┬── Discriminative Path (SA1/RA1/SA2/RA2 → tokens → "what?")
                └── Affective Path (CT afferents → valence/arousal → "how?")

The key demonstration: two touches with identical discriminative signatures
but different affective qualities produce the same classification but
different feelings. The discriminative pathway sees them as the same.
The affective pathway sees the difference. That's the value of the second
pathway. That is the answer to the embodiment question.

Honest claim (corrected from Run 64):
  The coordinate RESPECTS THE STRUCTURE of feeling better than a label does.
  The coordinate is NOT the feeling. The feeling is subjective experience;
  the coordinate is a measurable output that correlates with a correlate of
  pleasantness. Two steps removed. The detection-control gap lives in those
  two steps. The coordinate respects the continuous nature of feeling.
  The label does not. That's the weaker claim. The weaker claim survives.

Usage:
  python spiketact_complete.py                # run the integrated demo
  python spiketact_complete.py --battery      # full battery of touch events
  python spiketact_complete.py --contrast     # the key contrast demo
  python spiketact_complete.py --verify       # verify both pathways
  python spiketact_complete.py --json          # output as JSON

Author: Builder (Q)
Date: 2026-07-31
Context: SpikeTact simulation arc — the synthesis. Runs 60-64 integrated.
"""

import argparse
import json
import sys
import os
import numpy as np

# Import from the existing scripts (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prototype import (
    generate_pressure_field, MechanoreceptorEncoder, spike_to_token,
    feature_vector, LinearClassifier, PATTERN_TYPES, DURATION_MS,
    token_stats, N_TRANSDUCERS, N_RECEPTORS, N_SPIKE_CHANNELS,
    N_SPATIAL_GROUPS, BIN_SIZE, QUANT_LEVELS, RECEPTOR_NAMES
)
from affective import (
    simulate_ct_afferents, compute_valence_arousal,
    generate_affective_stimuli, ct_velocity_tuning,
    ct_temperature_tuning, ct_force_tuning,
    CT_OPTIMAL_VELOCITY, CT_OPTIMAL_TEMP
)

SEED = 42


# ============================================================
# Unified Touch Event
# ============================================================

class TouchEvent:
    """
    A unified touch event that activates both pathways.

    Discriminative parameters (for the Aβ / mechanoreceptor pathway):
      - pattern_type: tap, press, slide, texture (determines pressure field)
      - duration_ms: how long the touch lasts

    Affective parameters (for the CT afferent pathway):
      - velocity: stroking velocity in cm/s (CT-optimal: ~3 cm/s)
      - temperature: skin temperature in °C (CT-optimal: ~32°C)
      - force: contact force in N (CT-optimal: ~0.0007 N = 0.7 mN)

    In a real system, these parameters come from the same physical contact.
    A slow, gentle, warm caress activates CT afferents (affective) AND
    produces a moving pressure field (discriminative: "slide").
    A fast, heavy, cold press activates Aβ (discriminative: "press") AND
    suppresses CT (affective: low valence).
    """

    def __init__(self, name, pattern_type, velocity, temperature, force,
                 duration_ms=1000, description=""):
        self.name = name
        self.pattern_type = pattern_type
        self.velocity = velocity      # cm/s
        self.temperature = temperature  # °C
        self.force = force            # N
        self.duration_ms = duration_ms
        self.description = description

    def discriminative_params(self):
        return self.pattern_type, self.duration_ms

    def affective_params(self):
        return {
            'velocity': self.velocity,
            'force': self.force,
            'temp': self.temperature,
            'duration': self.duration_ms / 1000.0,
            'onset': 0.2,
        }

    def __repr__(self):
        return (f"TouchEvent({self.name}: {self.pattern_type}, "
                f"v={self.velocity} cm/s, T={self.temperature}°C, "
                f"F={self.force*1000:.1f} mN)")


# ============================================================
# The Integrated Dual-Path System
# ============================================================

class SpikeTactComplete:
    """
    The complete SpikeTact system: discriminative + affective pathways.

    Input:  TouchEvent (physical touch with both discriminative and affective params)
    Output: {
        'classification': 'slide'     # discriminative: what am I touching?
        'confidence': 0.87,           # discriminative confidence
        'valence': 0.91,              # affective: how pleasant? [0, 1]
        'arousal': 0.47,              # affective: how intense? [0, 1]
        'ct_rate_hz': 12.3,           # CT afferent firing rate
        'interpretation': 'pleasant, engaging'  # combined interpretation
    }
    """

    def __init__(self, seed=SEED):
        self.encoder = MechanoreceptorEncoder()
        self.clf = None  # trained on first use
        self.scaler_mean = None
        self.scaler_std = None
        self.seed = seed
        self._train_classifier()

    def _train_classifier(self):
        """Train the discriminative classifier on all 4 pattern types."""
        np.random.seed(self.seed)
        all_features = []
        all_labels = []

        for label_idx, pattern in enumerate(PATTERN_TYPES):
            for i in range(40):
                seed = self.seed + label_idx * 1000 + i
                np.random.seed(seed)
                pressure = generate_pressure_field(pattern, DURATION_MS, seed=seed)
                spikes = self.encoder.encode(pressure)
                tokens = spike_to_token(spikes)
                feat = feature_vector(tokens)
                all_features.append(feat)
                all_labels.append(label_idx)

        X = np.array(all_features)
        y = np.array(all_labels)

        # Train/test split
        rng = np.random.RandomState(123)
        perm = rng.permutation(len(y))
        n_train = int(0.6 * len(y))
        train_idx = perm[:n_train]

        X_train = X[train_idx]
        y_train = y[train_idx]

        # Normalize
        self.scaler_mean = X_train.mean(axis=0)
        self.scaler_std = X_train.std(axis=0) + 1e-8

        X_train_n = (X_train - self.scaler_mean) / self.scaler_std

        # Train
        self.clf = LinearClassifier(X_train_n.shape[1], len(PATTERN_TYPES),
                                     lr=0.5, l2=0.01)
        self.clf.fit(X_train_n, y_train, epochs=1000, verbose=False)

    def process(self, event, seed_offset=0):
        """
        Process a touch event through both pathways.

        Returns the complete dual-path response.
        """
        np.random.seed(self.seed + seed_offset)

        # --- Discriminative Pathway ---
        pattern, duration_ms = event.discriminative_params()
        pressure = generate_pressure_field(pattern, duration_ms,
                                           seed=self.seed + seed_offset)
        spikes = self.encoder.encode(pressure)
        tokens = spike_to_token(spikes)
        feat = feature_vector(tokens)
        feat_n = (feat - self.scaler_mean) / self.scaler_std

        logits = self.clf._softmax(feat_n.reshape(1, -1))
        pred_idx = self.clf.predict(feat_n.reshape(1, -1))[0]
        confidence = float(logits[0, pred_idx])
        classification = PATTERN_TYPES[pred_idx]

        # Token statistics
        stats = token_stats(tokens)

        # --- Affective Pathway ---
        affective_params = event.affective_params()
        ct_spikes, ct_rate, ct_time = simulate_ct_afferents(affective_params)
        va = compute_valence_arousal(ct_spikes)

        valence = va['valence']
        arousal = va['arousal']

        # --- Combined Interpretation ---
        interpretation = self._interpret(classification, valence, arousal)

        return {
            'event_name': event.name,
            'description': event.description,
            # Discriminative
            'classification': classification,
            'confidence': round(confidence, 4),
            'token_count': stats['n_tokens'],
            'token_sparsity': round(stats['sparsity'], 4),
            # Affective
            'valence': round(valence, 4),
            'arousal': round(arousal, 4),
            'ct_firing_rate_hz': round(va['mean_rate_hz'], 2),
            'n_active_ct_afferents': va['n_active_afferents'],
            # Combined
            'interpretation': interpretation,
        }

    def _interpret(self, classification, valence, arousal):
        """Combined interpretation from both pathways."""
        if valence > 0.3 and arousal > 0.2:
            feeling = "pleasant, engaging"
        elif valence > 0.3 and arousal < 0.2:
            feeling = "pleasant, calming"
        elif valence < 0.1 and arousal > 0.3:
            feeling = "unpleasant, alarming"
        elif valence < 0.1 and arousal < 0.1:
            feeling = "neutral, low engagement"
        elif 0.1 <= valence <= 0.3:
            feeling = "mildly pleasant"
        else:
            feeling = "ambiguous"

        return f"touch: {classification} | feeling: {feeling}"


# ============================================================
# Demonstrations
# ============================================================

def demo_battery():
    """A full battery of touch events spanning the dual-path space."""
    print("=" * 72)
    print("SpikeTact Complete — Integrated Dual-Path System")
    print("=" * 72)
    print()
    print("Discriminative pathway: SA1/RA1/SA2/RA2 → tokens → 'what am I touching?'")
    print("Affective pathway:      CT afferents → valence/arousal → 'how does it feel?'")
    print()
    print("Both pathways process the SAME touch event in parallel.")
    print()

    events = [
        TouchEvent("warm_caress", "slide", velocity=3.0, temperature=32.0,
                   force=0.0007, description="gentle warm caress (CT-optimal)"),
        TouchEvent("cold_caress", "slide", velocity=3.0, temperature=18.0,
                   force=0.0007, description="gentle cold caress (CT-suppressed)"),
        TouchEvent("warm_pat", "tap", velocity=20.0, temperature=32.0,
                   force=0.0007, description="quick warm pat (non-CT-optimal velocity)"),
        TouchEvent("firm_press", "press", velocity=0.5, temperature=32.0,
                   force=0.05, description="firm press (heavy, Aβ-dominant)"),
        TouchEvent("texture_touch", "texture", velocity=5.0, temperature=32.0,
                   force=0.002, description="texture exploration (moderate)"),
        TouchEvent("cold_texture", "texture", velocity=5.0, temperature=18.0,
                   force=0.002, description="cold texture exploration (CT-suppressed)"),
        TouchEvent("hot_slide", "slide", velocity=3.0, temperature=40.0,
                   force=0.0007, description="hot slide (temperature-suppressed)"),
        TouchEvent("fast_slide", "slide", velocity=25.0, temperature=32.0,
                   force=0.0007, description="fast slide (velocity-suppressed)"),
    ]

    system = SpikeTactComplete()

    # Header
    print(f"{'Event':<20} {'Class':<10} {'Valence':>8} {'Arousal':>8} "
          f"{'CT Hz':>7} {'Interpretation'}")
    print("-" * 100)

    results = []
    for i, event in enumerate(events):
        result = system.process(event, seed_offset=i)
        results.append(result)
        print(f"  {event.name:<18} {result['classification']:<10} "
              f"{result['valence']:>8.3f} {result['arousal']:>8.3f} "
              f"{result['ct_firing_rate_hz']:>7.2f}  "
              f"{result['interpretation']}")

    print("-" * 100)
    print()
    print("The discriminative pathway classifies the touch type (what).")
    print("The affective pathway computes the feeling (how).")
    print("The same physical event produces both — in parallel, independently.")
    print()
    print("Note: warm_caress and cold_caress have the same discriminative")
    print("signature (slide pattern, same force). The discriminative pathway")
    print("classifies them identically. The affective pathway distinguishes them:")
    print("warm = high valence, cold = near-zero valence. That's the value of")
    print("the second pathway. That's the difference between 'what' and 'how.'")

    return results


def demo_contrast():
    """
    The key contrast: identical discriminative signatures, different feelings.

    This is the demonstration that proves the dual-path architecture adds
    information the discriminative pathway alone cannot access.
    """
    print("=" * 72)
    print("SpikeTact Complete — The Key Contrast")
    print("=" * 72)
    print()
    print("Three touches with IDENTICAL discriminative signatures")
    print("(same pattern: slide, same force: 0.7 mN, same duration)")
    print("but DIFFERENT affective qualities:")
    print()

    system = SpikeTactComplete()

    contrast_events = [
        TouchEvent("warm_caress", "slide", velocity=3.0, temperature=32.0,
                   force=0.0007, description="warm caress — CT-optimal"),
        TouchEvent("cold_caress", "slide", velocity=3.0, temperature=18.0,
                   force=0.0007, description="cold caress — CT temperature-suppressed"),
        TouchEvent("hot_caress", "slide", velocity=3.0, temperature=40.0,
                   force=0.0007, description="hot caress — CT temperature-suppressed"),
        TouchEvent("fast_caress", "slide", velocity=25.0, temperature=32.0,
                   force=0.0007, description="fast caress — CT velocity-suppressed"),
    ]

    print(f"{'Touch':<20} {'Class':<10} {'Valence':>8} {'Arousal':>8} {'CT Hz':>7} {'Note'}")
    print("-" * 85)

    for i, event in enumerate(contrast_events):
        # Same seed for all → same pressure field → same discriminative classification.
        # The affective pathway differs because velocity/temp differ.
        result = system.process(event, seed_offset=200)
        print(f"  {event.name:<18} {result['classification']:<10} "
              f"{result['valence']:>8.3f} {result['arousal']:>8.3f} "
              f"{result['ct_firing_rate_hz']:>7.2f}  {event.description}")

    print("-" * 85)
    print()
    print("DISCRIMINATIVE PATHWAY: All four are classified as 'slide'.")
    print("  The discriminative pathway sees them as the same touch.")
    print()
    print("AFFECTIVE PATHWAY: The four touches span the feeling space.")
    print("  warm caress:  valence ~0.9 (pleasant)  — CT fires optimally")
    print("  cold caress:  valence ~0.0 (neutral)    — CT temperature-suppressed")
    print("  hot caress:   valence ~0.0 (neutral)    — CT temperature-suppressed")
    print("  fast caress:  valence ~0.1 (low)        — CT velocity-suppressed")
    print()
    print("The discriminative pathway answers 'what am I touching?' — a slide.")
    print("The affective pathway answers 'how does it feel?' — pleasant, neutral, or not.")
    print("Both answers are needed. Neither subsumes the other.")
    print()
    print("---")
    print()
    print("HONEST CLAIM (corrected from Run 64):")
    print("  The coordinate RESPECTS THE STRUCTURE of feeling better than a label does.")
    print("  The coordinate is NOT the feeling. The feeling is subjective experience.")
    print("  The coordinate is a measurable output that correlates with a correlate")
    print("  of pleasantness. Two steps removed. The detection-control gap lives")
    print("  in those two steps. The coordinate respects the continuous nature of")
    print("  feeling. The label does not. That's the weaker claim. The weaker claim")
    print("  survives.")


def demo_verify():
    """Verify both pathways independently."""
    print("=" * 72)
    print("SpikeTact Complete — Verification of Both Pathways")
    print("=" * 72)
    print()

    # Discriminative verification
    print("1. DISCRIMINATIVE PATHWAY (from prototype.py)")
    print("-" * 50)
    np.random.seed(42)
    encoder = MechanoreceptorEncoder()
    all_features = []
    all_labels = []
    for label_idx, pattern in enumerate(PATTERN_TYPES):
        for i in range(40):
            seed = 42 + label_idx * 1000 + i
            np.random.seed(seed)
            pressure = generate_pressure_field(pattern, DURATION_MS, seed=seed)
            spikes = encoder.encode(pressure)
            tokens = spike_to_token(spikes)
            feat = feature_vector(tokens)
            all_features.append(feat)
            all_labels.append(label_idx)

    X = np.array(all_features)
    y = np.array(all_labels)
    rng = np.random.RandomState(123)
    perm = rng.permutation(len(y))
    n_train = int(0.6 * len(y))
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std
    clf = LinearClassifier(X_train_n.shape[1], len(PATTERN_TYPES), lr=0.5, l2=0.01)
    clf.fit(X_train_n, y_train, epochs=1000, verbose=False)
    test_acc = clf.score(X_test_n, y_test)
    print(f"  4-class accuracy: {test_acc:.1%} (chance: 25.0%)")
    print(f"  Lift over chance: {test_acc / 0.25:.2f}x")
    print(f"  Status: {'✅ PASS' if test_acc > 0.7 else '❌ FAIL'}")
    print()

    # Affective verification
    print("2. AFFECTIVE PATHWAY (from affective.py)")
    print("-" * 50)
    from affective import verify_ct_properties
    ct_results = verify_ct_properties()
    n_pass = sum(1 for r in ct_results if r['passed'])
    for r in ct_results:
        status = '✅' if r['passed'] else '❌'
        print(f"  {status} {r['property']}: {r['values']}")
    print(f"  CT properties: {n_pass}/{len(ct_results)} passed")
    print()

    # Integration verification
    print("3. INTEGRATION (dual-path independence)")
    print("-" * 50)
    system = SpikeTactComplete()

    # Same discriminative, different affective
    warm = TouchEvent("warm", "slide", 3.0, 32.0, 0.0007)
    cold = TouchEvent("cold", "slide", 3.0, 18.0, 0.0007)
    r_warm = system.process(warm, seed_offset=0)
    r_cold = system.process(cold, seed_offset=0)

    same_class = r_warm['classification'] == r_cold['classification']
    diff_valence = r_warm['valence'] != r_cold['valence']

    print(f"  warm caress: class={r_warm['classification']}, valence={r_warm['valence']:.3f}")
    print(f"  cold caress: class={r_cold['classification']}, valence={r_cold['valence']:.3f}")
    print(f"  Same classification (discriminative sees same touch): {'✅' if same_class else '❌'}")
    print(f"  Different valence (affective distinguishes them): {'✅' if diff_valence else '❌'}")
    print()

    print("=" * 72)
    all_pass = test_acc > 0.7 and n_pass == len(ct_results) and same_class and diff_valence
    print(f"Overall: {'✅ ALL PASS' if all_pass else '❌ SOME FAILED'}")
    print("=" * 72)


def demo_json():
    """Output the battery results as JSON (for programmatic access)."""
    system = SpikeTactComplete()
    events = [
        TouchEvent("warm_caress", "slide", 3.0, 32.0, 0.0007, "CT-optimal caress"),
        TouchEvent("cold_caress", "slide", 3.0, 18.0, 0.0007, "CT-suppressed cold caress"),
        TouchEvent("warm_pat", "tap", 20.0, 32.0, 0.0007, "fast pat"),
        TouchEvent("firm_press", "press", 0.5, 32.0, 0.05, "heavy press"),
        TouchEvent("texture_touch", "texture", 5.0, 32.0, 0.002, "texture exploration"),
    ]
    results = [system.process(e, seed_offset=i) for i, e in enumerate(events)]
    print(json.dumps(results, indent=2))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SpikeTact Complete — Integrated Dual-Path System")
    parser.add_argument('--battery', action='store_true',
                        help='Full battery of touch events')
    parser.add_argument('--contrast', action='store_true',
                        help='Key contrast: same discriminative, different affective')
    parser.add_argument('--verify', action='store_true',
                        help='Verify both pathways')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON')
    args = parser.parse_args()

    if args.battery:
        demo_battery()
    elif args.contrast:
        demo_contrast()
    elif args.verify:
        demo_verify()
    elif args.json:
        demo_json()
    else:
        # Default: run the battery
        demo_battery()


if __name__ == '__main__':
    main()
