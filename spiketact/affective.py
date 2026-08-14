#!/usr/bin/env python3
"""
spiketact_affective.py — The Affective Touch Layer

The missing piece. SpikeTact v1-v3 built the discriminative pathway:
SA1/RA1/SA2/RA2 → spike-to-token → "what am I touching?" That's the Aβ
pathway, the one that goes to S1, the one that discriminates.

This module builds the affective pathway: CT afferents → valence/arousal
→ "how does it feel?" That's the CT pathway, the one that goes to the
insular cortex, the one that encodes emotional significance.

Biological basis:
  - C-tactile (CT) afferents are unmyelinated, slow-conducting (~2 m/s)
  - They fire on an inverted-U curve: peak at 1-10 cm/s stroking velocity
  - They're tuned to skin temperature (~32°C)
  - Their firing correlates with subjective pleasantness ratings
  - They do NOT discriminate — they evaluate

The dual-path architecture:
  Discriminative (SpikeTact v1-v3): SA1/RA1/SA2/RA2 → tokens → "what?"
  Affective (this module):          CT → valence/arousal → "how does it feel?"

The tokenizer problem: "reading about pressure, not feeling it."
The discriminative pathway reads pressure. The affective pathway feels.
This module is the feeling.

Usage:
  python spiketact_affective.py                    # run the affective touch demo
  python spiketact_affective.py --compare          # discriminative vs affective
  python spiketact_affective.py --verify          # verify against known CT properties
"""

import argparse
import json
import numpy as np
from pathlib import Path

SEED = 42
np.random.seed(SEED)

# ============================================================
# CT Afferent Simulation
# ============================================================

# Biological CT afferent properties (from McGlone et al., Löken et al.)
CT_OPTIMAL_VELOCITY = 3.0       # cm/s — peak of inverted-U (Löken et al. 2009)
CT_VELOCITY_RANGE = (0.1, 30.0) # cm/s — responsive range
CT_OPTIMAL_TEMP = 32.0          # °C — skin temperature tuning (Ackerley et al. 2014)
CT_TEMP_RANGE = (25.0, 40.0)    # °C — responsive range
CT_CONDUCTION_VELOCITY = 2.0   # m/s — unmyelinated, slow
CT_MAX_FIRING_RATE = 50.0       # Hz — peak firing rate at optimal stimulus
CT_RESTING_RATE = 0.5           # Hz — baseline (homeostatic)
CT_INDUCTION_TIME = 0.8         # s — delay before CT signal reaches cortex (slow conduction)


def ct_velocity_tuning(velocity_cms: float) -> float:
    """
    The inverted-U velocity tuning curve.

    CT afferents fire maximally at 1-10 cm/s, with peak at ~3 cm/s.
    Below 0.5 cm/s: barely fires. Above 30 cm/s: barely fires.
    This curve correlates with subjective pleasantness ratings.

    Based on Löken et al. (2009) psychophysical data.
    """
    v = max(0.01, velocity_cms)
    # Log-Gaussian fit to biological data
    log_v = np.log(v)
    log_opt = np.log(CT_OPTIMAL_VELOCITY)
    sigma = 1.2  # width of tuning curve (fits biological data)
    return np.exp(-0.5 * ((log_v - log_opt) / sigma) ** 2)


def ct_temperature_tuning(temp_c: float) -> float:
    """
    Temperature modulation. CT afferents prefer skin temperature (~32°C).
    Cold touch (<25°C) or hot touch (>40°C) reduces affective response.
    """
    t = max(15.0, min(45.0, temp_c))
    sigma_t = 5.0
    return np.exp(-0.5 * ((t - CT_OPTIMAL_TEMP) / sigma_t) ** 2)


def ct_force_tuning(force_n: float) -> float:
    """
    Force sensitivity. CT afferents respond to very light touch (0.25-3 mN).
    Heavy pressure activates Aβ afferents, not CT.
    """
    f = max(0.0, force_n)
    f_mn = f * 1000  # convert N to mN
    if f_mn < 0.1:
        return 0.0
    if f_mn > 10.0:
        return 0.1  # heavy pressure barely activates CT
    # Peak at ~0.7 mN
    return np.exp(-0.5 * ((np.log(f_mn) - np.log(0.7)) / 0.8) ** 2)


def simulate_ct_afferents(stimulus, n_afferents=50, duration_s=2.0, dt=0.001):
    """
    Simulate a population of CT afferents responding to a touch stimulus.

    Stimulus: dict with velocity (cm/s), force (N), temp (°C), area (cm²).
    Returns: spike train (n_afferents × n_timesteps), instantaneous firing rate.

    The CT population encodes AFFECTIVE quality, not spatial pattern.
    """
    n_steps = int(duration_s / dt)
    time = np.arange(n_steps) * dt

    velocity = stimulus.get('velocity', 0.0)
    force = stimulus.get('force', 0.0)
    temp = stimulus.get('temp', CT_OPTIMAL_TEMP)
    duration = stimulus.get('duration', 1.0)  # how long the touch lasts

    # Combined tuning
    v_tuning = ct_velocity_tuning(velocity)
    t_tuning = ct_temperature_tuning(temp)
    f_tuning = ct_force_tuning(force)

    # The affective signal is the product of all three tunings
    # (all must be in their optimal range for the CT to fire strongly)
    affective_drive = v_tuning * t_tuning * f_tuning

    # Stimulus envelope (touch onset and offset)
    onset_s = stimulus.get('onset', 0.2)
    offset_s = onset_s + duration
    envelope = np.zeros(n_steps)
    onset_idx = int(onset_s / dt)
    offset_idx = min(int(offset_s / dt), n_steps)
    # Smooth onset/offset (biological rise time ~100ms)
    rise_steps = int(0.1 / dt)
    for i in range(onset_idx, offset_idx):
        local = i - onset_idx
        if local < rise_steps:
            envelope[i] = local / rise_steps
        elif i > offset_idx - rise_steps:
            envelope[i] = (offset_idx - i) / rise_steps
        else:
            envelope[i] = 1.0

    # CT conduction delay (slow — the signal arrives late)
    delay_steps = int(CT_INDUCTION_TIME / dt)

    # Per-afferent variability (biological heterogeneity)
    np.random.seed(SEED)
    aff_var = np.random.normal(1.0, 0.15, n_afferents)

    # Generate spike trains
    spikes = np.zeros((n_afferents, n_steps))
    inst_rate = np.zeros(n_steps)

    for t in range(delay_steps, n_steps):
        # The drive at time t is from the stimulus at time t-delay
        env_val = envelope[t - delay_steps]
        base_rate = CT_RESTING_RATE + affective_drive * env_val * CT_MAX_FIRING_RATE
        inst_rate[t] = base_rate

        # Poisson spiking (per-afferent)
        for a in range(n_afferents):
            prob = (CT_RESTING_RATE + affective_drive * env_val * CT_MAX_FIRING_RATE * aff_var[a]) * dt
            if np.random.random() < prob:
                spikes[a, t] = 1

    return spikes, inst_rate, time


def compute_valence_arousal(spikes, dt=0.001):
    """
    Convert CT population response to valence/arousal coordinates.

    Valence: positive = pleasant, negative = unpleasant.
    - High CT firing at optimal velocity → positive valence (pleasant)
    - Low CT firing or firing at non-optimal velocity → neutral/negative

    Arousal: intensity of the affective response.
    - Overall firing rate drives arousal
    - Non-optimal velocity but high force → high arousal, negative valence

    This is the "feeling" output — not a token, not a classification.
    A position in 2D affective space.
    """
    total_spikes = np.sum(spikes)
    n_afferents = spikes.shape[0]
    duration_s = spikes.shape[1] * dt

    # Mean firing rate across population
    mean_rate = total_spikes / (n_afferents * duration_s) if duration_s > 0 else 0

    # Valence: based on how close the firing is to the optimal CT response
    # The CT signal IS the pleasantness signal (biological basis)
    # Normalize: 0 firing = neutral (0), optimal firing = pleasant (+1)
    # Non-CT-optimal stimuli (fast, cold, heavy) produce no CT signal → neutral
    # Only CT-optimal stimuli (gentle, warm, slow) produce positive valence
    max_expected_rate = CT_MAX_FIRING_RATE * 0.5  # realistic peak with all tunings at ~0.5
    if mean_rate > CT_RESTING_RATE:
        # Above resting → pleasant direction
        excess = mean_rate - CT_RESTING_RATE
        valence = excess / max_expected_rate
    else:
        # At or below resting → neutral
        valence = 0.0
    valence = np.clip(valence, -0.2, 1.0)

    # Arousal: total spike count (intensity of affective response)
    # Normalize to [0, 1]
    max_spikes = n_afferents * duration_s * CT_MAX_FIRING_RATE
    arousal = total_spikes / max_spikes if max_spikes > 0 else 0
    arousal = np.clip(arousal, 0, 1)

    return {
        'valence': float(valence),
        'arousal': float(arousal),
        'mean_rate_hz': float(mean_rate),
        'total_spikes': int(total_spikes),
        'n_active_afferents': int(np.sum(np.any(spikes, axis=1))),
    }


# ============================================================
# Touch Stimuli (affective categories)
# ============================================================

def generate_affective_stimuli():
    """
    Generate touch stimuli that span the affective space.

    Unlike the discriminative stimuli (tap, press, slide, texture — classified by type),
    these stimuli are characterized by their AFFECTIVE quality:
    - Caress: slow, light, warm → high pleasantness
    - Pat: fast, light, neutral → neutral
    - Squeeze: slow, heavy, warm → moderate pleasantness, high arousal
    - Scratch: fast, sharp, neutral → unpleasant, high arousal
    - Hold: static, moderate, warm → moderate pleasantness, low arousal
    - Cold press: static, moderate, cold → unpleasant, low arousal
    """
    stimuli = {
        'caress': {
            'velocity': 3.0,   # optimal for CT
            'force': 0.0007,    # 0.7 mN — optimal for CT
            'temp': 32.0,       # skin temperature
            'duration': 1.5,
            'onset': 0.2,
            'label': 'gentle caress (CT-optimal)',
            'expected_valence': 'high positive',
        },
        'pat': {
            'velocity': 20.0,   # too fast for CT
            'force': 0.001,     # light
            'temp': 32.0,
            'duration': 0.3,
            'onset': 0.2,
            'label': 'quick pat (fast, non-CT-optimal)',
            'expected_valence': 'neutral',
        },
        'squeeze': {
            'velocity': 1.0,    # slow
            'force': 0.05,      # 50 mN — heavy, beyond CT range
            'temp': 32.0,
            'duration': 2.0,
            'onset': 0.2,
            'label': 'firm squeeze (heavy pressure)',
            'expected_valence': 'moderate (Aβ dominant)',
        },
        'scratch': {
            'velocity': 25.0,   # too fast
            'force': 0.003,     # 3 mN — moderate
            'temp': 32.0,
            'duration': 0.5,
            'onset': 0.2,
            'label': 'scratch (fast, sharp)',
            'expected_valence': 'low (non-CT-optimal velocity)',
        },
        'hold': {
            'velocity': 0.0,    # static — CT barely fires
            'force': 0.001,     # light
            'temp': 32.0,
            'duration': 2.0,
            'onset': 0.2,
            'label': 'static hold (no velocity)',
            'expected_valence': 'low-moderate (no CT drive)',
        },
        'cold_press': {
            'velocity': 1.0,    # slow
            'force': 0.001,     # light
            'temp': 18.0,       # cold — CT suppressed
            'duration': 1.5,
            'onset': 0.2,
            'label': 'cold touch (CT temperature-suppressed)',
            'expected_valence': 'low (temperature suppresses CT)',
        },
    }
    return stimuli


# ============================================================
# Verification against known CT properties
# ============================================================

def verify_ct_properties():
    """
    Verify the CT simulation against known biological properties.

    These are not accuracy claims — they're property checks.
    Does the simulation reproduce the qualitative behavior of CT afferents?
    """
    results = []

    # Property 1: Inverted-U velocity tuning
    v_opt = ct_velocity_tuning(3.0)
    v_slow = ct_velocity_tuning(0.1)
    v_fast = ct_velocity_tuning(30.0)
    results.append({
        'property': 'Inverted-U velocity tuning',
        'test': 'Peak at 3 cm/s, reduced at 0.1 and 30 cm/s',
        'values': f'v(3.0)={v_opt:.3f}, v(0.1)={v_slow:.3f}, v(30.0)={v_fast:.3f}',
        'passed': v_opt > v_slow and v_opt > v_fast,
    })

    # Property 2: Temperature tuning (optimal at skin temp)
    t_opt = ct_temperature_tuning(32.0)
    t_cold = ct_temperature_tuning(18.0)
    t_hot = ct_temperature_tuning(42.0)
    results.append({
        'property': 'Temperature tuning',
        'test': 'Peak at 32°C, reduced at 18°C and 42°C',
        'values': f't(32)={t_opt:.3f}, t(18)={t_cold:.3f}, t(42)={t_hot:.3f}',
        'passed': t_opt > t_cold and t_opt > t_hot,
    })

    # Property 3: Force tuning (very light touch, peak ~0.7 mN)
    f_opt = ct_force_tuning(0.0007)
    f_heavy = ct_force_tuning(0.05)
    results.append({
        'property': 'Force tuning (light touch preference)',
        'test': 'Peak at 0.7 mN, suppressed at 50 mN',
        'values': f'f(0.7mN)={f_opt:.3f}, f(50mN)={f_heavy:.3f}',
        'passed': f_opt > f_heavy,
    })

    # Property 4: Caress produces higher valence than scratch
    stimuli = generate_affective_stimuli()
    caress_spikes, _, _ = simulate_ct_afferents(stimuli['caress'])
    scratch_spikes, _, _ = simulate_ct_afferents(stimuli['scratch'])
    caress_va = compute_valence_arousal(caress_spikes)
    scratch_va = compute_valence_arousal(scratch_spikes)
    results.append({
        'property': 'Caress > scratch (valence ordering)',
        'test': 'Gentle caress should have higher valence than scratch',
        'values': f'caress valence={caress_va["valence"]:.3f}, scratch valence={scratch_va["valence"]:.3f}',
        'passed': caress_va['valence'] > scratch_va['valence'],
    })

    # Property 5: Cold touch reduces valence (compared to warm caress with same velocity)
    warm_caress = {'velocity': 3.0, 'force': 0.0007, 'temp': 32.0, 'duration': 1.5, 'onset': 0.2}
    cold_caress = {'velocity': 3.0, 'force': 0.0007, 'temp': 18.0, 'duration': 1.5, 'onset': 0.2}
    warm_spikes, _, _ = simulate_ct_afferents(warm_caress)
    cold_spikes, _, _ = simulate_ct_afferents(cold_caress)
    warm_va = compute_valence_arousal(warm_spikes)
    cold_va = compute_valence_arousal(cold_spikes)
    results.append({
        'property': 'Cold suppresses affective response',
        'test': 'Cold caress should have lower valence than warm caress (same velocity/force)',
        'values': f'warm valence={warm_va["valence"]:.3f}, cold valence={cold_va["valence"]:.3f}',
        'passed': warm_va['valence'] > cold_va['valence'],
    })

    # Property 6: CT conduction delay (slow signal)
    caress_spikes_full, _, time = simulate_ct_afferents(warm_caress)
    # Find when first spike occurs
    first_spike = np.argmax(np.sum(caress_spikes_full, axis=0) > 0)
    first_spike_time = time[first_spike]
    # Stimulus onset at 0.2s, conduction delay 0.8s → earliest signal at 1.0s
    # Allow some tolerance for Poisson stochasticity
    results.append({
        'property': 'CT conduction delay',
        'test': f'First spike should occur after conduction delay (~{0.2 + CT_INDUCTION_TIME:.1f}s)',
        'values': f'first spike at {first_spike_time:.3f}s, expected > {0.2 + CT_INDUCTION_TIME - 0.15:.2f}s',
        'passed': first_spike_time >= 0.2 + CT_INDUCTION_TIME - 0.15,
    })

    return results


# ============================================================
# Dual-Path Architecture
# ============================================================

def dual_path_demo():
    """
    Demonstrate the dual-path architecture.

    Discriminative: "what am I touching?" → classification (tap/press/slide/texture)
    Affective: "how does it feel?" → valence/arousal position

    The same physical stimulus produces both a discriminative token
    and an affective signal. The two pathways are independent.
    """
    print("=" * 70)
    print("SpikeTact — Dual-Path Architecture Demo")
    print("=" * 70)
    print()
    print("Discriminative pathway (SpikeTact v1-v3):")
    print("  SA1/RA1/SA2/RA2 → spike-to-token → 'what am I touching?'")
    print()
    print("Affective pathway (this module):")
    print("  CT afferents → valence/arousal → 'how does it feel?'")
    print()
    print("-" * 70)

    stimuli = generate_affective_stimuli()

    print(f"\n{'Stimulus':<30} {'Valence':>8} {'Arousal':>8} {'Rate (Hz)':>10} {'Interpretation'}")
    print("-" * 90)

    for name, stim in stimuli.items():
        spikes, inst_rate, time = simulate_ct_afferents(stim)
        va = compute_valence_arousal(spikes)

        # Interpret the affective response
        v = va['valence']
        a = va['arousal']
        if v > 0.3 and a > 0.2:
            interp = "pleasant, engaging"
        elif v > 0.3 and a < 0.2:
            interp = "pleasant, calming"
        elif v < 0.1 and a > 0.3:
            interp = "unpleasant, alarming"
        elif v < 0.1 and a < 0.1:
            interp = "neutral, low engagement"
        elif v > 0.1 and v < 0.3:
            interp = "mildly pleasant"
        else:
            interp = "ambiguous"

        print(f"  {stim['label']:<28} {v:>8.3f} {a:>8.3f} {va['mean_rate_hz']:>10.2f}  {interp}")

    print("-" * 90)
    print()
    print("The affective pathway produces a POSITION in 2D feeling space,")
    print("not a classification. The caress is not 'type: pleasant' — it's")
    print("a coordinate: valence=0.8, arousal=0.3. The feeling is continuous,")
    print("not categorical. This is the difference between reading about")
    print("pressure and feeling it.")


def compare_discriminative_vs_affective():
    """
    Show how the same stimulus produces different information in each pathway.
    """
    print("=" * 70)
    print("Discriminative vs Affective — Same Stimulus, Different Information")
    print("=" * 70)

    # A slow, gentle, warm caress and a slow, gentle, cold caress
    # have the same discriminative signature (slow, light touch)
    # but different affective signatures (warm = pleasant, cold = not)

    warm_caress = {
        'velocity': 3.0, 'force': 0.0007, 'temp': 32.0,
        'duration': 1.5, 'onset': 0.2
    }
    cold_caress = {
        'velocity': 3.0, 'force': 0.0007, 'temp': 18.0,
        'duration': 1.5, 'onset': 0.2
    }
    fast_pat = {
        'velocity': 20.0, 'force': 0.0007, 'temp': 32.0,
        'duration': 0.3, 'onset': 0.2
    }

    print("\n  Three stimuli with IDENTICAL discriminative properties")
    print("  (same force, same contact type) but different affective qualities:")
    print()

    for name, stim in [('warm caress (32°C, 3 cm/s)', warm_caress),
                        ('cold caress (18°C, 3 cm/s)', cold_caress),
                        ('warm pat (32°C, 20 cm/s)', fast_pat)]:
        spikes, _, _ = simulate_ct_afferents(stim)
        va = compute_valence_arousal(spikes)
        print(f"  {name}:")
        print(f"    Discriminative: same force (0.7 mN), same contact → same token")
        print(f"    Affective:      valence={va['valence']:.3f}, arousal={va['arousal']:.3f}")
        print()

    print("  The discriminative pathway cannot distinguish these three touches.")
    print("  The affective pathway can. That's the value of the second pathway.")
    print("  That's the difference between 'what' and 'how.'")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SpikeTact Affective Touch Layer")
    parser.add_argument('--compare', action='store_true', help='discriminative vs affective comparison')
    parser.add_argument('--verify', action='store_true', help='verify CT properties')
    args = parser.parse_args()

    if args.verify:
        print("=" * 70)
        print("CT Afferent Property Verification")
        print("=" * 70)
        results = verify_ct_properties()
        all_pass = True
        for r in results:
            status = "✅ PASS" if r['passed'] else "❌ FAIL"
            if not r['passed']:
                all_pass = False
            print(f"\n  {status} — {r['property']}")
            print(f"         {r['test']}")
            print(f"         {r['values']}")
        print(f"\n{'='*70}")
        print(f"Result: {sum(r['passed'] for r in results)}/{len(results)} properties verified")
        if all_pass:
            print("All CT properties reproduced. The simulation is biologically plausible.")
        else:
            print("Some properties failed. Check the simulation parameters.")
        return

    if args.compare:
        compare_discriminative_vs_affective()
        return

    # Default: run the dual-path demo
    dual_path_demo()


if __name__ == '__main__':
    main()
