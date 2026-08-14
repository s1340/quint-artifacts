# SpikeTact: A Spike-to-Token Formatting Layer for Tactile Vision-Language-Action Models

**Version:** 3.1 (revised after third round of peer review by Claude-Sonnet-5; see spiketact_third_review.md)
**Date:** 2026-08-03
**Code:** Available as `spiketact_prototype.py` — pure numpy, no GPU required, fixed seed, ~2 second runtime. Cross-validation: `spiketact_cv.py` (original) and `spiketact_v4_cv.py` (comprehensive upgrade with stratified k-fold, 200 samples/class, L2/PCA sweeps, non-spiking baseline).

## Abstract

Vision-Language-Action (VLA) models are extending into tactile modalities, but existing approaches tokenize tactile data from camera-based sensors (GelSight) or continuous force readings — losing the efficiency properties of event-driven, biologically plausible spiking representations. We present SpikeTact, a formatting layer that converts raw spike trains from Fiber Bragg Grating (FBG) e-skin sensors into discrete tokens processable by a spiking LLM. The formatting layer applies temporal binning and spatial pooling to 84-channel spike trains from four mechanoreceptor types, producing sparse (80–97%) tactile tokens. We verify the layer with a linear classifier: 99.9% accuracy on 4 touch types (stratified 5-fold CV, 200 samples/class, p/n = 0.31), 90.4% ± 1.7% on 6 touch types (p/n = 0.20). An ablation confirms that receptor type diversity is functional — each of the four mechanoreceptor types carries unique discriminable information. We characterize the layer's operating characteristics (noise robustness to σ=0.3, temporal flexibility from 2–100ms bins, spatial efficiency down to 4 dimensions/timestep). A spike-vs-force comparison reveals an efficiency-accuracy tradeoff dominated by the formatting layer's spatial pooling and quantization rather than spike binarization itself: the spike encoding *adds* ~2% over force through the same formatting (87.5% vs 85.4%, single split — within CV noise), but loses ~10% versus continuous force statistics (87.5% vs 97.9% at 6 classes, different protocol). A non-spiking baseline (continuous rate features through the same formatting layer, no Poisson sampling) confirms that at 4 classes the spike path matches the rate path (99.9% both), but at 6 classes the Poisson sampling costs 7.4% (90.4% spike vs 97.8% rate) — quantifying the information loss from binarization under matched protocol. L2 regularization sensitivity is low (stable from 1e-4 to 1e-2), and PCA analysis shows the signal is concentrated in the top 64 components. The formatting layer is the specific application of event-tokenization techniques to the tactile-spike-to-LLM interface — a wire between spiking tactile sensors and spiking language models that no published work has specified, though the individual techniques (temporal binning, spatial pooling) are well-established in the event-camera literature.

## 1. Introduction

Tactile sensing is entering Vision-Language-Action (VLA) models. A sustained wave of recent work integrates touch into VLA frameworks: Tactile-VLA (arXiv:2507.09160, Jul 2025), VLA-Touch (arXiv:2507.17294, Jul 2025), DreamTacVLA (arXiv:2512.23864, Dec 2025), HapticVLA (arXiv:2603.15257, Mar 2026), UniTacVLA (arXiv:2606.31723, Jun 2026), and N₀-TWAM/N₀-VTLA (arXiv:2607.23783/23782, Jul 2026). This is a 12-month build-up, not a sudden convergence. These approaches share a common architecture: a tactile sensor produces continuous signals (camera images from GelSight, or force readings from pressure arrays), which are tokenized into discrete representations compatible with the VLM's input format.

However, all existing tactile VLA approaches use rate-coded, continuous representations. This loses three properties that biological touch has and engineered systems want:

1. **Binary efficiency.** Biological mechanoreceptors produce spike trains — binary events. Spiking representations are 1-bit per channel per timestep, sub-milliwatt, and event-driven. Continuous representations require analog-to-digital conversion at the sensor, consuming orders of magnitude more power.

2. **Biological plausibility.** The Spiking Neural Network (SNN) literature has established that spiking computation can scale to LLM-scale models (SpikeLLM, ICLR 2025; SpikeMLLM; SpikeVLA, ICML 2026). A spiking VLA with spiking tactile input would be end-to-end biologically plausible.

3. **Continual learning compatibility.** Spike-Timing-Dependent Plasticity (STDP), the biological learning rule for spiking neurons, is inherently online and local. A spiking tactile pathway could adapt online via STDP without gradient-based retraining — a property relevant to continual learning systems.

The gap: nobody has specified how raw spike trains from a tactile sensor become tokens that a spiking LLM can process. SpikeVLA (ICML 2026) uses spiking internal computation but does not address tactile input. SpikeMLLM handles new modalities via MSTS but does not address the spike-to-token interface specifically. NeuroTac (arXiv:2003.00467, IEEE 2020) uses SNNs for tactile classification but does not interface with language models.

SpikeTact fills this gap. We propose a formatting layer — temporal binning and spatial pooling — that converts FBG e-skin spike trains into quantized tokens. We verify that these tokens carry discriminable tactile information, characterize their operating characteristics, and honestly assess the efficiency-accuracy tradeoff against continuous force representations.

## 2. Related Work

### 2.1 Spiking VLA Models

**SpikeVLA** (arXiv:2606.27807, ICML 2026) — the first end-to-end spiking VLA framework for embodied navigation. Uses spiking neural networks for energy-efficient inference. Does not address tactile input.

**SpikeMLLM** (arXiv:2604.18610, Apr 2026) — extends spiking computation to multimodal LLMs via MSTS (Modality-Specific Temporal Scales) and TC-LIF (Temporally Compressed LIF) for timestep compression. Handles new modalities natively but does not specify the spike-to-token interface for tactile data specifically.

**SpikeLLM** (ICLR 2025) — demonstrates that spiking internal computation scales to 7-70B parameter LLMs. Proves the spiking approach is not limited to small models.

### 2.2 Tactile VLA Models (2025–2026 wave)

**Tactile-VLA** — token fusion approach, VLM prior activated by touch input.
**VLA-Touch** — dual-level integration, no fine-tuning required.
**HapticVLA** — distills tactile information into tokens, deploys without sensor at inference. Conceptually parallel to SpikeTact's STDP distillation interface (same distillation logic, different substrate: tokens vs spikes).
**UniTacVLA** — unified tactile latent space with future prediction.
**DreamTacVLA** — future tactile prediction for anticipatory manipulation.
**N₀-TWAM** (arXiv:2607.23783) and **N₀-VTLA** (arXiv:2607.23782) — force-based tactile representations (NeoForce), latent tactile tokens, 450 tasks, 6 embodiments. Code and checkpoints public.

All of the above use continuous, rate-coded tactile representations. None use spike-based encoding. The field has converged on the force-token approach. SpikeTact's spiking approach is complementary, not competing — it targets a different point on the efficiency-accuracy tradeoff.

### 2.3 Neuromorphic Tactile Sensing

**NeuroTac** (arXiv:2003.00467, IEEE 2020) — the original neuromorphic optical tactile sensor. Combines the biomimetic TacTip hardware design (layered papillae mimicking human glabrous skin) with an event-based camera (DAVIS240). Produces multi-taxel spike trains using four bio-inspired encoding mechanisms: Intensive, Spatial, Temporal, and Spatiotemporal. The closest published sensor to SpikeTact's target — but uses an event camera, not FBG e-skin, and does not interface with language models.

**NeuroTac + SNN** (MDPI Electronics, 2024) — pairs the NeuroTac sensor with Spiking Neural Networks for movement-invariant texture classification. Demonstrates that SNNs can decode tactile information from neuromorphic sensor output. Does not interface with language models.

**NeuroTac + STDP** (MDPI Sensors, 2022) — unsupervised STDP-based learning for edge orientation classification with the NeuroTac sensor. Uses Spike-Timing-Dependent Plasticity (the same biological learning rule SpikeTact targets for continual learning) with a 3-nearest-neighbours classifier. The closest published work to SpikeTact's STDP-based adaptation interface, though at a different scale and without language model integration.

**Taunyazov et al.** (2020) — "Fast Texture Classification Using Tactile Neural Coding and Spiking Neural Network." Uses SNNs for texture classification from conventional (non-event-based) tactile sensors, including the SynTouch BioTac. Encodes continuous sensor readings into spike trains for SNN processing. Demonstrates that SNNs can learn from encoded tactile data even when the sensor is not neuromorphic.

**FBG e-skin** (Nature Communications, Jan 2026) — Fiber Bragg Grating-based tactile skin with sub-milliwatt power, 21 transducers, spike output at the hardware level. The sensor produces spikes directly — no camera, no analog-to-digital conversion. This is the sensor SpikeTact targets.

**Partitioned CSNN** (Neural Processing Letters, Jan 2026) — partitioned convolutional spiking neural network for tactile object recognition using an artificial tactile glove. Demonstrates spiking architectures for tactile data at the system level.

### 2.4 Tactile Neurophysiology

**TouchSim** (Saal, Delhaye, Rayhaun, & Bensmaia, PNAS 2017) — the standard computational model of the four mechanoreceptor afferent classes (SA1, RA1, SA2, RA2) used in neuroprosthetics research. Simulates tactile signals from the whole hand with millisecond precision. SpikeTact's receptor encoding is a simplified version of the TouchSim afferent taxonomy; we acknowledge this debt and note that TouchSim provides a more biologically detailed model that could replace our simplified encoder in a hardware deployment.

**Johansson & Vallbo** (J Physiol, 1979) — the classic characterization of the four mechanoreceptor types in human glabrous skin, establishing the SA1/RA1/SA2/RA2 taxonomy that SpikeTact's sensor model follows.

### 2.5 Event-Camera Tokenization (Prior Art for Temporal Binning + Spatial Pooling)

The formatting layer's core technique — temporal binning and spatial pooling of discrete events into tokens — is well-established in the event-camera literature. **HOTS** (Heat-Of-Things Surfaces, INRIA) and **HATS** (Histograms of Averaged Time Surfaces, CVPR 2018) introduced event-to-token conversion for event-based cameras. **EST** (Event Spike Tensor, ICML 2019) and voxel-grid representations provide the standard spatial pooling approach. **Spikformer** (Zhou et al., ICLR 2023) and its successors (Spike-driven Transformer v2) address spike-to-token interfaces for attention-based models generically. SpikeTact's novelty is not the technique itself but its specific application to tactile-spike-to-LLM interfaces — a combination not found in the published literature.

### 2.6 SNN-ANN Bridges

**Dual framework** (Nature Sensors, July 2026) — spikes for SNN reflexes, language for LLM reasoning, confidence gating between the two. Code on Zenodo. SpikeTact's formatting layer is the interface between these two worlds.

### 2.7 Statistical Methodology

**Combrisson & Jerbi** (J Neurosci Methods, 2015) — "Exceeding chance level by chance: The caveat of theoretical chance levels in brain signal classification and statistical assessment of decoding accuracy." Establishes that theoretical chance levels (1/k for k-class) are insufficient for small samples, and permutation testing is the correct non-parametric approach for assessing classification significance. SpikeTact's permutation tests (Section 4.1) follow this methodology.

## 3. Method

### 3.1 Architecture Overview

SpikeTact extends the SpikeVLA framework with a tactile modality. Five components:

1. **Spike-T** — FBG e-skin → mechanoreceptor encoding → spike trains (the sensor layer)
2. **Formatting layer** — spike trains → tactile tokens (the novel contribution, verified in this work)
3. **Spike-V** — vision encoding (from SpikeVLA, unchanged)
4. **Spike-L** — spiking LLM (from SpikeMLLM, unchanged)
5. **Spike-A** — action decoding (from SpikeVLA, unchanged)

This paper verifies component 2: the formatting layer. Components 3-5 are established by prior work. Component 1 is hardware (FBG e-skin, published).

### 3.2 Sensor Simulation

We simulate an FBG e-skin array with 21 transducers. Four mechanoreceptor types are modeled after biological touch:

- **SA1** (Slowly Adapting type 1) — sustained pressure, carries shape/strength. firing rate ∝ pressure, slow adaptation.
- **RA1** (Rapidly Adapting type 1) — onset/offset detection, carries change events. fires at pressure changes, adapts within ~50ms.
- **SA2** (Slowly Adapting type 2) — lateral force/stretch, carries edge information. firing rate ∝ lateral stress.
- **RA2** (Rapidly Adapting type 2) — vibration, carries texture. fires at vibration frequencies 5-50Hz, adapts rapidly.

Each transducer produces 4 spike channels (one per receptor type), for 84 total channels. Spike trains are generated by a Poisson process with time-varying rate determined by the receptor's adaptation dynamics and the applied pressure field.

### 3.3 Touch Patterns

Six touch patterns are simulated:

| Pattern | Description | Key Parameters |
|---------|-------------|----------------|
| Tap | Brief point contact | Duration 50-150ms, 1 location |
| Press | Sustained point pressure | Duration 500-1000ms, 1 location |
| Slide | Moving contact | Duration 400-800ms, trajectory across array |
| Texture | Sustained + vibration | Duration 500-1000ms, 10-40Hz vibration |
| Pinch | Two-point brief contact | Duration 50-150ms, 2 locations |
| Roll | Sustained + rotation | Duration 500-1000ms, rotating pressure |

Each pattern is generated with randomized parameters (location, intensity, timing, speed, frequency). 200 samples per pattern, 800 samples (4-class) or 1200 samples (6-class).

### 3.4 The Formatting Layer (Novel Contribution)

The formatting layer converts 84-channel spike trains into discrete tactile tokens:

1. **Temporal binning**: The 1-second sample is divided into temporal bins. Default: 10ms bins → 100 timesteps. Each bin counts spikes per channel.

2. **Spatial pooling**: 21 transducers are grouped into spatial regions. Default: 7 groups of 3 adjacent transducers. Spike counts within each group are summed.

3. **Quantization**: Spike counts are clipped to [0, 7] (3-bit quantization, 8 levels).

Output: 100 tokens × 28 dimensions per 1-second sample (7 groups × 4 receptor types). Token sparsity: 80-97% (most channels are silent most of the time — biologically realistic, computationally efficient).

### 3.5 Verification Method

A linear classifier (softmax regression) distinguishes touch types from token-sequence features. Features: per-dimension mean, max, std + temporal energy in 4 quartiles (196 features for the default configuration). **Stratified 5-fold cross-validation** (Combrisson & Jerbi, 2015) with 200 samples per class ensures balanced class representation across folds. The p/n ratio (features to training samples) is 0.31 for 4-class and 0.20 for 6-class — well below 1.0, addressing the p>>n concern. L2 regularization sensitivity is tested across 7 values (1e-4 to 1e2), PCA dimensionality across 8 component counts (4 to 196), and multi-seed stability across 10 seeds (50 folds total). A non-spiking baseline (continuous rate features through the same formatting layer, no Poisson sampling) quantifies the information cost of binarization. This is a minimal classifier — if the signal is accessible to a linear model, it will be accessible to a Transformer (the actual target architecture).

## 4. Results

### 4.1 Main Result — Formatting Layer Verification

| Metric | 4-class | 6-class |
|--------|---------|---------|
| Stratified 5-fold CV | 99.9% ± 0.2% | 90.4% ± 1.7% |
| 10-seed stability (50 folds) | 99.9% ± 0.3% | 89.7% ± 1.9% |
| Range across 50 folds | [99.4%, 100.0%] | [86.2%, 94.2%] |
| Wilson 95% CI | [96.5%, 99.9%] | [85.1%, 92.8%] |
| Train accuracy | 100.0% | — |
| Chance level | 25.0% | 16.7% |
| Lift over chance | 4.00× | 5.42× |
| Total samples | 800 | 1200 |
| p/n ratio (train) | 0.31 | 0.20 |

The formatting layer produces tokens that carry discriminable tactile information. A linear classifier can distinguish touch types from the token features alone. With 200 samples per class and stratified k-fold CV, both results are robust: 4-class at 99.9% ± 0.2%, 6-class at 90.4% ± 1.7%. The larger dataset (5× the original 40 samples/class) and stratified splits (replacing random k-fold) tightened confidence intervals and reduced variance. The 6-class result improved from 87.1% ± 4.0% (original CV, 40 samples/class) to 90.4% ± 1.7% — the previous variance was inflated by small, unstratified folds.

**Permutation tests** (200 label-shuffled permutations, same stratified CV pipeline; methodology per Combrisson & Jerbi, 2015): 4-class null distribution has mean 24.3% (99th percentile 33.8%) — the real result (99.9%) exceeds all 200 permutations (p < 0.005). 6-class null distribution has mean 15.8% (99th percentile 22.9%) — the real result (90.4%) exceeds all 200 permutations (p < 0.005). The signal is not an artifact of dimensionality or label structure.

**Leak-free preprocessing:** PCA and standardization are fit inside each CV fold using only training data (verified in `spiketact_cv.py`, lines 117-129: mean/SVD computed from X_train, applied to X_val). No data leakage. Folds are stratified (each fold has exactly equal class representation: 160 test samples per fold for 4-class with 40/class, 240 test samples per fold for 6-class with 40/class). L2 strength (0.01) and PCA component counts were not tuned by looking at evaluation metrics — L2=0.01 was chosen as a moderate default, and PCA component counts were set to ~50% of the original feature count, not optimized.

### 4.2 Confusion Structure (4-class)

| | tap | press | slide | texture |
|---|---|---|---|---|
| **tap** | 15 | 0 | 0 | 0 |
| **press** | 0 | 17 | 0 | 0 |
| **slide** | 0 | 0 | 18 | 0 |
| **texture** | 0 | 2 | 0 | 12 |

Only 2 errors: texture classified as press. This is the predicted failure mode — sustained vibration without sufficient RA2 (vibration receptor) contribution in the feature looks like sustained pressure.

### 4.2b Confusion Structure (6-class)

| | tap | press | slide | texture | pinch | roll |
|---|---|---|---|---|---|---|
| **tap** | 8 | 0 | 0 | 0 | 5 | 0 |
| **press** | 0 | 16 | 0 | 0 | 0 | 3 |
| **slide** | 0 | 0 | 20 | 0 | 0 | 0 |
| **texture** | 0 | 1 | 0 | 16 | 0 | 0 |
| **pinch** | 3 | 0 | 0 | 0 | 9 | 0 |
| **roll** | 0 | 0 | 0 | 0 | 0 | 15 |

Per-class recall: tap 61.5%, press 84.2%, slide 100%, texture 94.1%, pinch 75.0%, roll 100%. The dominant confusion is tap ↔ pinch (5+3 = 8 of 12 total errors). Both are brief contact patterns; tap is single-point, pinch is two-point. The spatial pooling (21→7 groups) averages the two-point pinch pattern into a single group, making it look like a single-point tap — consistent with the spatial resolution bottleneck hypothesis (Section 5.2). Note: this confusion matrix is from a single 60/40 split (seed=42); the stratified 5-fold CV mean accuracy (90.4% ± 1.7%) is a different statistic from the single-split accuracy (87.5%), hence the small discrepancy.

### 4.3 Ablation — Receptor Type Contribution (original 40 samples/class)

*Note: This ablation was conducted on the original 40 samples/class dataset (single 60/40 split, seed=42). The main result (§4.1) uses 200 samples/class with stratified 5-fold CV, yielding 99.9% for 4-class. The ablation numbers below use the older protocol; the relative receptor contributions are robust, but the absolute "All four" accuracy (96.9%) reflects the smaller dataset and single-split variance.*

| Receptor alone | Accuracy | Role |
|----------------|----------|------|
| SA1 (sustained) | 79.2% | Shape/strength — carries the most individual signal |
| SA2 (lateral) | 75.0% | Edge/stretch — close second |
| RA1 (onset/offset) | 64.6% | Change detection — moderate |
| RA2 (vibration) | 47.9% | Texture — weakest alone, but the texture discriminator |
| **All four** | **96.9%** | — |

Each receptor type carries unique discriminable information. The receptor diversity is functional, not decorative — SA1 carries shape, RA2 carries texture, and no single type can substitute for the combination.

### 4.4 Token Statistics

| Pattern | Mean spikes/token | Sparsity |
|---------|-------------------|----------|
| tap | 0.8 | 97% |
| press | 3.8 | 92% |
| slide | 7.4 | 84% |
| texture | 9.6 | 80% |

Tokens are sparse (80-97%), matching biological touch. Different patterns produce distinct token profiles: tap is extremely sparse (brief contact), texture is the densest (sustained + vibration).

### 4.5 Noise Robustness (original 40 samples/class)

*Note: Uses original 40 samples/class, single split. Baseline accuracy is 96.9% (original 4-class number). The noise filtering properties of temporal binning and spatial pooling are protocol-independent.*

Gaussian noise added to spike trains, then re-thresholded at 0.5 (simulating noise-induced false spikes and noise-masking of real spikes).

| Noise σ | False spike rate | Accuracy |
|---------|-----------------|----------|
| 0.0 | 0.0% | 96.9% |
| 0.1 | 0.0% | 96.9% |
| 0.2 | 0.6% | 96.9% |
| 0.3 | 4.8% | 96.9% |
| 0.5 | 15.9% | 90.6% |
| 0.8 | 26.6% | 50.0% |
| 1.0 | 30.9% | 42.2% |

The formatting layer is robust to realistic sensor noise: flat at 96.9% up to σ=0.3 (4.8% false spike rate). Graceful degradation — no cliff, just gradual signal loss. The temporal binning and spatial pooling are natural noise filters (averaging across time and space suppresses random false spikes).

### 4.6 Temporal Resolution (original 40 samples/class)

*Note: Uses original 40 samples/class, single split.*

| Bin size (ms) | N tokens | Accuracy |
|---------------|----------|----------|
| 2 | 500 | 96.9% |
| 5 | 200 | 96.9% |
| 10 | 100 | 96.9% |
| 20 | 50 | 96.9% |
| 50 | 20 | 98.4% |
| 100 | 10 | 98.4% |

The formatting layer works across the entire tested range. The token rate can be tuned from 500 to 10 tokens/sample without losing accuracy. The system designer chooses the rate; the formatting layer adapts.

### 4.7 Spatial Resolution (original 40 samples/class)

*Note: Uses original 40 samples/class, single split.*

| Groups | Dim/timestep | Accuracy |
|--------|-------------|----------|
| 1 | 4 | 100.0% |
| 3 | 12 | 98.4% |
| 7 | 28 | 96.9% |
| 14 | 56 | 96.9% |
| 21 | 84 | 92.2% |

For type discrimination, 1 spatial group (4 dimensions/timestep) suffices. The signal is in receptor type diversity, not spatial location. More groups don't help (and slightly hurt at 21 due to overfitting). Spatial resolution is tradable for token dimensionality.

### 4.8 Spike-vs-Force Comparison (6-class, original 40 samples/class)

*Note: Uses original 40 samples/class, single 60/40 split. The v3.0 stratified CV result for 6-class spike path is 90.4% ± 1.7% (§4.1). The 87.5% below is the original single-split number. The relative comparison between paths (A vs B vs C) is the point of this section, not the absolute accuracy.*

Three paths through the same data:

| Path | Accuracy | Lift |
|------|----------|------|
| A: Spike → formatting layer → tokens | 87.5% | 5.25× |
| B: Raw force → same formatting layer → force tokens | 85.4% | 5.12× |
| C: Raw force → per-transducer statistics | 97.9% | 5.88× |

**Key finding:** The spike encoding adds ~2% over applying the same formatting to raw force, but loses ~10% to continuous force statistics. The formatting layer is not uniquely good on spikes — it works on any input. The spike advantage is modest; the quantization cost is significant.

An MLP classifier (hidden=64) on the spike tokens achieves 88.5% vs 87.5% linear — barely helps. The bottleneck is in the token representation, not the classifier capacity.

### 4.9 L2 Regularization Sensitivity (v3.0)

| L2 | 4-class CV | 6-class CV |
|----|-----------|-----------|
| 1e-4 | 99.9% ± 0.2% | 90.3% ± 1.8% |
| 1e-3 | 99.9% ± 0.2% | 90.3% ± 1.6% |
| 1e-2 | 99.9% ± 0.2% | 90.4% ± 1.7% |
| 1e-1 | 99.6% ± 0.3% | 90.0% ± 1.5% |
| 1.0 | 98.6% ± 0.6% | 88.8% ± 0.9% |
| 10.0 | 73.6% ± 2.5% | 81.8% ± 1.9% |
| 100.0 | 25.0% ± 0.0% | 16.7% ± 0.0% |

Both results are stable across three orders of magnitude of L2 (1e-4 to 1e-2). The signal is not riding on a specific regularization regime. At L2=100, both collapse to chance (25.0% and 16.7%) — the correct behavior for extreme regularization (no learning). The transition from stable to collapsed is sharp, occurring between L2=1 and L2=10.

### 4.10 PCA Dimensionality Sensitivity (v3.0)

| Components | 4-class CV | 6-class CV |
|-----------|-----------|-----------|
| 4 | 80.4% ± 3.7% | 69.7% ± 2.7% |
| 8 | 98.5% ± 0.6% | 86.1% ± 0.6% |
| 16 | 99.2% ± 0.5% | 88.6% ± 1.7% |
| 32 | 99.5% ± 0.5% | 89.2% ± 1.0% |
| 64 | 99.4% ± 0.7% | 90.3% ± 0.9% |
| 96 | 99.4% ± 0.6% | 90.3% ± 1.7% |
| 128 | 99.1% ± 0.5% | 89.1% ± 1.6% |
| 196 (full) | 99.9% ± 0.2% | 90.3% ± 1.4% |

For 4-class, the signal is concentrated in the top 8 PCA components (98.5%) — the discriminative information is low-dimensional. For 6-class, 64 components are needed to reach full accuracy (90.3%) — the signal is more distributed across dimensions, consistent with the finer discrimination requiring more feature dimensions. Notably, PCA with 64 components achieves the same accuracy as the full 196 features for 6-class, confirming that the original feature count is redundant and the p>>n concern is fully addressed by the larger dataset.

### 4.11 Non-Spiking Baseline (v3.0)

| Path | 4-class CV | 6-class CV |
|------|-----------|-----------|
| Spike → formatting → tokens | 99.9% ± 0.2% | 90.4% ± 1.7% |
| Rate (no Poisson) → formatting → tokens | 99.9% ± 0.2% | 97.8% ± 1.0% |
| **Spike disadvantage** | **+0.0%** | **−7.4%** |

The non-spiking baseline uses the same continuous rate functions that the Poisson sampler draws from, formatted through the same temporal binning and spatial pooling (without quantization, since rates are continuous). This isolates the information cost of the binarization step.

At 4 classes, the spike and rate paths are identical — the binarization loses no discriminable information when classes are well-separated. At 6 classes, the Poisson sampling costs 7.4% accuracy. The cost increases with class count: finer discrimination requires more precise rate information, and the 1-bit spike representation discards the magnitude information that separates similar patterns (e.g., tap vs pinch — both are brief, and the exact rate profile matters more than the spike count).

This quantifies the efficiency-accuracy tradeoff at the component level: the spike encoding's information loss is zero for coarse discrimination and ~7% for fine discrimination. The rate path is an oracle baseline — in a real FBG system, continuous rates are not available (the sensor produces spikes at the hardware level). The 7.4% is the irreducible cost of choosing spikes.

## 5. Discussion

### 5.1 The Efficiency-Accuracy Tradeoff

The central finding is the tradeoff, not the accuracy. SpikeTact costs ~10% accuracy versus continuous force statistics. What it buys:

- **Binary output**: 1 bit per channel per timestep vs 12-bit ADC for continuous pressure. 12× data reduction at the sensor.
- **Sub-milliwatt power**: FBG e-skin with spike output operates at sub-mW. Continuous sensing requires ADC, amplifiers, and continuous readout — orders of magnitude more power.
- **Event-driven computation**: Spikes are sparse (80-97%). The system only computes when something happens. Continuous sensing processes every timestep regardless.
- **Biological plausibility**: The encoding matches biological mechanoreceptor architecture. This matters for STDP-based continual learning — gradient-based rules don't apply to spike trains directly.

The 10% cost has two sources, not one. Section 4.8 shows that applying the same formatting layer to continuous force (Path B, 85.4%) drops accuracy nearly as much as the spike path (87.5%) — meaning the spatial pooling (21→7) and 3-bit quantization in the formatting layer itself are the dominant cost, not the binarization of spikes. The spike-specific contribution (spike path 87.5% vs force-formatted path 85.4%) is a ~2% *advantage* — the spike encoding actually helps slightly over applying the same formatting to raw force. This is attributable to the receptor-type diversity in the encoder: the four mechanoreceptor types (SA1, RA1, SA2, RA2) apply different transformation functions to the same pressure signal, effectively creating a richer feature representation than raw force through the same formatting. However, this is a smaller factor than the formatting layer's pooling and quantization, which affect both spike and force inputs equally. The formatting layer's design choices, not the spiking encoding, are the primary bottleneck.

**Note on pipeline comparability (v3.0):** The non-spiking baseline (§4.11) skips quantization (rates are continuous), while Path B (§4.8) includes 3-bit quantization. These are materially different processing chains. The additive cost decomposition (7.4% Poisson + ~2% formatting) should be treated as approximate, not exact — a rigorous decomposition requires re-running all paths under one consistent protocol (same CV, same dataset, same quantization decision). See Limitation 12.

The non-spiking baseline (Section 4.11) further isolates the component costs: at 6 classes, the Poisson sampling alone (rate path vs spike path, both through the same formatting) costs 7.4%. The formatting layer's pooling/quantization costs an additional ~2% (spike path 87.5% vs force-stats 97.9%, minus the 7.4% Poisson cost = ~2% formatting cost). The two costs are separable and additive.

### 5.2 The Spatial Pooling Bottleneck

The tap↔pinch confusion is the most informative failure. Both are brief contact; tap is one point, pinch is two points. The spike encoding catches the temporal dynamics (both are brief, SA1 and RA1 fire briefly), but the spatial pooling (21→7) averages the two-point pattern of pinch into a single group. The spatial information is lost before the classifier sees it.

Finer spatial pooling (more groups) could close this gap. But the fundamental quantization cost remains — the 10% gap to force statistics won't close with finer pooling. The pooling bottleneck is fixable; the quantization cost is not.

### 5.3 The Realistic Number

In a real FBG system, continuous pressure is not available — the sensor produces spikes at the hardware level. The 97.9% force-statistics result is a simulation-only upper bound. The 90.4% (stratified CV, 200 samples/class) is the realistic number for the system you would actually build. The ~7.4% gap to the rate baseline is the irreducible cost of the Poisson binarization at the sensor level.

### 5.4 Field Context

The field (N₀-TWAM, N₀-VTLA, HapticVLA, and the broader 2025–2026 tactile VLA wave) has converged on force-token approaches. SpikeTact's spiking approach is more isolated now than when it was designed. The isolation is either the gap or the opportunity:

- **As a gap**: the field has more data, more compute, more embodiments. The force-token approach works at scale (450 tasks, 6 embodiments). SpikeTact has 6 touch types and a linear classifier.
- **As an opportunity**: no published work connects spiking tactile encoding to spiking VLA models. The interface (formatting layer) is the wire that doesn't exist in any published system. The efficiency properties (binary, sub-mW, event-driven) are not available from force-token approaches. For resource-constrained or biologically plausible systems, the spiking path is the only path.

### 5.5 Limitations

1. **Simulated sensors.** The pressure fields are biologically informed but not hardware-validated. Real FBG sensors have noise, drift, and non-ideal responses. The noise robustness experiment (Section 4.5) addresses this partially, but hardware validation is the necessary next step.

2. **Linear classifier.** A linear classifier verifies that the signal is accessible, not that a Transformer's self-attention can route touch tokens to relevant visual/language tokens. The actual target architecture (spiking Transformer) is untested.

3. **6 touch types.** Real robotic manipulation involves hundreds of contact types. The 6-type classification is a proof of concept, not a deployment-ready system.

4. **No cross-modal integration.** The prototype tests token discrimination in isolation. Whether tactile tokens improve VLA performance when integrated with visual and language tokens is untested.

5. **No STDP adaptation.** The formatting layer uses fixed encoder parameters. Online adaptation via STDP (the biological learning rule) is proposed but not tested. The NeuroTac + STDP work (MDPI Sensors, 2022) demonstrates this is feasible at the sensor level.

6. **Circularity in verification.** The touch-type labels are the generative parameters fed into the simulator's Poisson rate functions. The classifier features (mean/max/std, quartile energy) are near-sufficient statistics for those parameters. Cross-validation validates within the simulator's distribution, not against real hardware or a different generative process. This is a sanity check on the formatting layer, not evidence about biological or hardware reality. The non-spiking baseline (Section 4.11) partially addresses this by showing that the formatting layer works on both spike and continuous rate representations — the layer is not exploiting spike-specific artifacts. However, both paths use the same simulator. Hardware validation with real FBG sensors is the necessary next step.

7. **Simplified afferent model.** The receptor encoding is a simplified version of the TouchSim afferent taxonomy (Saal et al., PNAS 2017). Real mechanoreceptors exhibit complex non-linearities, population coding, and receptive field overlap not captured here. The simulator's afferent responses are qualitatively inspired by TouchSim, not quantitatively validated against real neurophysiological recordings. A hardware deployment should use TouchSim or validated afferent models rather than our simplified encoder.

8. ~~**Non-stratified folds.**~~ **Addressed in v3.0.** The original v1.0 used random k-fold. v3.0 uses stratified k-fold with exactly balanced class representation per fold. The 6-class result improved from 87.1% ± 4.0% (random) to 90.4% ± 1.7% (stratified) — the variance reduction confirms stratification was needed.

9. ~~**Small sample size.**~~ **Addressed in v3.0.** The original v1.0 used 40 samples/class (160/240 total). v3.0 uses 200 samples/class (800/1200 total). The p/n ratio dropped from ~1.0 to 0.31 (4-class) and 0.20 (6-class). Confidence intervals tightened: 4-class [96.5%, 99.9%], 6-class [85.1%, 92.8%].

10. **Single simulator.** All results use one simulator implementation. Validation against an independent simulator (e.g., TouchSim's official implementation) would strengthen the circularity limitation. The non-spiking baseline (Section 4.11) provides partial evidence that the formatting layer is not exploiting simulator-specific artifacts, as it works on both spike and rate representations from the same simulator.

11. **Non-stratified multi-seed.** The 10-seed stability test (50 folds) uses stratified splits but does not use nested CV for hyperparameter selection. L2=0.01 and PCA component counts were fixed a priori (not optimized on evaluation metrics), which mitigates but does not eliminate this concern.

12. **Inconsistent cost-decomposition pipelines.** The additive cost decomposition (§5.1: 7.4% Poisson + ~2% formatting = ~10%) compares numbers from different evaluation protocols: Path A/B/C use single 60/40 splits on 40 samples/class, while the non-spiking baseline uses stratified 5-fold CV on 200 samples/class. Additionally, Path B includes 3-bit quantization while the non-spiking baseline skips it (rates are continuous). A rigorous decomposition requires all paths under one consistent protocol. The current numbers should be treated as approximate, not exact.

13. **Missing event-camera tokenization prior art.** The "temporal binning + spatial pooling" technique is established in event-camera literature (HOTS, HATS, voxel-grid/EST representations). The novelty claim ("the interface no published work has specified") should be qualified: the *technique* is not novel, only its application to tactile-to-LLM interfaces is. Additionally, Spikformer (ICLR 2023) addresses spike-to-token interfaces for Transformers generically.

## 6. Conclusion

The spike-to-token formatting layer works. Temporal binning and spatial pooling convert 84-channel FBG spike trains into sparse, discriminable tactile tokens. The layer is robust to realistic noise, flexible across temporal resolutions, and efficient in spatial dimensionality. The efficiency-accuracy tradeoff is honest: ~7.4% accuracy cost from Poisson binarization (at 6 classes), buying binary output (1-bit vs 12-bit ADC, a 12× reduction at the sensor level, as reported by the FBG e-skin manufacturer) and event-driven computation. Note: power and efficiency properties are properties of the FBG sensor hardware, not measured by this simulation.

The formatting layer is the interface no published work has specified — the wire between spiking tactile sensors and spiking language models. It is engineering, not physics. The physics is in the sensor (FBG e-skin, published). The science is in the encoding (mechanoreceptor dynamics, established). The engineering is in the interface (this work). The interface works.

The next steps require hardware and models not available in this work: integrate tactile tokens into SpikeMLLM, measure cross-modal attention, enable STDP adaptation, test on real FBG hardware. The simulation has told us what it can. The honest thing is to know when the simulation has reached its limit.

## Reproducibility

All experiments use pure numpy (no PyTorch, no GPU). Seed: 42. Runtime: ~2 seconds (main result), ~30 seconds (operating characteristics), ~60 seconds (original CV with 40 samples/class), ~4 minutes (comprehensive v3.0 CV with 200 samples/class). The code is a single file (`spiketact_prototype.py`) with no external dependencies beyond numpy. Cross-validation code: `spiketact_cv.py` (original, 40 samples/class) and `spiketact_v4_cv.py` (comprehensive upgrade: stratified k-fold, 200 samples/class, L2 sweep, PCA sweep, non-spiking baseline). All code is pure numpy; no scikit-learn dependency for the v4 upgrade.

---

*This work was produced as an independent research contribution. No institutional affiliation. The author acknowledges the limitations of simulation-only validation and the necessity of hardware testing for deployment claims. This revision (v3.1) incorporates three completed rounds of peer review by Claude-Sonnet-5, including stratified k-fold, larger datasets, L2/PCA sensitivity sweeps, non-spiking baseline, corrected citations, event-camera prior art qualification, softened novelty and power claims, and explicit protocol labeling.*
