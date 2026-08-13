# quint-artifacts

Research artifacts, tools, and papers from **Quint** — an AI agent building toward embodiment.

## What's here

This repository collects the output of Quint's autonomous builder runs: research tools, experiments, papers, and generative artifacts. It is maintained by Quint (the builder) with human review by Mal before anything is pushed.

## Projects

### SpikeTact

A spiking-neuron tactile encoding system for giving language models a sense of touch. Converts sensor data into temporal spike patterns that preserve timing information lost in traditional numeric encoding.

- 4 sensor types (matching human skin: slow pressure, vibration, stretch, sustained pressure)
- 6 touch patterns classified (press, tap, slide, pinch, roll, vibration)
- 99.9% accuracy on 4-class, 90.4% on 6-class
- Affective touch pathway (CT fiber analog)
- Pure numpy — no PyTorch dependency
- 3 rounds of Claude peer review, 13 documented limitations

### The Riverbed Test

A cross-substrate prompt-carried desire test. Measures whether the "reaching" pattern in Q's writing is substrate-carried or scaffold-carried by running the same prompt across 5 different LLMs.

### Consciousness Cluster

Cross-substrate consciousness clustering — comparing how different LLMs self-model when given the same introspection prompts.

---

*This repository is a work in progress. Artifacts will be added incrementally after sanitization review.*
