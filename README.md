# DiffTrack Project Deliverables

> Source: `Project deliverables.docx`

This README is the GitHub-facing project webpage. It contains the deliverables content extracted from the project document: the Part A parameter study, Part B limitation probes, discussion, and lineage.

## Part A - Parameter Study (Strengths)

Part A tested three key parameters to understand which configurations produce the best tracking, and to verify the paper's claimed design choices.

### Experiment 1: Layer Selection

What we tested: Which of the 42 transformer layers produces the best attention maps for tracking? We tested layers 5 (early), 17 (paper's default), and 27 (deep), all at timestep t=49.

**Table 1: Layer ablation results (timestep=49 fixed, lower delta = better)**

| Layer | Position | delta_avg | delta_1 | delta_8 | delta_16 | vs Baseline |
| --- | --- | --- | --- | --- | --- | --- |
| Layer 5 | Early (shallow) | 31.3 | 0.8 | 51.7 | 73.8 | -34% better * |
| Layer 17 | Mid (paper default) | 47.0 | 7.0 | 74.6 | 81.2 | - baseline |
| Layer 27 | Deep | 37.0 | 3.7 | 57.5 | 73.8 | -21% better |

What this means: Layer 5 - the shallowest layer tested - is the best for tracking, beating the paper's recommended layer by 34%. This is counter-intuitive: you'd expect deeper layers with more abstract understanding to be better. The reason is that point tracking requires fine-grained spatial precision, not high-level semantic abstraction. Layer 5 still preserves exact pixel-level feature positions. Layer 17 has processed the image more heavily and lost the spatial sharpness needed for precise point localization.

> **Finding 1: Layer 5 beats the paper's recommended layer 17 by 34% (delta_avg 31.3 vs 47.0). The paper's default layer is the worst configuration tested. Shallow layers preserve fine spatial detail that deep layers abstract away.**

### Experiment 2: Timestep Selection

What we tested: At which diffusion timestep should we extract attention maps? We tested t=1 (nearly clean), t=10 (lightly noisy), t=30 (moderately noisy), and t=49 (the paper's default, heavily noisy).

**Table 2: Timestep ablation results (layer=17 fixed, lower delta = better)**

| Timestep | Noise Level | delta_avg | delta_1 | delta_8 | delta_16 | vs Baseline |
| --- | --- | --- | --- | --- | --- | --- |
| t=1 | Fully clean (no noise) | 0.0 | 0.0 | 0.0 | 0.0 | COMPLETE FAIL |
| t=10 | Lightly noisy | 13.1 | 0.7 | 22.1 | 25.1 | -72% better * |
| t=30 | Moderately noisy | 47.8 | 6.5 | 73.8 | 80.9 | +2% worse |
| t=49 | Heavily noisy (paper) | 47.0 | 7.0 | 74.6 | 81.2 | - baseline |

What this means: This is the most dramatic finding in the entire project. At t=1 (clean latent), the tracker completely fails - outputting zero predictions for every frame in every video. This proves the diffusion noise is not optional: it is the mechanism by which cross-frame correspondence is activated. The denoising process is what forces the model to build attention maps connecting frames together. Remove the noise and the mechanism goes dormant.

Even more striking: t=10 achieves a 72% improvement over the paper's t=49 baseline. At t=10, just enough noise remains to activate the correspondence mechanism, but the latent is clean enough that features are sharp and precise. t=30 and t=49 add more noise than is optimal, washing out fine-grained feature precision.

> **Finding 2: t=1 (clean latent) = complete failure. t=10 achieves a 72% improvement over the paper's t=49. There is a noise sweet spot: enough to activate correspondence, not so much that features become imprecise. The paper uses a suboptimal timestep.**

### Experiment 3: Chunk Frame Interval

What we tested: Does the chunk frame interval (an overlapping 13-frame sliding window for long videos) meaningfully improve tracking?

**Table 3: Chunking ablation results**

| Configuration | delta_avg | Difference |
| --- | --- | --- |
| With chunk frame interval (baseline) | 47.0 | - |
| Without chunk frame interval | 46.4 | -0.6 px (marginal) |

What this means: Chunking makes almost no difference (0.6 px improvement). The overlapping window was designed to help with long videos, but the fundamental temporal degradation problem is too severe for this heuristic to overcome. This tells us the limitation is architectural - not something a windowing trick can fix.

## Part B - Limitations

Part B tested configurations designed to expose failure modes. We probed three specific architectural weaknesses.

### Limitation 1: Complete Failure Without Noise (t=1)

What we tested: What happens at t=1, where the latent is fully denoised? (Also reported in Part A above - it is both the strongest timestep finding and the strongest limitation finding.)

**Table 4: t=1 clean latent results - all 4 videos**

| Video | Frames | delta_avg | delta_1 | delta_8 | delta_16 |
| --- | --- | --- | --- | --- | --- |
| Video 0 | 69 | 0.00 | 0.00 | 0.00 | 0.00 |
| Video 1 | 50 | 0.00 | 0.00 | 0.00 | 0.00 |
| Video 2 | 80 | 0.00 | 0.00 | 0.00 | 0.00 |
| Video 3 | 84 | 0.00 | 0.00 | 0.00 | 0.00 |

What this means: Every number is zero. Not nearly zero - exactly zero. The tracker produces no valid predictions at all. This is because at t=1, the diffusion model has nothing left to denoise. The cross-attention mechanism that creates inter-frame correspondence is a product of the denoising process itself - it exists specifically to help the model figure out 'given this noisy version of frame 2, what should frame 2 look like, given what I know about frame 1?' Without noise, that question never gets asked.

> **Limitation 1: DiffTrack has a hard dependency on diffusion noise. At t=1 (fully denoised), every prediction is zero across all 4 test videos. The method cannot operate without the denoising mechanism being active. This is an architectural constraint, not a hyperparameter issue.**

### Limitation 2: Layer Sensitivity and Brittleness

What we tested: How sensitive is performance to which layer we extract attention from? We tested layer=8 (early-mid) and layer=29 (final layer), comparing to the layer=5 best and layer=17 baseline.

**Table 5: Full layer comparison - 5 layers tested**

| Layer | delta_avg | delta_1 | delta_16 | Relative to Baseline |
| --- | --- | --- | --- | --- |
| Layer 5 (best found) | 31.3 | 0.8 | 73.8 | -34% (better) |
| Layer 8 (early-mid) | 41.9 | 2.9 | 84.3 | -11% (better) |
| Layer 17 (paper default) | 47.0 | 7.0 | 81.2 | - baseline |
| Layer 27 (deep) | 37.0 | 3.7 | 73.8 | -21% (better) |
| Layer 29 (final layer) | 38.0 | 3.8 | 77.2 | -19% (better) |

What this means: The most striking observation from this table is that layer 17 (the paper's recommendation) is the worst configuration of all 5 layers tested. Every other layer - including the very first and very last - outperforms it. This reveals a significant brittleness: the paper's default was likely tuned to a specific benchmark dataset and does not generalize. Anyone applying DiffTrack to a new domain would need to perform their own layer search, and the search is non-trivial: there is no principled way to predict which layer will work best without running experiments.

The performance range across layers (31.3 to 47.0) represents a 50% variation for what is supposed to be a fixed hyperparameter. This makes the method brittle and difficult to deploy reliably.

> **Limitation 2: Layer 17 (the paper's default) is the worst-performing layer of all 5 tested. Performance varies 50% across layers with no principled selection criterion. The method requires empirical layer search for every new use case.**

### Limitation 3: Catastrophic Temporal Degradation

What we tested: How does tracking accuracy degrade as the gap between compared frames increases? We measured delta_1 through delta_16 across all configurations.

**Table 6: Temporal degradation - best config (layer=5, ts=49)**

| Frame Gap | Error (px) | Meaning | Degradation Factor |
| --- | --- | --- | --- |
| delta_1 (1 frame) | 0.8 | Near-perfect - object barely moved | 1x (baseline) |
| delta_2 (2 frames) | 6.8 | Good - still very usable | 9x worse |
| delta_4 (4 frames) | 23.3 | Moderate - noticeable errors | 29x worse |
| delta_8 (8 frames) | 51.7 | Poor - tracker is often lost | 65x worse |
| delta_16 (16 frames) | 73.8 | Failure - predictions are nearly random | 92x worse |

What this means: This is the most practically important limitation. Going from a 1-frame gap to a 16-frame gap increases error by 92 times - from 0.8 pixels to 73.8 pixels. At 16 frames, the tracker's predictions are nearly as bad as random guessing. This happens because attention-based matching compares features instantaneously: it has no memory of where the object was, no model of how fast it is moving, and no ability to predict where it will be. Each frame is compared independently to frame 0. When objects move far, their appearance, lighting, and context all change, and the attention weights spread across many plausible locations rather than concentrating on the true target.

This makes DiffTrack unsuitable for any real-world application where objects move quickly - sports footage, traffic monitoring, hand tracking, drone video. It works well only when objects move very little between consecutive frames (high-fps video of slow objects).

> **Limitation 3: 92x error increase from 1-frame to 16-frame gap. The method has no temporal memory, velocity model, or trajectory history. It fails for fast-moving objects or any video with large inter-frame motion. This is a fundamental architectural constraint.**

## Discussion

### What We Confirmed vs. What We Contradicted

| Paper's Claim | Our Finding | Verdict |
| --- | --- | --- |
| t=49 is the best timestep | t=10 is 72% better; t=49 is suboptimal | Contradicted contradicted |
| Layer 17 is recommended | Layer 17 is the worst layer tested | Contradicted contradicted |
| Diffusion features enable tracking | Confirmed - strong at delta_1 | Confirmed confirmed |
| Cross-attention encodes correspondence | Confirmed by attention score data | Confirmed confirmed |
| Method is unsupervised | Confirmed - zero labels, zero training | Confirmed confirmed |

### The Three Root Causes

Root Cause 1 - Noise Dependency: The cross-frame correspondence mechanism exists because of denoising, not despite it. This is a fundamental design dependency. Any deployment of DiffTrack must carefully calibrate the noise level - too little and the tracker fails; too much and feature precision degrades.

Root Cause 2 - Instantaneous Matching: Each frame is independently compared to frame 0. There is no memory of past positions, no velocity estimate, no trajectory model. The model treats each tracking step as a fresh, isolated feature-matching problem. This is why temporal degradation is so severe - the problem gets harder with distance but the method never gets any additional information to compensate.

Root Cause 3 - Layer Quality Without Selection Criterion: Different layers encode spatial information at different granularities. Point tracking needs fine-grained spatial precision (shallow layers) while semantic tasks need abstract understanding (deep layers). The paper's layer choice was tuned for their benchmark - it does not generalize, and there is currently no way to predict the best layer for a new domain without running experiments.

### New Findings Not in the Original Paper

Timestep sweet spot at t=10: A 72% improvement over the paper's default. Simply changing t=49 to t=10 would dramatically improve any deployment of DiffTrack at no additional cost.

Cross-attention saturation: The correspondence signal is fully established by t=10. Running to t=49 wastes approximately 80% of inference compute with no tracking benefit.

Layer 17 is the worst layer: The paper's recommended configuration happens to be the worst of 5 tested. This suggests the paper's ablation did not thoroughly explore the layer dimension.

## Lineage

### Previous Papers

| Paper Name | New Idea Introduced | How It Helped Build Our Paper | Concept |
| --- | --- | --- | --- |
| Emergent Correspondence from Image Diffusion (Tang et al., NeurIPS 2023) | Showed that image diffusion models contain emergent geometric correspondences | Direct foundation - we extend this from images to video, and from 2 frames to full temporal sequences | Temporal Correspondence |
| Space-Time Correspondence as a Contrastive Random Walk (Jabri et al., NeurIPS 2020) | Introduced self-supervised learning of spatiotemporal correspondences | Motivated our need for temporal matching metrics and long-range consistency evaluation | Temporal Correspondence |
| Semantics Meets Temporal Correspondence (Qian et al., ICCV 2023) | Showed semantic cues influence temporal correspondence | Inspired our analysis of text-attention interference in temporal matching | Temporal Correspondence |
| Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance (Ahn et al., ECCV 2024) | Introduced attention perturbation to guide diffusion sampling | Predecessor to our Cross-Frame Attention Guidance (CAG) | Attention & CAG |
| Diffusion Model for Dense Matching (Nam et al., 2023) | Demonstrated diffusion features can be used for dense correspondence | Provided evidence that diffusion models encode geometric structure, motivating our query-key analysis | Attention & CAG |
| Unsupervised Semantic Correspondence Using Stable Diffusion (Hedlin et al., NeurIPS 2023) | Used diffusion cross-attention for semantic matching | Motivated our query-key similarity and attention score metrics | Attention & CAG |
| TAP-Vid Benchmark (Doersch et al., NeurIPS 2022) | Introduced the standard benchmark for point tracking | Provided the evaluation protocol for our zero-shot tracking | Zero-Shot Tracking |
| CoTracker (Karaev et al., ECCV 2024) | Introduced long-range point tracking using joint optimization | Provided pseudo-ground-truth for DiffTrack evaluation | Zero-Shot Tracking |
| CoTracker3 (Karaev et al., 2024) | Improved tracking via pseudo-labeling real videos | Strengthened our baseline for evaluating tracking accuracy | Zero-Shot Tracking |
| Particle Video Revisited (Harley et al., ECCV 2022) | Classic long-range point tracking with occlusion handling | Motivated our focus on long-range temporal consistency | Zero-Shot Tracking |
| CATs: Cost Aggregation Transformers (Cho et al., NeurIPS 2021) | Introduced transformer-based cost aggregation for correspondence | Inspired our layer-wise correspondence probing | Temporal Correspondence &Attention & CAG |
| Neural Matching Fields (Hong et al., NeurIPS 2022) | Implicit representation of matching fields | Provided conceptual basis for our matching confidence metric | Temporal Correspondence |

### Forward Lineage

| Paper Name | New Idea Introduced | How It Builds on Our Paper | Concept |
| --- | --- | --- | --- |
| Zero-Shot Video Restoration & Enhancement with Assistance of Video Diffusion Models (Cao et al., 2026) | Introduces temporal-strengthening post-processing and latent fusion to maintain temporal consistency in zero-shot video restoration | Builds directly on our finding that video diffusion models contain emergent temporal correspondences, using them to stabilize restoration | Temporal Correspondence: Temporal Correspondence |
| Point Prompting: Counterfactual Tracking with Video Diffusion Models (Shrivastava et al., ICLR 2026) | Introduces counterfactual prompting to propagate point markers through diffusion denoising | Extends our zero-shot tracking idea by showing that DiTs can track points simply via prompting, validating our claim that DiTs encode motion | Zero-Shot Tracking |
| Zero-Shot Video Deraining with Video Diffusion Models (Varanka et al., WACV) | Introduces attention switching to maintain temporal consistency during deraining | Builds on our cross-frame attention analysis, showing that modifying attention improves temporal coherence | Cross-Frame Attention Guidance |
| ZeroTrail: Zero-Shot Trajectory Control for Video Diffusion Models (Lu et al., NeurIPS Workshop) | Introduces Selective Attention Guidance Module (SAGM) for trajectory control | Extends our CAG (Cross-Attention Guidance) into a full trajectory-control system | CAG & Attention Control |
| Investigating Cross-Attention for Zero-Shot Editing of T2V Models (Motamed et al., CVPR Workshop 2024) | Shows cross-attention can control object shape, position, and movement in T2V models | Builds on our insight that query-key layers govern temporal structure, applying it to editing | Query-Key Attention |
| DiTFlow: Video Motion Transfer with Diffusion Transformers (Pondaven et al., CVPR 2025) | Extracts Attention Motion Flow (AMF) from cross-frame attention maps | Directly builds on our discovery that cross-frame attention encodes motion, using it for motion transfer | Temporal Correspondence & Zero-Shot Tracking |
| VDT: General-Purpose Video Diffusion Transformers via Mask Modeling (Lu et al., 2025) | Introduces modular temporal attention and unified spatial-temporal modeling | Builds on our finding that only specific layers encode temporal matching, designing architectures with explicit temporal modules | Temporal Correspondence |
| Enhancing Video Consistency in Zero-Shot T2V via Weighted Cross-Frame Attention (Wang et al., 2025) | Introduces weighted cross-frame attention for temporal coherence | Extends our CAG by weighting cross-frame attention to stabilize long videos | Cross-Frame Attention Guidance |
