# RB32 - Night images: LightDiff and DualCross (image-level fixes)

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: Light the Night: A Multi-Condition Diffusion Framework for Unpaired Low-Light Enhancement in Autonomous Driving (Li et al., CVPR 2024) https://arxiv.org/abs/2404.04804 (abstract page opened); DualCross: Cross-Modality Cross-Domain Adaptation for Monocular BEV Perception (authors not read, IROS 2023) https://arxiv.org/abs/2305.03724 (abstract page opened)
- depth: A
- stage: B, T
- idea: LightDiff enhances dark camera images with a diffusion model conditioned on a degraded image, LiDAR-derived depth and text, trained on synthetic day-night pairs and aligned to detectors with RL rewards. DualCross transfers LiDAR knowledge into a camera-only monocular BEV model for cross-domain use.
- results: abstract level, no numbers; LightDiff improves several 3D detectors at night on nuScenes.
- compute: not stated; a diffusion pass per image is expected to be expensive (my inference).
- fit_to_us: not_suitable - effort L - need raw images in the loop and add latency; the foundation is frozen and cached.
- verified: checked (abstracts)
