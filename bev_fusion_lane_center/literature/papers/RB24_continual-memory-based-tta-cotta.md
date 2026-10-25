# RB24 - Continual / memory-based TTA: CoTTA, RoTTA, distribution-aware tuning

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: Continual Test-Time Domain Adaptation (Wang et al., CVPR 2022) https://arxiv.org/abs/2203.13591; Robust Test-Time Adaptation in Dynamic Scenarios (RoTTA, Yuan et al., CVPR 2023) https://arxiv.org/abs/2303.13899; Distribution-Aware Continual Test-Time Adaptation for Semantic Segmentation (arXiv 2309.13604) https://arxiv.org/abs/2309.13604 (all abstract pages opened)
- depth: A
- stage: T
- idea: CoTTA: weight-averaged teacher + augmentation-averaged predictions + stochastic restore of a few weights to the source. RoTTA: robust BN, category-balanced memory bank with recency / uncertainty weighting, time-aware reweighting, for correlated streams such as driving. DAT: update only two small parameter groups (domain-specific for big shifts, task-relevant for small shifts) for driving segmentation.
- results: none seen (abstracts only; CoTTA's segmentation number appeared only in a third-party summary and is not used).
- compute: CoTTA multiplies forward passes (from memory); DAT is built for low cost.
- code: https://qin.ee/cotta ; https://github.com/BIT-DA/RoTTA.
- fit_to_us: reuse_idea - effort M - none was evaluated on BEV or lane outputs; the RoTTA memory bank is the closest to a streaming driving case.
- tried_before: adaptive BN only.
- verified: checked (abstracts)
