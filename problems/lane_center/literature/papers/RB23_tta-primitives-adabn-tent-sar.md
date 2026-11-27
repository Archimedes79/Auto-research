# RB23 - TTA primitives: AdaBN, TENT, SAR

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: Revisiting Batch Normalization For Practical Domain Adaptation (AdaBN, 2016; authors from memory Li et al.) https://arxiv.org/abs/1603.04779; Tent: Fully Test-time Adaptation by Entropy Minimization (Wang et al., ICLR 2021, both from memory) https://arxiv.org/abs/2006.10726; Towards Stable Test-Time Adaptation in Dynamic Wild World (SAR, Niu et al., ICLR 2023) https://arxiv.org/abs/2302.12400 (all abstract pages opened; pre-2023 papers listed as background)
- depth: A
- stage: T
- idea: AdaBN replaces BN statistics by target statistics (parameter-free). TENT additionally minimises prediction entropy over BN scale / shift. SAR states TTA fails under mixed shifts, small batches and label imbalance, blames BN, and fixes it by dropping high-gradient samples and a sharpness-aware step.
- formulation: not read. results: none seen beyond RB22's batch-1 BEV table.
- compute: AdaBN no extra pass; TENT one backward over affine parameters per batch; SAR more.
- code: TENT / SAR links not seen on the pages.
- fit_to_us: reuse_idea - effort S - ordering by safety: AdaBN with mixing < SAR-style filtered entropy; entropy of a dense lane heat map is a possible loss but untested.
- tried_before: adaptive BN (+0.04).
- verified: checked (abstracts)
