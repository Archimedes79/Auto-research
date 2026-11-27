# RB20 - RoboDrive challenge and the Samsung-RAL map-segmentation entry: what practitioners did

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: The RoboDrive Challenge: Drive Anytime Anywhere in Any Condition (arXiv 2405.08816, 2024) https://arxiv.org/abs/2405.08816 (abstract page opened); Team Samsung-RAL: Technical Report for 2024 RoboDrive Challenge - Robust Map Segmentation Track (arXiv 2405.10567) https://arxiv.org/abs/2405.10567 (opened HTML v2)
- depth: M (technical report) / A (challenge)
- stage: T, E
- idea: Challenge: 140 teams, five tracks; top solutions used heavy augmentation, multi-sensor fusion, self-supervised error correction. Samsung-RAL: BEVerse with Swin backbones, temporal module, image augmentations (GridMask, hue, saturation, channel swap, value).
- results: Table 2 temporal: 21.68 -> 24.08 mIoU (+2.4); backbone Tiny / Small / Base 16.31 / 17.33 / 24.08; BEVerse-Tiny with augmentation 29.76 (phase 2, ranked third; tables mix phase 1 and 2).
- compute: 4 x A6000, 20 epochs.
- fit_to_us: baseline - effort S - ranking of levers: backbone strength (+7.8) > temporal (+2.4) > augmentation.
- tried_before: temporal hold, failure fine-tuning.
- verified: checked
