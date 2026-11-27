# RB35 - Temporal redundancy: StreamMapNet and a stability benchmark

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction (Yuan et al., WACV 2024) https://arxiv.org/abs/2308.12570 (abstract page opened); Stability Under Scrutiny: Benchmarking Representation Paradigms for Online HD Mapping (arXiv 2510.10660, Oct 2025) https://arxiv.org/abs/2510.10660 (abstract page opened; PDF too large to read)
- depth: A
- stage: S8, E
- idea: StreamMapNet carries a memory feature through the sequence, 14.2 FPS; the benchmark measures temporal stability (presence, localisation, shape stability, mean Average Stability) on 42 models and states that accuracy and stability are largely independent.
- results: temporal gain under corruption is quantified in MapBench (RB18): mRR 47.1 -> 55.5; MapBench and the Samsung entry (RB20) give +2.4 mIoU from a temporal module.
- fit_to_us: reuse_idea - effort S - evaluation: add a stability score to our temporal-consistency metric; supports learned recursion over the spatial hold (cf. RB02).
- tried_before: spatial hold; input memory in system B; temporal feature fusion (E41 +0.008 F1).
- verified: partly
