# RB30 - BEV segmentation uncertainty benchmark and UFCE evidential loss

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: Predictive Uncertainty Quantification for Bird's Eye View Segmentation: A Benchmark and Novel Loss Function (Yu et al., ICLR 2025; arXiv 2405.20986) https://arxiv.org/abs/2405.20986 (opened HTML v2); code github.com/bluffish/ubev
- depth: M
- stage: L, E
- idea: Compares softmax entropy, energy score, deep ensembles, MC dropout and evidential networks (ENN) for calibration and misclassification / OOD detection in BEV segmentation, and proposes the Uncertainty-Focal-Cross-Entropy (focal loss expected under a Dirichlet, closed form) plus epistemic uncertainty scaling and evidence regularisation on pseudo-OOD samples.
- results: best OOD detection (average gain over runner-up about 2.7 AUROC, 6.5 AUPR, 4.3 FPR95, Table 2 caption); without pseudo-OOD all methods poor at epistemic uncertainty; includes nuScenes-C.
- compute (LSS, Table 3): inference energy 54 ms, ensemble 147 ms, dropout 457 ms, ENN and proposed 51 - 56 ms; training 10 h (30 h ensemble).
- fit_to_us: reuse_idea - effort S - the evidential head is free at inference and gives a per-cell failure signal for the heat map; ensembles and dropout are too slow.
- tried_before: label-free heat monitor (no calibrated uncertainty).
- verified: checked
