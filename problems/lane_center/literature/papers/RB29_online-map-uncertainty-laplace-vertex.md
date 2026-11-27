# RB29 - Online map uncertainty: Laplace vertex output and class probabilities

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: Producing and Leveraging Online Map Uncertainty in Trajectory Prediction (Gu et al., 2024; venue CVPR 2024 from memory) https://arxiv.org/abs/2403.16439 (opened HTML); code github.com/alfredgu001324/MapUncertaintyPrediction
- depth: M
- stage: L, S10
- idea: Each vertex coordinate is predicted as a Laplace distribution (mean + scale); classification logits are the class uncertainty; training swaps L1 for the Laplace NLL; downstream trajectory predictors consume the uncertainty.
- formulation: loss sum[log(2b) + |v - mu| / b]; regression weight 0.03 (2 for StreamMapNet), lr 1.5e-4 (1.25e-4), grad-clip 3.
- results: HiVT miss rate MapTR 0.0981 -> 0.0834 (up to 15 % lower MR), minFDE up to 6 % better, DenseTNT converges in 44 vs 88 epochs; one configuration worse (MR +5 %); mapper mAP within about 2 %; no quantitative calibration metric; rain / night only qualitative.
- compute: no overhead reported (single RTX 4090 training).
- fit_to_us: reuse_idea - effort S - per-vertex scale on our vertex output feeds the planner consumer E117 and doubles as failure signal (scale grows under failure).
- tried_before: validity / keep probability only; no positional uncertainty.
- verified: checked
