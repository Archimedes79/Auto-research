# A07 - PrevPredMap / Uni-PrevPredMap: previous predictions as queries and map priors, trained with perturbed priors and mode dropout

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: PrevPredMap: Exploring Temporal Modeling with Previous Predictions for Online Vectorized HD Map Construction (2024) https://arxiv.org/abs/2407.17378 ; Uni-PrevPredMap: Extending PrevPredMap to a Unified Framework of Prior-Informed Modeling (2025) https://arxiv.org/abs/2504.06647
- depth: M (both)
- stage: S8, T
- idea: The previous frame's top-k predictions (k = 10 on nuScenes) are ego-transformed and used directly as decoder queries (category and confidence -> content query, location -> position query), no BEV memory. Training randomly picks single-frame mode with p = 0.5. The unified version stores predictions and map priors in a UTM-tile global vector map (write threshold 0.4 train / 0.9 test) and trains a tri-mode model (no prior, temporal prior, temporal + map prior) with perturbed priors (about half the vectors displaced by up to +-6 m).
- formulation: performance saturates at about 12 previous queries; the paper notes a very large k adds uncertain predictions that hurt. Robustness table (nuScenes, perfect map 85.6): instance displacement +-3/6/9 m 81.9/80.9/80.3; frame displacement +-3/6/9 m 77.0/75.7/74.6 (empty prior 74.0); rotation +-15 deg and scale 0.8-1.2 both 77.0.
- input: previous predictions (+ optional map vectors); output: vector map.
- training: dual-mode; Hungarian host. Gradient across frames not stated.
- differentiable: partial (prediction dictionary updated during training, matching).
- sensors_data: camera, nuScenes / AV2.
- results: PrevPredMap nuScenes 66.3 mAP temporal vs 65.7 single mode (24 epochs), 71.3 at 110 epochs; FPS 15.7 vs 16.1. Uni-PrevPredMap: no prior 64.9, temporal prior 74.0 (24 ep) / 77.0 (72 ep), map prior 71.3, both 80.9 / 81.8; FPS 14.2 / 12.2 / 13.1 / 11.5 (A6000).
- compute: see above; the previous-prediction prior costs about 6.5 % FPS.
- code: not checked
- fit_to_us: reuse_idea - effort S - this is the closest published analogue of our "prior from the previous frame's graph" recursion; the transferable tricks are (i) train with p = 0.5 no-history, (ii) perturb the prior by metres, (iii) confidence-ranked top-k, (iv) test unseen perturbation types.
- compat_needs: previous graph in ego frame with confidences; our vertices (position + token + validity) are directly usable as "queries".
- compat_conflicts: they decode with a DETR set + Hungarian; ours is a tracer; the global tile store needs pose.
- tried_before: loop 16 one-step recursion (prior from previous graph) - this paper's perturbation / mode-dropout recipe was not reported as used there (check).
- verified: partly (PrevPredMap pose handling: only the phrase "ego transformation" was found)
