# A08 - SQD-MapNet: stream query denoising (training-only)

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: Stream Query Denoising for Vectorized HD-Map Construction (Wang et al., ECCV 2024) https://arxiv.org/abs/2401.09112
- depth: M
- stage: T, S8
- idea: In training, the previous frame's ground truth is ego-warped into the current frame, noised, and fed as extra "denoising" queries that must reconstruct the current frame's ground truth. Adaptive Temporal Matching pairs previous and current instances by bidirectional Chamfer distance below a threshold relative to instance size; the noise magnitude decays with the matched distance. Removed at inference: zero cost.
- formulation: noise = box shift (lambda1, lambda2 x width / height) and box scale ((1 +- lambda3), (1 +- lambda4)); values of lambda not stated; L = L_map + lambda4 L_focal^DN + lambda5 L_line^DN. Best matching scale alpha = 0.1, decay gamma = 0.2.
- input: previous GT (training); output: n/a at inference.
- training: same losses as the map head on the denoising queries; single-frame training uses ordinary denoising.
- differentiable: partial (matching).
- sensors_data: camera, nuScenes + AV2.
- results: nuScenes 60x30 m 63.9 mAP vs StreamMapNet 60.4; ablation 59.9 (single frame) -> 59.2 (streaming alone) -> 62.9 (dynamic noising) -> 63.9 (+ATM). No quantitative consistency metric in the paper.
- compute: no inference cost (training only).
- code: "available soon" at v1; not checked since
- fit_to_us: reimplement - effort S - for our recursion: train the tracer on the previous-frame ground-truth graph warped and perturbed in position, scale, and with dropped / spurious pieces, so the prior can be wrong; note the ablation shows streaming alone slightly hurt (59.2) until denoising was added, i.e. a memory that is trusted blindly can lose.
- compat_needs: GT graph of the previous frame (we have it), ego transform.
- compat_conflicts: none (loss-level).
- tried_before: none seen.
- verified: partly
