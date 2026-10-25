# A01 - FASTopoWM: fast-slow pipelines with a latent world model for lane-segment topology

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: FASTopoWM: Fast-Slow Lane Segment Topology Reasoning with Latent World Models (authors not captured, arXiv 2025) https://arxiv.org/abs/2507.23325
- depth: M
- stage: S8, T
- idea: Two decoders share layers and heads: a "slow" branch that replaces the lowest-ranked initial queries by propagated stream queries and fuses the previous BEV by GRU, and a "fast" single-frame branch. Both are trained in parallel with the same losses; at inference the slow output is used when the pose is reliable, otherwise the fast output. Two small "world models" (query and BEV) predict the current frame's stream queries / BEV from the previous frame, conditioned on an action latent made from the relative ego pose, with an MSE / query loss against the actual current frame.
- formulation: slow loss = lane-segment loss + 0.3 x latent loss; fast loss = lane-segment loss. World-model depth 2 layers chosen (2/4/6 layers give 37.1 - 37.8 mAP). Action conditioning: none 36.5, predicted trajectory 31.2, relative ego pose 37.4 mAP. Architecture of the latent model: linear 35.2, MLP 36.2, transformer 37.4. Criterion for "pose reliable" is not stated.
- input: BEV features (from images), previous stream queries + BEV, relative ego pose.
- output: lane segments (centreline + boundaries) + topology (adjacency), Hungarian-matched DETR set.
- training: Hungarian matching; 24 epochs, first 12 single-frame; parallel fast-slow training 37.4 vs random alternation 35.3 temporal mAP.
- differentiable: partial - Hungarian matching in the loss; gradient flow through the propagated memory not stated.
- sensors_data: camera, OpenLane-V2 subset A / B.
- results: OpenLane-V2 subset A: temporal 37.4 mAP, TOP_lsls 29.6, 11.4 FPS; the same model run single-frame 34.1 mAP at 14.0 FPS; best prior temporal in its table (SQD-MapNet) 26.0, best prior non-temporal (Topo2Seq) 33.6. Subset B: 46.3 OLS temporal vs 41.8 single-frame.
- compute: 11.4 FPS (temporal) vs 14.0 FPS (single-frame); parameters not stated.
- code: not checked
- fit_to_us: reuse_idea - effort S/M - the principle (train with and without memory in parallel, switch by a reliability signal, condition the propagation on ego motion) fits our hold + recursion; the pose-reliability test is unspecified, we would use the foundation's own confidence.
- compat_needs: a signal that tells "memory trustworthy" (pose OK, previous frame OK); previous-frame features and relative pose.
- compat_conflicts: their queries / Hungarian set; world model is trained against the actual current features (needs a stable target, ok for cached features).
- tried_before: E39 / E41 temporal feature fusion (+0.008 F1, kept small); time recursion in builder (loop 16). The fast-slow switch and the latent predictor are new.
- verified: partly (digest of paper; authors and venue not captured)
