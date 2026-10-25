# A15 - TopoStreamer: streaming for lane-segment topology (with explicit adjacency)

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: TopoStreamer: Temporal Lane Segment Topology Reasoning in Autonomous Driving (2025) https://arxiv.org/abs/2507.00709
- depth: M
- stage: S8, S6
- idea: LaneSegNet-style lane segments with a streaming layer: BEV warped + GRU, top 30 % of queries (66 of 200) propagated through an MLP of previous queries + ego transform with residual. Streaming attribute constraints supervise coordinates, class, boundary type and a BEV mask on stream queries using instance IDs of matched positives; dynamic lane-boundary positional encoding recomputed per decoder layer from refined reference points; lane-segment denoising incl. topology (adjacency focal loss).
- formulation: topology = MLP on concatenated query projections + sigmoid, focal loss, threshold 0.5; ablation: static PE 33.8 vs dynamic 36.6 mAP; no ID tracking 34.4.
- input: BEV features, ego transform; output: lane segments + adjacency matrix.
- training: AdamW 2e-4, batch 8, 24 epochs, first 12 single-frame; Hungarian matching. Gradient through memory not stated.
- differentiable: partial (Hungarian).
- sensors_data: camera, OpenLane-V2 subset A.
- results: lane-segment mAP 36.6 vs LaneSegNet 32.6; TOP_lsls 28.5 vs 25.4; centerline OLS 44.4 vs 40.7; 13.6 FPS vs 14.7.
- compute: 13.6 FPS; parameters not stated.
- code: not checked
- fit_to_us: reuse_idea - effort M - confirms that memory helps topology (+3.1 TOP_lsls) and that the cost is about 1 FPS; their topology head is a matched adjacency MLP, which we do not use.
- compat_needs: Hungarian-matched instance ids.
- compat_conflicts: our edges come from the tracer; their adjacency is a pairwise head (dropped in E42 / E45).
- tried_before: E42, E45, E46 learned link heads: dropped (rules decide).
- verified: partly
