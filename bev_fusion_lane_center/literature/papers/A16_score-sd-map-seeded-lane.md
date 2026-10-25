# A16 - Score (CORTE): SD-map-seeded lane queries with ConvGRU temporal fusion

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: Coherent Online Road Topology Estimation and Reasoning with Standard-Definition Maps (Pham, Witte, Behley, Betz, Stachniss, 2025) https://arxiv.org/abs/2507.01397
- depth: A (page via aggregator; paper body not opened)
- stage: S8, S9
- idea: Single network over multi-view images + SD map: 50 lane queries whose reference points are sampled along SD edges plus learned offset, DN-style denoising queries, lane attention, ConvGRU recurrent BEV fusion with ego-motion compensation, topology post-processing blending learned similarity and a distance term (alpha = 2.5, beta = 0.8, tuned on validation).
- formulation: not read.
- input: images, SD map; output: lane segments, boundaries, traffic elements, topology.
- training: not read.
- differentiable: not read.
- sensors_data: camera + SD map, OpenLane-V2 subset A.
- results: aggregator page: OLUS 54.9 val / 55.8 test; temporal fusion raises TOP_ll from 36.2 to 40.0 (per the aggregator, not verified in the paper).
- compute: not stated.
- code: not checked
- fit_to_us: reuse_idea - effort M - one of the few places where SD-map seeding and temporal memory are combined for lane topology; the +3.8 TOP_ll from ConvGRU is the size of gain to expect.
- compat_needs: SD map aligned in ego frame (assumed aligned; misplacement not corrected).
- compat_conflicts: as A15.
- tried_before: our route head E104b handles SD map separately.
- verified: partly (aggregator page; numbers not cross-checked)
