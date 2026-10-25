# A14 - PredMapNet: per-instance raster history memory + short-term forecast on top of MapTracker

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: PredMapNet: Future and Historical Reasoning for Consistent Online HD Vectorized Map Construction (2026) https://arxiv.org/abs/2602.16669
- depth: M
- stage: S8
- idea: Each tracked instance keeps a stored BEV mask, blended with the new prediction (confidence-weighted, beta = 0.9), warped by ego motion each step, pruned when the track is lost. History-Map Guidance lets track queries cross-attend to BEV features sampled where the stored mask is above 0.5. Short-Term Future Guidance: an MLP predicts per-point offsets to the next frame from the last n decoded polylines.
- formulation: loss adds mask (Dice 2, BCE 1), Chamfer trajectory loss and auxiliary depth loss. Ablation on MapTracker (74.7 mmAP baseline): +SAQG 0.3, +HMG 0.4, +STFG 0.9, +depth 0.6 -> 76.9.
- input / output: as MapTracker.
- training: 72 epochs nuScenes in three stages, 4 x A100; the paper says masks come from the query generator so "end-to-end differentiability" is preserved; detachment of the memory not stated.
- differentiable: partial (matching) / memory claim not verified.
- sensors_data: camera, nuScenes + AV2.
- results: nuScenes old split 76.9 mmAP / 69.7 C-mmAP (MapTracker 76.1 / 69.1); AV2 77.3 / 69.1; non-overlapping splits +1.8 and +0.9 mmAP.
- compute: 10.1 FPS vs MapTracker 10.9.
- code: not checked
- fit_to_us: reuse_idea - effort M - confidence-weighted blending of a stored dense mask with decay is close to our hold; the gain over MapTracker is small (+0.8 mmAP) for 7 % slower.
- compat_needs: tracked instances.
- compat_conflicts: identity tracking.
- tried_before: none.
- verified: partly
