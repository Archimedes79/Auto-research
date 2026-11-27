# A19 - GlobalMapNet: online global vector map with merging and a rendered-map input

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: GlobalMapNet: An Online Framework for Vectorized Global HD Map Construction (Shi et al., 2024) https://arxiv.org/abs/2409.10063
- depth: M
- stage: S8, S7
- idea: GlobalMapBuilder transforms local elements to world coordinates, Hungarian-matches them (Chamfer) to a clipped part of the global map and merges by replacing the matched sub-sequence; Map NMS removes duplicates by buffered IoU. GlobalMapFusion renders a local clip of the global map as soft Gaussian masks (max over elements of exp(-d / tau)) plus a "traced region" mask and concatenates them with the BEV feature (zeros when no map).
- formulation: training with an empty map for the first 4 epochs, only 1/4 of scenes update the map. Pose-error / drift effect not analysed.
- input: BEV + rendered global map; output: global vector map.
- training: host losses; geometric builder not differentiable.
- differentiable: no for builder (Hungarian, NMS, replace); yes for rendering and fusion.
- sensors_data: camera, nuScenes / AV2.
- results: nuScenes 60x30 m mGAP 13.9 -> 17.6, mAP 32.9 -> 34.8; AV2 mGAP 31.7 -> 35.6; cross-scene nuScenes 10.5 -> 15.2. Latency not reported.
- compute: not stated.
- code: not checked
- fit_to_us: reuse_idea (the "traced region" mask: a channel that says where the system has already looked) - effort S; the global builder not suitable (matching, pose drift).
- compat_needs: global pose.
- compat_conflicts: Hungarian merging.
- tried_before: E40 / E48 map prior at the grid.
- verified: partly
