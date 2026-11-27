# A33 - D2HDMap: driveline prior with noise-trained robustness and removable prior

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: D2HDMap: Non-visible Driveline Map Prior for Online Vectorized HD Map Prediction (2026) https://arxiv.org/abs/2606.20725
- depth: M
- stage: S9, T
- idea: A static rasterised driveline (centreline sequence) prior and Uni-PrevPredMap's temporal prior are retrieved with the ego pose, rasterised, concatenated and used to initialise queries. Trained with translation noise 0 - 2 m and rotation noise 0 - 5 deg on the prior.
- formulation: at 1.0 m / 2.5 deg noise mAP 33.3 vs 44.8 clean; robust training gives +3.86 mAP on average and shrinks the worst-case drop from -19.4 to -13.1.
- results: nuScenes geosplit 60x30 m 44.8 mAP vs MapDiffusion 35.6; without prior at test time 37.8 (-7.0), versus -42.3 for an HD-map-prior model. AV2 numbers inconsistent between table and text.
- compute: latency not reported.
- code: not checked
- training: as host.
- differentiable: partial (host).
- sensors_data: camera, nuScenes / AV2.
- fit_to_us: reuse_idea - effort S - recipe for a prior that can be missing or wrong: train with noise and with prior removed (modes), report the no-prior drop; no lane-graph metric given.
- compat_needs: prior raster.
- compat_conflicts: none.
- tried_before: E40 / E48 (map prior), E104b (route head with SD raster).
- verified: partly
