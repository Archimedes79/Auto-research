# A09 - MapTCL: bidirectional temporal consistency loss (training-only)

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: MapTCL: Temporal Consistency Learning via Bidirectional Alignment for Vectorized HD Map Construction (2026, IROS 2026) https://arxiv.org/abs/2608.05209
- depth: M
- stage: L, T
- idea: Two auxiliary losses that vanish at inference. BVCL: past-frame predictions with confidence above 0.3 are warped into the current frame and Hungarian-matched to current predictions (instance + point level), and the reverse; SmoothL1 on points + KL on class distribution, weights decay with frame distance. RCL: a BEV raster head with focal loss; past raster maps warped to the current frame and compared with a focal consistency term with decaying weights.
- formulation: total = map loss + BVCL + RCL (beta = 1.0 for RCL); memory length N_t = 5 best (AV2 56.7 mAP, 51.6 at N_t = 7); tau = 0.3 best (0.1 -> 54.0 mAP / 23.8 C-mAP). Applied only in the temporal training stage.
- input: model outputs of past and current frames; output: loss.
- training: 4 x A6000, batch 16, 24 epochs nuScenes / 6 AV2.
- differentiable: partial (Hungarian inside BVCL; RCL fully differentiable).
- sensors_data: camera, nuScenes / AV2 new splits.
- results: new split 60x30 m mAP / C-mAP: StreamMapNet nuScenes 35.2 / 25.7 -> 38.9 / 28.5; SQD-MapNet 36.6 / 25.4 -> 37.9 / 26.8; MapTracker 39.4 / 30.8 -> 41.0 / 32.4; heavy occlusion StreamMapNet 29.5 -> 32.4.
- compute: none at inference.
- code: no link on the page
- fit_to_us: reimplement - effort S - the RCL variant (consistency between warped previous dense output and current dense output) needs no matching and is exactly our heat-map output; bidirectional warp avoids favouring either frame. Directly attacks flicker, relevant to our "temporal consistency" metric.
- compat_needs: dense output per frame, ego warp; for BVCL a matching which we avoid -> use RCL only.
- compat_conflicts: BVCL uses Hungarian (our rule: not in our losses).
- tried_before: none.
- verified: partly
