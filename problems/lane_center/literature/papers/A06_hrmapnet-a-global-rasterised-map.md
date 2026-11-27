# A06 - HRMapNet: a global rasterised map built from the network's own past predictions

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: Enhancing Vectorized Map Perception with Historical Rasterized Maps (Zhang et al., ECCV 2024) https://arxiv.org/abs/2409.00620
- depth: M+C
- stage: S8, S9
- idea: Local vector predictions are rasterised (3 channels, 0.3 m cells) and merged into a global grid by a per-pixel counter: +30 where the local map is 1, -1 where 0 (asymmetric evidence, 8-bit, about 1 MB/km). The current local patch is retrieved with the ego pose (round(R p + t)), thresholded, and used twice: extra BEV queries at non-empty cells inside the BEV encoder (F_BEV = Conv(Concat(F_I + F_M, M_l))), and prior embeddings that initialise the decoder queries via cross-attention.
- formulation: S_th not stated; no decay or log-odds clamp stated; map starts empty each epoch in training; whether gradients pass through the map is not stated. Pose noise: sigma_t 0.2 m, sigma_r 0.02 rad gives 63.8 mAP (baseline MapTRv2 61.5); training-time pose augmentation helps robustness at some accuracy cost (sigma_r 0.1 rad: 63.1 vs 65.8 clean).
- input: image BEV features + retrieved raster; output: vector map (set).
- training: standard map loss of the host; empty map first.
- differentiable: no for the map itself (rasterise / threshold / counter update); yes in the fusion convolutions.
- sensors_data: camera, nuScenes / AV2.
- results: nuScenes mAP StreamMapNet 60.4 -> 66.3, MapTRv2 61.5 -> 67.2 (24 epochs); night 39.6 -> 42.6, rainy 50.8 -> 62.6, with an empty initial map; a map pre-built from training data gives 83.7 (overlap with val, not fair); location-disjoint split StreamMapNet 33.9 -> 37.1.
- compute: FPS 22.5 -> 21.1 (StreamMapNet), 19.6 -> 17.0 (MapTRv2); +9 GB training memory; no ms figures.
- code: https://github.com/HXMap/HRMapNet (MIT; MMDetection3D-style folder, mmcv not confirmed)
- fit_to_us: reuse_idea - effort S/M - the asymmetric counter update is a cheap non-learned long-horizon memory for lane evidence and is the one place where night / rain gains were quantified; it is a non-differentiable side memory, so it can only be an input channel.
- compat_needs: global pose; vector -> raster step.
- compat_conflicts: requires consistent global pose over a drive; our pose-jump failure would corrupt it.
- tried_before: E40 / E48 map prior at the grid with learned registration (+0.075 F1 exact pose, -0.057 unaligned) - same family, same lesson.
- verified: checked
