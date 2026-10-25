# A05 - MapTracker: raster + vector memory with strided fusion and pose-noise training

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping (Chen, Wu, Furukawa, ECCV 2024 oral) https://arxiv.org/abs/2403.15951
- depth: M+C(README)
- stage: S8
- idea: Mapping is tracking. BEV memory M_BEV (50x100x256, 15 m left/right, 30 m front/back, last 20 frames) is warped to the new frame by affine ego motion + bilinear sampling; pixels outside the previous map are filled with learnable per-pixel embeddings ("MaskBlend"). Vector memory = one 512-d latent per tracked element, propagated by a two-layer PropMLP fed with quaternion + translation (positionally encoded); new candidates (100) join the propagated set. Strided fusion: four past memories at about 1, 5, 10, 15 m travelled distance are aligned, concatenated, passed through a residual conv block (BEV) or cross-attention (vector).
- formulation: L = L_BEV (10 focal + 1 Dice) + L_track (5 focal + 50 line) + 0.1 L_trans. Stride ablation (nuScenes old split): {1,5,10,15} 76.1 mAP / 69.1 C-mAP, {5,10,15,20} C-mAP 51.5. Pose noise: Gaussian noise added to the transform during training (no standalone test-time evaluation). Clip training: 4 of the previous 10 frames. Thresholds 0.4 first frame, 0.5 propagated, 0.6 new.
- input: BEV features from images, ego pose.
- output: tracked polylines with identities (consistent map).
- training: tracking-style label inheritance + bipartite matching for new elements; 3 stages (18/6/48 epochs nuScenes); image backbone frozen on the first four frames; 8 x A5000, ~3 days.
- differentiable: partial - matching for new elements; track labels inherited; memory gradient not detailed.
- sensors_data: camera, nuScenes / Argoverse2.
- results: nuScenes consistent GT 76.1 mAP / 69.1 C-mAP vs StreamMapNet 70.4 / 56.4 and MapTRv2 69.5 / 50.5; Argoverse2 76.9 / 68.3 vs 70.3 / 57.5. No missing-frame experiment reported. Stated limitation: merges and splits of elements are not handled by the tracking formulation.
- compute: 11.5 FPS on RTX 6000 (StreamMapNet 14.2, MapTRv2 12.5); extra time from un-batched loops in vector fusion.
- code: https://github.com/woodfrog/maptracker (GPL-3.0, built on StreamMapNet code; framework not named in README)
- fit_to_us: reuse_idea - effort S/M - take MaskBlend (learned fill for cells without history), distance-based strides instead of time strides, pose-noise augmentation of the warp; the vector tracking itself needs identities (we have none).
- compat_needs: ego pose; distance-based buffer of past inputs.
- compat_conflicts: element identities / bipartite matching; our graphs have forks and merges, which they explicitly do not handle.
- tried_before: none for strides / MaskBlend; map-offset and pose-jump failures exist in our synthetic set.
- verified: checked
