# A30 - CRT-Fusion: camera-radar recurrent temporal fusion guided by predicted motion

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: CRT-Fusion: Camera, Radar, Temporal Fusion Using Motion Information for 3D Object Detection (Kim, Seong, Choi, NeurIPS 2024) https://arxiv.org/abs/2411.03013
- depth: M
- stage: S8, B
- idea: Recurrent over N timestamps with a memory bank; a motion feature estimator predicts per-pixel BEV velocity and occupancy; features are shifted by velocity x dt where speed exceeds 1.0 m/s, concatenated with the next BEV and multiplied by the occupancy map, 1x1 conv. Ego motion compensation not described in the digest.
- training: 24 epochs, first 6 without the temporal module; loss weights depth 3, segmentation 25, velocity 1, occupancy 30; 4 x 3090.
- differentiable: yes.
- sensors_data: camera + radar, nuScenes.
- results: R50 val 57.2 NDS / 50.0 mAP vs CRN 56.0 / 49.0; night mAP 33.0 vs CRN 30.4 and BEVDepth 16.8; sunny 54.7 vs 54.8.
- compute: 14.5 FPS (R50), light 20.5 FPS, total 67.1 ms (light 48.8 ms).
- code: not checked
- fit_to_us: reuse_idea - effort - - only camera-radar temporal paper found; shows radar-camera temporal gives night gains in detection; the velocity-shift part is for moving objects, not relevant to static lanes.
- compat_needs: radar + camera BEV.
- compat_conflicts: moving-object motion model.
- tried_before: none.
- verified: partly
