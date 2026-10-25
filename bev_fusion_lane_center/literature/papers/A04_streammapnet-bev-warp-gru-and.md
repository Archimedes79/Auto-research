# A04 - StreamMapNet: BEV warp + GRU and query propagation

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction (Yuan, Liu, Wang, Zhao, WACV 2024) https://arxiv.org/abs/2308.12570
- depth: M+C(framework only)
- stage: S8
- idea: Two memories. (1) BEV memory: previous BEV feature warped by the 4x4 ego transform T and fused with the current one by a GRU + LayerNorm. (2) Query memory: top k = 33 of N_q = 100 queries by confidence are kept; propagated by Q_t = MLP(concat(Q_{t-1}, flatten(T))) + Q_{t-1}; their polylines are transformed by T and used as reference points; concatenated with the remaining new queries.
- formulation: loss = 50 L_line (Smooth L1 over 20 points, min over permutations) + 5 L_focal + 5 L_trans (aux loss supervising the transformed queries). Gradients through memory do not flow back to previous frames; each training sequence is randomly split in 2 parts per epoch; first 4 epochs single-frame.
- input: BEV feature (from images), previous BEV and queries, ego pose T.
- output: vectorised map polylines (set, matching based).
- training: Hungarian / permutation matching, 8 x 3090, batch 32, AdamW 5e-4.
- differentiable: partial - memory detached across frames; Hungarian in loss.
- sensors_data: camera, nuScenes + Argoverse2; 60x30 and 100x50 m.
- results: new (geographically disjoint) nuScenes split mAP: 30 m 33.9 (vs MapTR 20.9), 50 m 23.0 (vs 14.8); 14.2 FPS (Argoverse2 setting); GPU memory / latency numbers not stated.
- compute: 13.2 FPS (nuScenes table), 14.2 FPS (abstract setting).
- code: https://github.com/yuantianyuan01/StreamMapNet (README: OpenMMLab stack mmcv-full 1.6.0, mmdet 2.28.2, mmdet3d 1.0.0rc6; GPL-3.0)
- fit_to_us: reuse_idea - effort S - GRU on warped BEV and a transformed-query MLP are small; "detach memory, single-frame warm-up" is the standard recipe our time recursion already follows.
- compat_needs: ego transform; confidence ranking.
- compat_conflicts: their code needs mmcv / mmdet3d (not in our env).
- tried_before: E41 temporal feature fusion (+0.008 F1); loop 16 one-step recursion.
- verified: checked
