# A12 - MambaMap: BEV and instance-query fusion with a gated (diagonal) state-space block

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: MambaMap: Online Vectorized HD Map Construction using State Space Model (Yang et al., IROS 2025) https://arxiv.org/abs/2507.20224
- depth: M
- stage: S8
- idea: Memory bank of the last N = 4 refined BEV features and instance queries. BEV Mamba Fusion: warp history by 4x4 affine, concat, conv + LayerNorm, scan in four directions through a gated SSM block (GELU gate, expansion 4, DSS branch with reduction 0.5), average the four scans. Instance Mamba Fusion: historical queries aligned to current ones by L2-distance Hungarian matching, two sequences (instance-first, temporal-first) through gated SSM blocks. The authors note the SSM is DSS, "inspired by" Mamba.
- formulation: L_map = 5 L_pts + 50 L_cls(as stated; weights differ in orientation across papers). Memory features detached; two-stage training (single frame, then sequences). Ablation 60x30 nuScenes: baseline 61.5, + BMF 65.2, + IMF 63.7, both 67.3; SSM variant: S4 64.6, Mamba 65.3, Mamba-2 65.2, DSS 67.3; N = 4 best (67.3), N = 6 66.5.
- input: BEV features + queries; output: vector map.
- training: AdamW 2.5e-4, batch 8, 2 x A30.
- differentiable: partial (Hungarian in query alignment and loss; memory detached).
- sensors_data: camera, nuScenes / AV2.
- results: nuScenes 60x30 67.3 mAP (StreamMapNet 63.4), 100x50 66.5 (63.5); AV2 64.9 (61.1); geo-disjoint nuScenes 40.1 (MemFusionMap 38.0, StreamMapNet 34.1). Paper text claims a 3.1 mAP minimum gain on the disjoint split; table gives 2.1 over the strongest baseline.
- compute: 12.7 FPS vs StreamMapNet 14.9 (nuScenes), 13.6 vs 15.9 (AV2): slower; no FLOPs / params stated.
- code: "forthcoming" per search summary; not checked
- fit_to_us: baseline / reuse_idea - effort M - evidence that an SSM memory is not faster than GRU at 4 frames; the linear-complexity argument pays only for long sequences. Not recommended as first move.
- compat_needs: grid-shaped features (fine for 200x200).
- compat_conflicts: query alignment uses Hungarian.
- tried_before: none.
- verified: partly
