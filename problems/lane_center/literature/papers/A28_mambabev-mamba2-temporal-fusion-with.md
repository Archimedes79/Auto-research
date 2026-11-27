# A28 - MambaBEV: Mamba2 temporal fusion with four-direction scan

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: MambaBEV: An efficient 3D detection model with Mamba2 (2024) https://arxiv.org/abs/2410.12673
- depth: M
- stage: S8
- idea: TemporalMamba block: BEV serialised in four scan directions through Mamba2 and averaged; past BEV features aligned by ego rotation angle only (as described); 3 history frames (tiny) / 4 (base).
- formulation: conv fusion beats concatenation by about 1 point; kernel 3 beats 5; four-direction scan beats one direction by 1.02 % NDS / 1.28 % mAP.
- results: nuScenes val base 51.7 NDS / 42.7 mAP; claims +3.51 mAP and +5.97 NDS over BEVFormer-S and a 37 % drop in velocity error; tiny-model numbers inconsistent across tables.
- compute: no latency; limitations section says still a bit more expensive than deformable attention.
- code: not checked
- training: AdamW 2e-4, batch 1 per GPU, no augmentation.
- differentiable: yes.
- sensors_data: camera, nuScenes.
- fit_to_us: baseline - effort M - like A12: SSM is not shown to be faster than a GRU at 3 - 4 frames. Not an early candidate.
- compat_needs: grid features.
- compat_conflicts: none.
- tried_before: none.
- verified: partly
