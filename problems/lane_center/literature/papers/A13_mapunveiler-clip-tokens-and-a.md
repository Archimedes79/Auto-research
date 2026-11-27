# A13 - MapUnveiler: clip tokens and a Token-Turing-Machine-style memory

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: Unveiling the Hidden: Online Vectorized HD Map Construction with Clip-Level Token Interaction and Propagation (Kim, Seong, Ji, Jang, NeurIPS 2024) https://arxiv.org/abs/2411.11002
- depth: M
- stage: S8
- idea: Per clip (T = 3 frames, stride S = 2): 50 compact clip tokens are made from the frame-level map queries by cross-attention; the BEV is updated by cross-attending to the clip tokens (BEV queries, token keys) instead of dense spatio-temporal attention; the map generator decodes with deformable attention. An inter-clip memory of M = 96 tokens is read and written in Token-Turing-Machine style; the first clip uses learned embeddings.
- formulation: memory 96 or more similar, smaller degrades; clip tokens 50 best (25: 69.2, 100+ worse); read: memory + map queries 69.8 vs memory only 68.9; write: clip tokens 69.1, + map tokens 69.8, adding BEV 68.9; frozen frame-level MapNet 66.7 vs end-to-end 69.8; T = 5, S = 3 gives 70.1 but over 40 GB. Ego-pose usage and cross-clip gradient flow: not found.
- input: BEV features from MapTRv2 backbone; output: vector map.
- training: built on MapTRv2, frame-level pre-train then main; 8 x A100, batch 16; the model runs clip inference three times per step and applies the loss after each.
- differentiable: partial (Hungarian host; gradient across clips not stated).
- sensors_data: camera, nuScenes / AV2.
- results: nuScenes val 68.0 (60x30) / 68.7 (100x50) vs MapTRv2 61.5 / 58.6; AV2 70.5 / 65.4; heavy-occlusion split 63.8 vs StreamMapNet 53.1 and MapTRv2 47.1; geo-disjoint 39.4 vs 36.6. (Slide listing quotes different occlusion gains; the digest used the proceedings figure.)
- compute: 12.7 FPS vs MapTRv2 15.6 (A100); peak GPU memory about 2x and parameters about 3x of MapTRv2.
- code: no repository on the project page (https://mapunveiler.github.io)
- fit_to_us: reuse_idea - effort M/L - a fixed-size learned memory (96 tokens, read / write) is the best published answer to long-horizon memory with forgetting; heavy for our latency budget.
- compat_needs: token interface; our vertex tokens (128-d) could be the clip tokens.
- compat_conflicts: cost (2x memory, 3x params).
- tried_before: none.
- verified: partly
