# A32 - Online lane graph extraction from video: early temporal aggregation in BEV

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: Online Lane Graph Extraction from Onboard Video (Can, Liniger, Paudel, Van Gool, 2023) https://arxiv.org/abs/2304.00930
- depth: M
- stage: S8, S6
- idea: Compared three ways to use several frames for a lane graph: merging per-frame graphs (post-merge), a transformer over spatio-temporal embeddings, and early aggregation (proposed): warp each frame's features to BEV by homography, mask, align by odometry, aggregate with a max operator after a residual block per frame. Detection and connectivity head: DETR-style, Bezier control points + existence + association features.
- formulation: max beats mean; training with 3 frames (1 past, 1 future), test spacing 2 s; the model accepts any number of frames including one.
- results: nuScenes M-F / detection / C-F: 1 frame 58.3 / 60.0 / 61.0, 3 frames 58.8 / 60.3 / 61.3; post-merge 57.3 / 59.9 / 56.0; Argoverse 3 frames 58.2 / 62.2 / 58.1. Connectivity benefits from future frames only if at least one past frame is present.
- compute: about 10 FPS (1 frame) vs 3 FPS (3 frames).
- code: "will be made public" (no link found)
- training: matched DETR set.
- differentiable: partial (matching).
- sensors_data: camera, nuScenes + Argoverse.
- fit_to_us: reuse_idea - effort S - evidence that early aggregation of aligned dense features beats merging graphs (our E41 +0.008 F1 is consistent with the small gains); max aggregation across frames is the simplest operator and matches our hold's "higher heat wins".
- compat_needs: aligned BEV.
- compat_conflicts: future frames are not available online.
- tried_before: E39 / E41.
- verified: partly
