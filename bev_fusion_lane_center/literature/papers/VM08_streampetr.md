# VM08 - StreamPETR (object-centric memory queue, motion-aware layer norm)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: S8.*

## Papers

Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection (Wang et al., ICCV 2023; authors / venue from memory) https://arxiv.org/abs/2303.11926

## Idea

A FIFO memory queue keeps the top-K foreground queries of the last N frames; current queries read them by "hybrid attention" (self-attention over current + memory queries) and PETR-style cross-attention to image tokens. Ego-motion / velocity / time gap enter via Motion-aware Layer Normalisation (MLN): affine parameters (gamma, beta) computed from (ego pose E, velocity v, dt) modulate the position encoding of the ego-compensated centre and the context embedding. No BEV warping.

## Formulation

memory N x K = 4 frames x 256 objects; entry = (dt, context embedding Q_c, 3D centre Q_p, velocity v, ego pose E). E_{t-1}^t = E_t^-1 E_{t-1}; Q_p~ = E_{t-1}^t Q_p; gamma = xi_1(M), beta = xi_2(M); Q_pe~ = gamma LN(psi(Q_p~)) + beta, Q_c~ = gamma LN(Q_c) + beta; current frame v = 0, dt = 0. Explicit compensation alone did not help (attributed to early error propagation). 644 fresh + 256 propagated queries = 900 (lightweight 300 + 128). Training: local sliding windows of 8 key frames, one random frame skipped, gradients of the first 6 frames detached (losses on the last 2); AdamW 4e-4, batch 16, 24 / 60 epochs. Detection loss / matching not on the page (denoising from PETRv2).

## Input

image tokens with 3D position encoding + memory queue.

## Output

3D boxes (queries).

## Training

hungarian_set (from memory).

## Differentiable?

partial - TopK + gather, earlier-frame gradients detached.

## Sensors and data

camera; nuScenes.

## Reported results

nuScenes val R101 512 x 1408: mAP 0.504, NDS 0.592, 6.4 FPS; R50 256 x 704: 0.432 / 0.540, 27.1 FPS; test V2-99: 0.550 / 0.636, AMOTA 0.653.

## Compute

27.1 FPS (R50); flash attention cuts memory 61 -> 27 GB at 1600 x 640 (A100).

## Code

https://github.com/exiawsh/StreamPETR

## Fit to our project

reuse_idea - S - MLN (pose-conditioned affine modulation of carried tokens) is a cleaner alternative to concatenating a flattened 4x4 pose; the FIFO of top-K tokens with (dt, pose) per entry is the template for a multi-frame recursion memory.

## Needs

tokens with an explicit centre; per-entry timestamp and pose. Cached-BEV: only the memory logic (the model reads image tokens).

## Conflicts

detection queries (boxes), not polylines.

## Tried in our experiments

none.
