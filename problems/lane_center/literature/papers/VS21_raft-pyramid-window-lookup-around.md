# VS21 - RAFT: pyramid-window lookup around the current estimate + recurrent residual update (+ code facts)

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S2, S5.*

## Papers

RAFT (Teed, Deng, ECCV 2020) https://arxiv.org/abs/2003.12039 ; code files read https://github.com/princeton-vl/RAFT

## Idea

Keep one current estimate (flow); each iteration look up a local window of the correlation volume around the CURRENT estimate on every pyramid level, feed it to a ConvGRU, and add a residual. Weights are shared across iterations.

## Formulation

correlation volume = dot products of 1/8-resolution features (256 ch), 4-level pyramid by 2x2 average pooling (kernels 1, 2, 4, 8 in the paper); lookup radius r = 4, the paper defines the neighbourhood as an L1 ball but the code uses a SQUARE (2r + 1)^2 = 81-offset window per level, offsets applied in each level's own coordinates (centroid / 2^i), bilinear sampler, 4 x 81 = 324 channels; ConvGRU (separable 1x5 / 5x1) with correlation, flow and context inputs; f_{k+1} = f_k + Delta f; sequence loss sum_i gamma^{N-i} |f_gt - f_i|_1, gamma = 0.8; `coords1` is detached at the top of every iteration (gradient only through Delta f); 12 iterations in training (code default), 32 (Sintel) / 24 (KITTI) at test; hidden state from tanh of a context-network part, context input from relu of the rest; convex upsampling.

## Input

two feature maps (rasters) + current estimate. output: flow field (dense).

## Training

supervised sequence loss; matching_free.

## Differentiable?

yes with a stop-gradient on the estimate.

## Sensors and data

images (Sintel, KITTI).

## Reported results

Sintel train clean / final EPE 1.43 / 2.71; KITTI-15 train EPE 5.04, F1-all 17.4 %; inference updates (Sintel clean / final EPE): 1: 4.04 / 5.45, 3: 2.14 / 3.52, 8: 1.61 / 2.88, 32: 1.43 / 2.71, 100: 1.41 / 2.72, 200: 1.40 / 2.73 (no divergence).

## Compute

5.3 M parameters; 0.10 s with 10 updates on a GTX 1080 Ti.

## Code

checked (`corr.py`, `raft.py`).

## Fit to our project

reuse_idea - S - a pyramid-window lookup around the current tip estimate (r = 4 cells on f0 / f1 / f2 = 2 / 4 / 8 m half-width at 0.5 / 1 / 2 m cells; 3 x 81 = 243 reads per query) is a drop-in alternative to rings; weight-shared residual update with a gamma-weighted sequence loss is the matching-free multi-step loss.

## Needs

dense pyramid with bilinear lookup (we have it).

## Conflicts

a window lookup is isotropic; the teardrop is anisotropic by design.

## Tried in our experiments

E94 fan of 27 bilinear samples + 7 cross-section points (a fixed-pattern lookup); E120 refinement (+0.012 F1@1).
