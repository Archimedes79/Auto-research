# VS06 - NeRF volume rendering: alpha compositing with transmittance

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S3, X.*

## Papers

NeRF (Mildenhall et al., ECCV 2020) https://arxiv.org/abs/2003.08934

## Idea

A ray accumulates a per-sample quantity with weights w_i = T_i * alpha_i, where alpha_i is the chance that the ray is stopped inside interval i and T_i the chance that it was not stopped earlier. Hierarchical sampling re-draws samples where w is large.

## Formulation

C(r) = integral T(t) sigma(r(t)) c dt with T(t) = exp(-integral_{tn}^{t} sigma); discrete: C^ = sum_i T_i (1 - exp(-sigma_i delta_i)) c_i, T_i = exp(-sum_{j<i} sigma_j delta_j), delta_i = t_{i+1} - t_i (= prod_{j<i}(1 - alpha_j) with alpha_i = 1 - exp(-sigma_i delta_i); the paper says it "reduces to alpha compositing" but does not write the product). T is interpreted as the probability that the ray travels without hitting anything. Stratified sampling: N bins, one uniform draw per bin. Hierarchical: N_c = 64 coarse, N_f = 128 fine by inverse-transform sampling from the normalised w_i of the coarse pass; the fine pass uses all 192; loss = squared colour error of coarse + fine. The appendix quotes 64 + (64 + 128) at test time (differs from the training count; not reconciled). Near / far bounds are symbols only (synthetic scenes inside a cube of side 2; forward-facing scenes in NDC with far = infinity). The last delta and rays that leave the volume are not discussed.

## Input

density and attribute at sample points along a ray. output: rendered attribute, weights w_i.

## Training

self_supervised (photometric); no matching.

## Differentiable?

yes (the stratified draws are re-sampled each step).

## Sensors and data

multi-view RGB images.

## Reported results

none copied (rendering paper).

## Compute

4,096 rays per batch, 192 network calls per ray in the fine pass (training).

## Code

URL from memory, not opened: https://github.com/bmild/nerf

## Fit to our project

reuse_idea - S - map: alpha_i <-> hazard (1 - nu_i), T_i <-> survival, w_i <-> probability that the last verified sample is the one before i. Writing the per-sample probability as exp(-sigma_i delta_i) (hazard per metre times gap) keeps the model consistent when ring gaps differ (4 / 4 / 4 / 6 / 6 m) or are changed.

## Needs

a density along the ray; for us density = "lane absent".

## Conflicts

the remainder 1 - sum w_i (the ray never stopped) is ignored by the paper; for a censored ray it must be kept explicitly.

## Tried in our experiments

none.
