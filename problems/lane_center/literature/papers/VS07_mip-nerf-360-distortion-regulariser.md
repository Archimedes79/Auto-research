# VS07 - Mip-NeRF 360 distortion regulariser (compact ray weights)

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L, S3.*

## Papers

Mip-NeRF 360 (Barron et al., CVPR 2022) https://arxiv.org/abs/2111.12077

## Idea

A penalty on the ray weights w that pulls their mass into one short interval, so the endpoint distribution along a ray stays compact; plus a proposal-network loss that upper-bounds the fine weights with the coarse ones under stop-gradient.

## Formulation

L_dist(s, w) = sum_{i,j} w_i w_j |m_i - m_j| + (1/3) sum_i w_i^2 (s_{i+1} - s_i), m_i = interval midpoint in NORMALISED ray distance s (metric t "significantly upweights distant intervals"); weight lambda = 0.01. Effects named by the authors: shrink each interval, pull distant intervals together, consolidate weight into one or a few neighbouring intervals, drive weights to zero where possible. L_prop = sum_i (1 / w_i) max(0, w_i - bound(t^, w^, T_i))^2, bound = sum of proposal weights overlapping T_i, asymmetric, stop-gradient on the NeRF outputs. The paper's w_i equals (1 - exp(-tau_i (t_{i+1} - t_i))) * exp(-sum_{i'<i} tau_{i'} (t_{i'+1} - t_{i'})) (Eq. 4).

## Input

weights and interval borders along a ray. output: scalar loss.

## Training

regulariser (self_supervised context).

## Differentiable?

yes.

## Sensors and data

images.

## Reported results

without L_dist "floaters" remain (qualitative figure); metrics change little.

## Compute

O(K^2) per ray, trivial for K = 5.

## Code

URL from memory, not opened: https://github.com/google-research/multinerf

## Fit to our project

reuse_idea - S - regulariser on P(k) against bimodal endpoint laws (the "hop to the neighbouring lane" failure); needs the metric distance normalised to 24 m on our non-uniform rings.

## Needs

P(k) as weights; normalised distances.

## Conflicts

penalises genuine bimodality (a fork inside one ray); only sensible if forks are covered by separate rays.

## Tried in our experiments

none.
