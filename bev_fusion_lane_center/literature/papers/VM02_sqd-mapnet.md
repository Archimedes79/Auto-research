# VM02 - SQD-MapNet (stream query denoising)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: checked (parameters missing on the page are listed above).. Pipeline stage: T, L, S8.*

## Papers

Stream Query Denoising for Vectorized HD Map Construction (Wang et al., ECCV 2024) https://arxiv.org/abs/2401.09112

## Idea

Training-only add-on to StreamMapNet. The previous frame's GROUND TRUTH is mapped to the current frame by the ego transform, each element is wrapped in its bounding rectangle which is randomly shifted / rescaled, and the result is fed as extra "stream" queries whose targets are the current-frame GT instances found by "adaptive temporal matching" (ATM: bidirectional Chamfer between ego-warped previous GT and current GT with a per-instance threshold). The decoder practises turning a stale, shifted, partly out-of-range set into the current set, which its own early stream queries cannot teach. Removed at inference.

## Formulation

box shift |dx| < l1 w/2, |dy| < l2 h/2; scale w in [(1-l4) w, (1+l4) w], h in [(1-l3) h, (1+l3) h]; points move with the box, so no point jitter and no topology change. Noise decays with the matched Chamfer distance D: R = 1 - D / (gamma (w+h)/2); noised box = {x,y,w,h} + {dx,dy,dw,dh} * R; gamma = 0.2 best (0.1 / 0.3 / 0.5 / 0.7 worse). ATM threshold delta = alpha (w+h)/2, alpha = 0.1 best (0.05 / 0.2 / 0.3 worse; fixed thresholds 2.5 / 4 / 5 gave 62.0 - 62.8). L_DN = focal + line loss on the ATM-matched current GT; L = L_map + L_DN. Magnitudes l1 - l4, number of groups / queries and the attention mask are NOT stated on the page; no repository to check.

## Input

GT polylines of frame t-1 + pose (training only).

## Output

nothing extra at inference (model = StreamMapNet).

## Training

denoising branch assigned by ATM (a threshold rule), main branch keeps StreamMapNet's matching; regime denoising + hungarian_set.

## Differentiable?

yes for the DN branch; ATM is a label rule, so no gradient is needed there.

## Sensors and data

camera; nuScenes, AV2; 100 x 50 and 200 x 100 ranges (page writes "mm", read as metres); R50 + 1-layer BEVFormer, AdamW 5e-4, batch 32, 24 / 30 epochs.

## Reported results

nuScenes (old split, R50, 480 x 800, 24 ep, Table 1): StreamMapNet 60.4 -> 63.9 mAP (60 x 30 m), 60.6 -> 64.0 (100 x 50); with extra tricks 62.9 -> 65.0; V2-99, 110 ep: 74.0 / 75.2. AV2 (30 ep): 61.5 -> 63.3 (30 m), 57.7 -> 59.3 (50 m). Ablation (Table 3, cumulative, nuScenes 60 x 30 m): single-frame baseline 59.9; + StreamMapNet's temporal stream 59.2 (-0.7); + dynamic query noising 62.9; + ATM 63.9 (full model). Table 6: no denoising 59.2 (= the temporal model), normal denoising on current-frame GT 62.7, stream denoising 63.9. The paper explains the -0.7 by the network finding it hard to learn the constant change of curves between frames. New-split numbers only in the supplement (not seen).

## Compute

no inference cost; training cost not stated.

## Code

none found (paper: "will be available").

## Fit to our project

reuse_idea - S/M - give the time-recursion block (ego-warped previous set -> current set) a training source that does not depend on its own early outputs: warp the previous GT lane graph, perturb it, regress the current GT with our matching-free point-to-polyline loss. The noise decay rule and the ATM rule (which prior elements still have a counterpart) are cheap to copy; ATM gives the validity target of the prior.

## Needs

GT of frames t-1 and t plus ego transform in the data loader; a recursion block that accepts a prior set as input; per-element extent for the noise scale. Cached-BEV: yes (only the prior path is involved).

## Conflicts

noise acts on whole boxes: no per-point jitter, no dropped / duplicated / merged elements, no topology errors; noised GT is cleaner than real predictions (train-test mismatch only partly covered); built to run as extra queries beside matched queries with attention masking.

## Tried in our experiments

none.
