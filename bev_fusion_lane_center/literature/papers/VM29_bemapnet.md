# VM29 - BeMapNet (piecewise Bezier representation, Point-Curve-Region loss)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: L, S5.*

## Papers

End-to-End Vectorized HD-map Construction with Piecewise Bezier Curve (Qiao et al., CVPR 2023) https://arxiv.org/abs/2306.09700

## Idea

Each element is a piecewise Bezier curve <k, n> (k segments of degree n, consecutive segments sharing an endpoint): divider <3,2>, crossing <1,1>, boundary <7,3>. The head predicts, per instance, control points and an end index (softmax over the maximum number of pieces) that sets the piece count. The loss supervises control points, restored curve points and a rasterised region.

## Formulation

GT fitting: sample m = 100 points, least squares C = B+ P with the Bernstein matrix, extend a segment while the Chamfer distance stays below epsilon; k+1 explicit points (on the curve) and nk - k implicit points as offsets from the segment centre. L = 5 L_point(L1 on control points) + 10 L_curve(L1 on the restored m points) + 1 L_region (Dice on dilated rasterisations, width 5); auxiliary BEV / instance mask CE + Dice (1, 5). IPM-PE Align: sin-cos encodings of image-grid points (flat-ground assumption) and BEV grid points aligned by one shared FC. Semantic BEV decoder 2 enc + 4 dec layers with 64 x 32 BEV queries; instance Bezier decoder 6 masked cross-attention layers with 60 instance queries (20 / 25 / 15). Range [30, 30, 15, 15] m, 0.15 m per pixel, input 896 x 512, AdamW 2e-4, 30 / 110 epochs, 8 GPUs. Start-point / direction ambiguity: not addressed; matching bipartite, cost not given.

## Input

images -> BEV.

## Output

control points + end index per instance (post-processing free).

## Training

set prediction with bipartite matching.

## Differentiable?

partial (matching).

## Sensors and data

camera; nuScenes 28,130 train / 6,019 val; thresholds {0.2, 0.5, 1.0} m.

## Reported results

nuScenes mAP: Swin-T 30 ep 43.7 (div 49.1, cross 42.2, bound 39.9); Swin-B 110 ep 50.5; at {0.5, 1.0, 1.5}: 67.7. Abstract: at least +18.0 mAP over prior best in the same setting.

## Compute

not stated here.

## Code

https://github.com/er-muyue/BeMapNet

## Fit to our project

reuse_idea - S - Bezier pieces with a predicted piece count and a curve-restoration loss (control points -> dense points -> L1) is a per-element loss that needs no point correspondence between prediction and GT other than index order; relevant to our Bezier connectors.

## Needs

piece-count head; curve sampler. Cached-BEV: the Bezier head and PCR loss yes.

## Conflicts

IPM-PE is camera-specific.

## Tried in our experiments

E53 Bezier connectors (split / merge F1 0.55 / 0.61 -> 0.71 / 0.64, kept).
