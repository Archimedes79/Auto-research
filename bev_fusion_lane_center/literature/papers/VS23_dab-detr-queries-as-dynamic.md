# VS23 - DAB-DETR: queries as dynamic 4-D anchors with width / height-modulated attention

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S2, S5.*

## Papers

DAB-DETR (Liu et al., ICLR 2022) https://arxiv.org/abs/2201.12329

## Idea

Each query IS an anchor (x, y, w, h) updated layer by layer; the positional attention is modulated by the ratio of a reference width / height (predicted from the content) to the anchor size, giving an anisotropic, size-aware window.

## Formulation

P_q = MLP(PE(A_q)), sinusoidal PE of all four coordinates, temperature T = 20 instead of 10000 (coordinates in [0,1]); cross-attention query concatenates content and PE(x_q, y_q) * MLP(C_q); modulated attention = (PE(x) PE(x_ref) w_ref / w + PE(y) PE(y_ref) h_ref / h) / sqrt(D), (w_ref, h_ref) = sigmoid(MLP(C_q)); shared head predicts (dx, dy, dw, dh) per layer (update space and detach not stated on the page seen); 300 queries.

## Input

queries (anchors) + feature map. output: boxes.

## Training

hungarian_set.

## Differentiable?

partial.

## Sensors and data

images (COCO).

## Reported results

R50 50 epochs AP: 42.2 (42.6 with 3 patterns), DC5 44.5 (45.7); DC5 ablation: no anchor update 44.0 (update adds 1.7), no w/h modulation 45.0, 2-D anchor points instead of 4-D boxes 44.0, no temperature tuning 44.4, full 45.7.

## Compute

not stated.

## Code

URL from memory, not opened: https://github.com/IDEA-Research/DAB-DETR

## Fit to our project

reuse_idea - S - a per-tip (length, width) gain that modulates the sampling window (spec: gain in [0.5, 1.5]); 4-D anchors beat 2-D points by 1.7 AP in their ablation.

## Needs

anchor with size. compat_conflicts: Hungarian.

## Tried in our experiments

none.
