# TP11 - Lane Field (dense BEV direction-tensor field as conditioning and sampling guide)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: partly (last 2.3 k characters of the page unread; matching not described). Pipeline stage: S2 B X.*

## Papers

Learning Lane-Structured Conditioning Fields for Online HD Mapping (arXiv 2610.25284, 18 Oct 2026; authors not seen) https://arxiv.org/abs/2610.25284

## Idea

A dense BEV field R (H x W x 5) with a SIGN-FREE direction tensor M = sum_i I_i t_i t_i^T (xx, xy, yy), a proximity P and a confidence C, supervised from analytic ground truth. It (a) modulates the BEV features SPADE-style and (b) stretches the tangential component of deformable sampling offsets along the local lane direction. Confidence C = anisotropy of the tensor is low where lane directions cross, so guidance is switched off at intersections.

## Formulation

per segment i (endpoints A_i, B_i, tangent t_i, mean width w_i): I_i(p) = (sigma_i sqrt(2 pi)/2) exp(-d_perp^2 / (2 sigma_i^2)) [erf(u1) - erf(u0)], sigma_i = 0.5 w_i; M = sum_i I_i t_i t_i^T; P = sum_i I_i; C = Delta / tr, Delta = sqrt((Mxx - Myy)^2 + 4 Mxy^2). Targets: P_gt = P / (P + 1), M_gt = M / (tr M + eps) * sigmoid(k (P_gt - 0.3)), k = 20. FiLM: F_out = gamma * F_in + beta with gamma, beta from convs on the field. Guidance: s_new = r + [1 + g_C g_U g_H (Delta_par - 1)] s_par t^ + s_perp n^ with Delta_par = 4.0 and sigmoid gates g_C = sigmoid(10 (C - 0.3)), g_U (uncertainty), g_H (per head). Loss 5.0 Smooth-L1 on tensor + 2.0 L1 on finite differences + 1.0 Laplacian beta-NLL on proximity (beta 0.5). Matching: not described.

## Input

BEV features (plug-in), field head on the same grid.

## Output

field tensor; modulated features; guided sampling offsets.

## Training

with the host model's loss; OpenLane-V2 24 epochs; AV2 / nuScenes with MapTRv2 (24 epochs nuScenes, +-30 x +-15 m).

## Differentiable?

yes - the whole field path; ablation shows detaching it from the BEV encoder loses accuracy.

## Sensors and data

camera; OpenLane-V2 subset A (BEV 100 x 200, +-50 x +-25 m), Argoverse 2, nuScenes; SD-map variant with fixed sigma.

## Reported results

LaneSegNet host (mAP / AP_ls / AP_ped / TOP_lsls): baseline 32.1 / 32.2 / 31.9 / 25.5; +guidance 33.2 / 33.4 / 32.9 / 26.8; +modulation 34.9 / 35.7 / 34.0 / 27.9; both 36.1 / 35.8 / 36.4 / 28.4; with SD map 36.8 / 36.6 / 37.0 / 29.7. Gradient flow (mAP): stop-gradient in and out 34.1, fully attached 36.1. Plug-ins: MapTRv2 nuScenes 55.8 -> 57.7 mAP; TopoPoint centerline TOP_ll 30.0 / OLS 43.8 (its own table).

## Compute

+7.2 M parameters (4.8 head, 2.4 SPADE); LaneSegNet 14.6 -> 13.9 FPS, 1.3 -> 1.5 TFLOPs.

## Code

project page https://andberto.github.io/LaneField/ (no repository link)

## Fit to our project

reuse_idea - effort M - our dense head already predicts heat map + direction (2 ch); the tensor form removes the sign ambiguity of a direction vector, gives a free junction-confidence channel and can condition the cached features before the trace network; the tangent-stretched sampling is the same idea as our long, narrow forward ray.

## Needs

a dense head on the BEV grid and supervision from piece geometry (analytic, no matching).

## Conflicts

the reported gain needs the field gradient to reach the BEV encoder (frozen in our setting, so only the modulation of the cached features is possible); uses width annotations (we have no lane width in the centreline labels except the map expansion polygons).

## Tried in our experiments

none for the tensor form (dense head heat + direction exists; E44 key points).
