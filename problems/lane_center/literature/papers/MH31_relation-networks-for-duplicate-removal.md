# MH31 - Relation Networks for duplicate removal (learned NMS from rank embedding + relation module)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, S5.*

## Papers

Relation Networks for Object Detection (Hu, Gu, Zhang, Dai, Wei, CVPR 2018) https://arxiv.org/abs/1711.11575 (read via ar5iv)

## Idea

A relation module lets each detection attend to all others with appearance and geometry weights; the duplicate-removal network takes score rank, box and feature per detection and outputs s1 in [0,1]; final score = s0 * s1; trained end to end.

## Formulation

f_R(n) = sum_m w^{mn} (W_V f_A^m); w^{mn} = w_G^{mn} exp(w_A^{mn}) / sum_k w_G^{kn} exp(w_A^{kn}); w_A = dot(W_K f_A^m, W_Q f_A^n) / sqrt(d_k); w_G = max(0, W_G E_G(f_G^m, f_G^n)) with a sinusoidal embedding of the log-ratio box offsets; residual update; N_r = 16, d_k = 64, d_g = 64. Duplicate net: rank (1..N) embedded to 128-d (works better than the raw score) + projected feature, sigmoid output. Label: per GT, detections with IoU >= eta; the highest-scoring is 1, others 0; eta in {0.5, ..., 0.9}, one output per threshold, averaged; loss BCE on s0 * s1; labels change during training because they depend on the detector's current scores. - input / output: set of detections with features and boxes -> rescored set.

## Training

equal-weight sum of RPN, recognition and duplicate losses; regime `matching_free` (argmax per GT among current detections).

## Differentiable?

yes (gradient flows to head and backbone; labels constant).

## Sensors and data

COCO (Faster R-CNN R50).

## Reported results

instance recognition 29.6 -> 31.9 mAP; duplicate removal: NMS 29.6, SoftNMS 30.2, learned 30.5 separate / 31.0 end to end.

## Compute

duplicate net about 0.33 M parameters, 0.3 GFLOPs, about 2 ms (N = 100, Titan X); a full relation module about 3 M parameters.

## Code

not checked

## Fit to our project

reimplement - effort S - sibling / ancestor attention with rank embedding and "best one per GT is positive, others duplicates" is our stage C label, matching-free.

## Needs

scores sorted to ranks; geometry of the set.

## Conflicts

label depends on current scores.

## Tried in our experiments

none
