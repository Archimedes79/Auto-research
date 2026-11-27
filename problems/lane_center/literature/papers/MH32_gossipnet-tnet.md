# MH32 - GossipNet / Tnet (learned NMS on boxes and scores only)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4.*

## Papers

Learning Non-Maximum Suppression (Hosang, Benenson, Schiele, CVPR 2017) https://arxiv.org/abs/1705.02950 (read via ar5iv); predecessor: A convnet for non-maximum suppression https://arxiv.org/abs/1511.06437 (read; Tnet)

## Idea

A pure NMS network takes only boxes and scores; each detection has a 128-d feature updated in blocks that pair it with neighbours of IoU > 0.2, apply pairwise FC layers, max-pool over neighbours and add residually (default 16 blocks); one score per detection.

## Formulation

labels: detections are matched to GT in descending confidence as in the benchmark, matched = positive, unmatched (incl. duplicates) = negative; weighted logistic loss; Tnet (predecessor) is a fully convolutional net on a 2-D grid fed with GreedyNMS score maps at several thresholds. - input / output: detection set -> per-detection keep score.

## Training

Adam 2 M iterations on COCO; not trained jointly with the detector.

## Differentiable?

yes inside the NMS net; not end to end with the detector.

## Sensors and data

PETS, COCO persons (8-block net).

## Reported results

COCO persons AP0.5:0.95 36.9 vs 35.6 (GreedyNMS > 0.5) on minival; PETS +4.8 AP overall, +7.3 on highly occluded; Tnet oMNIST AR 86.0 vs 60.2.

## Compute

about 14 ms per image for 67 detections on a K40m (unoptimised TensorFlow).

## Code

not checked

## Fit to our project

reuse_idea - effort S - a set-to-set rescorer working on geometry only, usable on the b x M candidates before the beam cut.

## Needs

neighbours by IoU analogue (distance).

## Conflicts

needs large data; gains modest (about 1 AP on COCO).

## Tried in our experiments

none
