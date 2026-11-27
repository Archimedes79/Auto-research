# MH30 - One-to-many assignment plus NMS beats Hungarian (DETA) and one-to-several lane detection (O2SFormer)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **A** (A abstract, M method, C code page). Verified: partly. Pipeline stage: L.*

## Papers

NMS Strikes Back (Ouyang-Zhang, Cho, Zhou, Krahenbuhl, 2022) https://arxiv.org/abs/2212.06137 (abstract); End-to-End Lane detection with One-to-Several Transformer (Zhou, Zhou, 2023) https://arxiv.org/abs/2305.00675 (search summary)

## Idea

DETA: IoU-based one-to-many assignment with NMS outperforms one-to-one Hungarian matching in the same transformer. O2SFormer: lane detection mixing one-to-one and one-to-several assignment with a layer-wise soft label and dynamic anchor-based positional queries.

## Formulation

not read. - input / output: not read (lane anchors in O2SFormer).

## Training

not read.

## Differentiable?

not read.

## Sensors and data

COCO (DETA); CULane (O2SFormer).

## Reported results

DETA up to +2.5 mAP over one-to-one, Deformable-DETR 50.2 mAP at 12 epochs (abstract); O2SFormer R50 77.83 % F1 on CULane, converges 12.5x faster than DETR with R18 (search summary).

## Compute

not stated.

## Code

https://github.com/zkyseu/O2SFormer (named on a listing page, not opened)

## Fit to our project

baseline - effort S - supports keeping dense positives + suppression as the strong baseline against any learned selection.

## Needs

NMS at inference.

## Conflicts

NMS is not differentiable.

## Tried in our experiments

none
