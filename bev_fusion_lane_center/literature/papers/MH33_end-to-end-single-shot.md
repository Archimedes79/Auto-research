# MH33 - End-to-end single-shot detector with graph-based learnable duplicate removal (Ding et al.)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **A** (A abstract, M method, C code page). Verified: partly. Pipeline stage: S4, S5.*

## Papers

End-to-End Single Shot Detector using Graph-based Learnable Duplicate Removal (Ding, Rehder, Schneider, Cordts, Gall, GCPR 2022) https://pages.iai.uni-bonn.de/gall_juergen/download/nonmaximumsuppression_gcpr22.pdf (listing and abstract only; PDF text not decodable)

## Idea

A GNN over detections refines scores and boxes from image features, trained end to end with an SSD detector; a pre-filtering head cuts the number of proposals before the graph.

## Formulation

not read. - input / output: SSD proposals -> rescored detections.

## Training

end to end with the detector (abstract).

## Differentiable?

yes (abstract).

## Sensors and data

MS-COCO, KITTI.

## Reported results

"improves accuracy compared with other duplicate removal methods at significantly lower inference time" (abstract); no numbers seen.

## Compute

not stated.

## Code

none found

## Fit to our project

reuse_idea - effort S - same role as MH31; pre-filter + graph rescoring maps onto "take the best 3 x M candidates, then a GNN over siblings".

## Needs

not read.

## Conflicts

not read.

## Tried in our experiments

none
