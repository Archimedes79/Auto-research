# TP28 - Challenge winners (2023 OpenLane Topology, 2024 Mapless Driving): what they report

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: partly (LGmap row mapping and MapVision step values inconsistent in the digest; no 2025 report exists). Pipeline stage: T B X.*

## Papers

The 1st-place Solution for CVPR 2023 OpenLane Topology in Autonomous Driving Challenge (arXiv 2306.09590) https://arxiv.org/abs/2306.09590 ; LGmap: Local-to-Global Mapping Network for Online Long-Range Vectorized HD Map Construction (arXiv 2406.13988; claims first place, Mapless Driving 2024) https://arxiv.org/abs/2406.13988 ; MapVision: CVPR 2024 Autonomous Grand Challenge Mapless Driving Tech Report (arXiv 2406.10125) https://arxiv.org/abs/2406.10125

## Idea

What winners do is mostly not a better link head: stronger detector (big backbone, long schedule, ensembling), better auxiliary detectors, SD-map fusion, temporal fusion, a topology head trained last on frozen detectors.

## Formulation

2023 (test OLS 0.55 vs 0.47 of second place): PETRv2 lane detector with 300 queries, 3D Bezier control points, backbones ResNet-50 / VoV / ViT-L, BDA + HSV augmentation; YOLOv8-x traffic detector (mosaic / mixup, foreground-class reweighting 2, resampling rare classes x5-20, pseudo-labels, multi-scale TTA 0.7-1.4, no horizontal flip); two MLP topology heads trained 10 epochs on frozen detectors; no ensembling. 2024 LGmap (test UniScore 0.66): symmetric view transformation (BEVFormer + LSS with lidar depth supervision, SD-map polylines by cross-attention), hierarchical temporal fusion (streaming then stacking; N frames sampled from M previous), ped-crossing resampling (4 corners + 6 points per edge, 8 instead of 40 permutations), TopoMLP-style lane-lane head, ViT-L pre-trained on nuScenes map construction, 36 + 48 epochs, input 1536 x 1536, BEV 100 x 200, ensemble of three lane models (ViT and InternImage-XL at scales 0.5 / 0.75 / 1) plus YOLOv8 + YOLOv9 for traffic elements. 2024 MapVision (test OLUS 0.58): BEVFormer + SD map (OSM) encoder after SMERF with auto-encoder pre-training, InternImage-L, LDTR head, P2P IoU loss, YOLOX for traffic elements, topology fine-tuned (lr 4e-4) with the rest frozen, decoupled area head.

## Input

multi-view images (+ SD map in 2024).

## Output

lane segments / centerlines + topology.

## Training

as above.

## Differentiable?

partial - Hungarian host models; ensembling is outside the network.

## Sensors and data

camera, SD map; OpenLane-V2 (2023 subset A / B; 2024 lane segment subset A + traffic elements).

## Reported results

2023 validation: lane backbone DET_l ResNet-50 18.15, VoV 21.01, ViT-L 28.11 (20 ep) -> 34.16 (48 ep) -> 35.28 with BDA; traffic DET_t 65.32 -> 79.89 (aug +3.77, TTA to 79.89); topology Table 4 (old metric): DET_l 28.11 / TOP_ll 14.54 -> DET_l 35.28 / TOP_ll 23.01 (topology tracks detector quality). 2024 LGmap: BEVFormer 40.36 mAP, LSS 32.57, both 40.89, +augmentation 43.75; temporal none 52.93, streaming 56.61, streaming-stacking 57.13; lane-segment ensemble (test, row mapping ambiguous): single 48.67 / 40.04 / 48.27 (DET-l / TOP-ll / TOP-lt) -> three models 50.74 / 46.32 / 53.59. MapVision cumulative OLUS (validation): ResNet-50 0.36, +map encoder 0.40, +InternImage-L 0.45, +map pre-train 0.46, +LDTR head 0.48, +P2P IoU loss 0.50; ensemble step: +YOLOX 0.57, +topology fine-tune 0.58, +decoupled training 0.59; test OLUS 0.58 (DET_l 0.39, DET_a 0.40, DET_t 0.80, TOP_ll 0.38, TOP_lt 0.48).

## Compute

16 A800 (LGmap), no inference numbers.

## Code

none for LGmap / MapVision; the 2023 solution is TopoMLP (https://github.com/wudongming97/TopoMLP)

## Fit to our project

reuse_idea - effort S - recipe items that apply to a frozen-foundation setting: train the link / verifier head last on frozen detections; ensemble of models; temporal streaming-then-stacking; resampling of rare classes (forks / merges).

## Needs

multiple detectors or checkpoints for ensembling.

## Conflicts

ViT-L / InternImage backbones and 16-GPU schedules are out of reach; most gains come from the foundation, which we already have.

## Tried in our experiments

E27 (foundation choice) map-pretrained kept.
