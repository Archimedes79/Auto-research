# SMIFORMER SMIFormer: multi-view interactive transformer for 4D radar voxels (Sensors 2023) - ATTENTION core

| | |
|---|---|
| source | https://pmc.ncbi.nlm.nih.gov/articles/PMC10708838/ (Sensors 2023, MDPI; no arXiv id used) |
| fact status | checked (PMC page read, 2026-10-25) |
| groups | A1: 2 |
| code / licence | none given (built on OpenPCDet); article CC BY 4.0 |
| verdict | context (the best attention radar-only 4D encoder with a measured latency, 61 ms - too slow) |

**What it does**: 4D radar points are voxelised (0.05 x 0.05 x 0.125 m, 51.2 x 51.2 x 5 m), projected to BEV, front-view and side-view planes by sparse dimension compression; windows of size 4 in each view get intra-view self-attention (SAI) with a learned 2D position embedding and inter-view cross-attention (CAO) between windows that cover the same 3D block; each voxel feature = own feature + features sampled from the three planes; CenterPoint dense head. Layer counts and widths not reported.

**Results that matter** (VoD val, 5 scans, mAP entire area / corridor; checked): PointPillars 45.41 / 67.32; CenterPoint 45.42 / 65.06; RCFusion (radar + camera) 49.65 / 69.23; SMIFormer 48.77 / 71.13. Ablation from a VoxelNet base 46.48 / 68.36: + view decoupling 47.43 / 68.65; + self-attention 47.59 / 70.00; + cross-view attention (full) 48.77 / 71.13. So attention adds +1.3 entire area / +2.5 corridor on top of the same 3-plane voxel encoder. Window size 2 / 4 / 8: 48.41 / 48.77 / 46.51. Latency 61 ms for window 4 (90 ms without splitting), GPU not named; parameters and FPS not reported.

**Transferable technique**: windowed attention gets +1.3 to +2.5 mAP for a latency of tens of ms on a dense 3D voxel representation; compared with RadarPillars, +2.4 at 5 ms and 0.27 M parameters, the cost per point of mAP is much higher.

**Cost and fit**: 61 ms alone exceeds our whole few-ms budget for the radar branch.

**Doubts**: baselines are re-trained by the authors (PointPillars 45.4 here vs 39.5 in RadarPillars on the same split - the baselines are not comparable across papers); no direct pillar-vs-transformer ablation; no code; GPU not named.
