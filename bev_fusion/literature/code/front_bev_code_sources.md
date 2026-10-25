# Code sources for the front-camera + radar BEV plan (scan, 2026-10-25)

Plan: `../../experiment_execution/plans/front_bev_plan.md`. Level L0 only: GitHub pages and single raw files were read through a page fetcher (summaries of the page, not the full source); nothing was cloned or downloaded. Third-party code is not stored here. Our environment has no mmcv / mmdet / mmdet3d (PyTorch 2.8, Python 3.12), so "separable" means: the module can be re-written in plain PyTorch from the description, or its file can be read without a framework.

Marks: **C** checked = read on the repository page (URL in the row); **M** memory = from my knowledge, not seen on a page; "not shown" = the page did not display it (the page footer year is the site's, not a commit date, and is not used). Licence column: as stated on the page or in the LICENSE file read; "LICENSE empty / not named" is reported as found. For GPL, non-commercial, academic-only or no licence: read only, re-implement from a description.

## Recommendation per block (one line each)
| block | take | why | licence |
|---|---|---|---|
| 1 GT rasterisation | nuscenes-devkit `NuScenesMap.get_map_mask` (+ own box footprints, own FOV / occlusion ray cast); read BEVFusion `LoadBEVSegmentation` for the layer grouping; CVT label scheme for object visibility (M) | the only Apache-2.0 / MIT sources; LSS is non-commercial, HDMapNet is GPL-3 | Apache-2.0 / Apache-2.0 / MIT |
| 2 fixed lift | write our own sparse-matrix lift; read BEVDet `bev_pool_v2` and MatrixVT for the idea; Fast-BEV only as a timing reference | our table is column-polar and fixed, none of them has it | Apache-2.0 / MIT / not named |
| 3 Gaussian lift | GaussianLSS (MIT) as a reference for the variance-to-footprint rule; not its CUDA rasteriser | custom CUDA rasteriser, torch 2.1 / py 3.8 | MIT |
| 4 virtual camera | Gen-LaneNet `tools/utils.py` (homography functions, Apache-2.0) as a numpy reference for the homography spec test; BEV-LaneDet has no code | the homography H = K_v R K^-1 is 10 lines, the repo is a test oracle | Apache-2.0 |
| 5 radar encoding | RaCFormer (MIT) for radar depth + RCS embedding; CRN (MIT) for radar-aware depth / BEV generation scripts; CenterFusion (MIT) for pillar + frustum; RCBEVDet is academic-only | permissive code exists for three of seven | MIT |
| 6 temporal | FIERY (MIT) `fiery/layers/temporal.py` for warp + spatial GRU; BEVDet4D inside BEVDet (Apache-2.0); StreamPETR (Apache-2.0) for the memory queue; SOLOFusion (Apache-2.0) | FIERY is plain PyTorch, no mmdet | MIT / Apache-2.0 |
| 7 trunk + heads | CenterPoint (MIT) `det3d/models/bbox_heads/center_head.py` for targets and loss; FlashOCC (Apache-2.0) for the channel-to-height head | BEVerse licence not stated | MIT / Apache-2.0 |
| 8 loss balancing | nash-mtl `methods/weight_methods.py` class `Uncertainty` (MIT) or write it (4 lines); `lucidrains/gradnorm-pytorch` (MIT) if GradNorm is wanted | tiny | MIT |

## 1 BEV ground truth rasterisation (map layers, box footprints, FOV / visibility)
| repo | licence | framework | module (file / class) | status | marks |
|---|---|---|---|---|---|
| https://github.com/nutonomy/nuscenes-devkit | Apache-2.0, Motional 2021 (raw LICENSE.txt read) | plain Python, tested for Python 3.9 and 3.12 | `python-sdk/nuscenes/map_expansion/map_api.py`: classes `NuScenesMap`, `NuScenesMapExplorer`; methods `get_map_mask`, `render_map_mask`, `get_records_in_patch`, `extract_polygon`; default mask layers drivable_area, road_segment, road_block, lane, ped_crossing, walkway, stop_line, carpark_area, road_divider, lane_divider, traffic_light | devkit v1.2.0 on 2025-08-28 (README changelog); 240 commits | C (LICENSE.txt; GitHub map_api.py page; README) |
| https://github.com/mit-han-lab/bevfusion | Apache-2.0 (sidebar) | mmdetection3d fork (mmcv 1.4.0, mmdet 2.20.0, forked before the mmdet3d coordinate refactor) | `mmdet3d/datasets/pipelines/loading.py` class `LoadBEVSegmentation`: classes map to layers (`drivable_area` to road_segment + lane, `divider` to road_divider + lane_divider), canvas from the x / y bounds, calls `NuScenesMap.get_map_mask` with the lidar-to-global yaw; no visibility filter | archived by the owner on 2025-01-12, read-only; 51 commits | C (repo page; raw loading.py) |
| https://github.com/Tsinghua-MARS-Lab/HDMapNet | GPL-3.0 (sidebar) | plain PyTorch (+ pytorch-scatter) | `data/rasterize.py` (`mask_for_lines`, `line_geom_to_mask`, `preprocess_map` with instance and direction masks, `rasterize_map`; cv2 polylines at a given thickness; no distance transform), `data/vector_map.py`, `data/dataset.py` | 13 commits, no dates shown | C (repo page; raw rasterize.py) |
| https://github.com/bradyz/cross_view_transformers | MIT (README + LICENSE) | PyTorch + Lightning + Hydra | label generation `scripts/generate_data.py` (the doc names `scripts/generate_labels.py`: names differ, not resolved), data code `cross_view_transformer/data/nuscenes_dataset.py`, `nuscenes_dataset_generated.py`; pre-generated labels 361 MB offered; about 30 min per 1000 episodes; layers, visibility threshold and 0.5 m resolution not on the page read | 5 commits, no dates shown | C (repo page, tree, docs/label_generation.md); visibility tokens as in the CVT paper: M |
| https://github.com/aharley/simple_bev | MIT (page + LICENSE) | PyTorch 1.12 | `nets/segnet.py` (`Segnet`, bilinear unprojection via `vox_util.unproject_image_to_mem`, radar occupancy concatenated after folding Y into channels, `--use_radar`, `--use_metaradar`, `--nsweeps=5`), `utils/vox.py`, `utils/geom.py` (files exist; contents of vox.py not read) | 92 commits, no dates shown; single frame, no temporal | C (repo page, tree, segnet.py) |
| https://github.com/nv-tlabs/lift-splat-shoot | NVIDIA Source Code License for Lift, Splat, Shoot: research / evaluation use only (LICENSE read) | PyTorch | `src/data.py` `NuscData.get_binimg` (vehicle box bottom footprints, ego frame at the lidar pose, `cv2.fillPoly`; no map layers, no visibility filter), `src/tools.py` (`gen_dx_bx`, `get_nusc_maps`, `plot_nusc_map`) | 32 commits, no dates shown | C (LICENSE, raw data.py, raw tools.py) |

Notes: no repo here builds an occlusion mask by ray casting through box footprints; that part (plan section 9) is ours. For box footprints at an interpolated time nothing is available either.

## 2 Fixed lookup-table lift / depth-bin pooling
| repo | licence | framework | module | status | marks |
|---|---|---|---|---|---|
| https://github.com/HuangJunJie2017/BEVDet (branch dev3.0) | Apache-2.0 (sidebar) | mmdet3d fork (folder `mmdet3d`; version not stated) | `mmdet3d/ops/bev_pool_v2/` (CUDA op, directory exists); `mmdet3d/models/necks/view_transformer.py` (LSS-type transformer; file exists, class names not read: M `LSSViewTransformer`) ; `lss_fpn.py` | README news last entry 2024-07-01; 1,105 commits | C (repo page, ops and necks tree); class names M |
| https://github.com/Sense-GVT/Fast-BEV (branch dev) | not named: a LICENSE file is linked, raw URLs `LICENSE` and `LICENSE.md` returned 404 | mmcv-full 1.4.0, mmdet 2.14.0, mmseg 0.14.1, torch >= 1.8.1; an `mmdet3d` folder | `script/view_tranform_cuda/` (README, `check.ipynb`, `cpp/`, `input/`): CUDA / CPU 2D-to-3D view transform, quoted 0.0059-0.028 ms on GPU and 3.0-24.1 ms on CPU over six input configurations; the page does not say how the table is built (paper: static table, memory); TensorRT / INT8 are in a separate project | 61 commits, no dates shown; deployment section "TODO" | C (repo page, script README) |
| https://github.com/ZRandomize/MatrixVT | MIT (raw LICENSE.md read) | PyTorch 1.9.0 + mmdet3d 1.0.0rc4, built on BEVDepth | page says "official code for MatrixVT on BEVDepth"; it is a public fork of BEVDepth and the link to the authors is not shown; the BEVDepth page itself links `bevdepth/layers/backbones/matrixvt.py` (described as runnable on CPU) and the config `bevdepth/exps/nuscenes/MatrixVT/matrixvt_bev_depth_lss_r50_256x704_128x128_24e_ema.py` | 91 commits | C (both repo pages) |
| https://github.com/Megvii-BaseDetection/BEVDepth | MIT (LICENSE.md) | PyTorch 1.9.0, mmdet3d 1.0.0rc4 | depth supervision and voxel pooling not named on the page (M: `get_downsampled_gt_depth`, `voxel_pooling_train` in the LSS-type backbone `base_lss_fpn.py`) | latest changelog entry 2022-12-06; 91 commits | C (page); function names M |
| https://github.com/nv-tlabs/lift-splat-shoot | non-commercial (above) | PyTorch | `src/tools.py`: `cumsum_trick`, `QuickCumsum` (autograd cumsum pooling with backward), `gen_dx_bx`; `src/models.py` not read | as above | C (raw tools.py) |

Notes: none of these fixes the table by (column, range) as the plan's column-polar lift does; they fix it by camera calibration (850 calibrations on nuScenes) or pool by voxel index. The plan's lift is a sparse matrix product and needs none of their CUDA ops. A scatter / cumsum reference for the spec test: LSS `QuickCumsum` (read-only because of its licence), BEVDet `bev_pool_v2` (Apache-2.0).

## 3 Uncertainty / Gaussian lift
| repo | licence | framework | module | status | marks |
|---|---|---|---|---|---|
| https://github.com/HCIS-Lab/GaussianLSS | MIT (sidebar) | official PyTorch; torch 2.1.0 / Python 3.8 pinned; custom CUDA rasteriser | `GaussianLSS/model/diff-gaussian-rasterization` (installed with `pip install -e .`, acknowledged to graphdeco-inria gaussian-splatting; not gsplat); other folders `callbacks`, `data`, `model`, `visualizations`; files `common.py`, `losses.py`, `metrics.py`; vehicle, pedestrian and map classes on nuScenes; reported IoU 38.3 (visibility 1) / 42.8 (visibility 2) for Eff-b4 at 224 x 480 | 11 commits, no dates shown | C (repo page, tree) |
| https://github.com/Megvii-BaseDetection/BEVDepth | MIT | see block 2 | depth supervision (lidar-projected, downsampled) | see block 2 | C / M |
| https://github.com/huang-yh/GaussianFormer | the LICENSE file exists but is empty (0 bytes), so no licence is granted in practice | based on Sparse4D, migrated to the SelfOcc framework; mmdet3d / mmcv versions not stated | top-level `model`, `loss`, `dataset`, `config`, `train.py`, `eval.py`; the page says only that the Gaussian-to-voxel splatting "can be efficiently implemented with CUDA"; local-aggregation file not identified | 22 commits; news entry 2025-03-01 (GaussianFormer-2 accepted at CVPR 2025) | C (repo page, empty LICENSE) |

Note: the plan's variance-level footprints (3-4 tables, soft level weights) need none of these rasterisers; GaussianLSS is the reference for the rule "variance sets the footprint" (paper note: 2504.01957).

## 4 Virtual camera / homography front end
| repo | licence | framework | module | status | marks |
|---|---|---|---|---|---|
| https://github.com/gigo-team/bev_lane_det (BEV-LaneDet) | not stated | not named | none: the repo has only `.gitignore` and `README.md`; the README says the code is closed for now | 1 commit | C |
| https://github.com/OpenDriveLab/PersFormer_3DLane | Apache-2.0 ("all code within this repository") | PyTorch | `models/PersFormer.py`; `models/networks/`: `Lane3D.py`, `Lane2D.py`, `Deform_ATTN.py`, `PE.py`, `GeoNet3D_ext.py`, `Loss_crit.py`, `libs/`; `models/ops`, `models/nms`; the IPM / perspective transformer file was not identified | 79 commits; newest dated changelog entry 2022-11-03 | C (page, tree); location of the IPM inside `Lane3D.py` / `PersFormer.py`: not seen |
| https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection (Gen-LaneNet) | Apache-2.0, copyright Baidu Apollo (README) | PyTorch 1.4.0, torchvision 0.5.0 | `tools/utils.py` with `homography_im2ipm_norm`, `homography_ipmnorm2g`, `homography_crop_resize`, `projective_transformation`, `transform_lane_gflat2g`, `transform_lane_g2gflat`, `draw_on_ipm` (function names read from the raw file) | 142 commits, no dates shown | C (page, tree, raw utils.py) |

Note: BEV-LaneDet's learned virtual camera cannot be read; the paper note (2210.06006) is the only source. The pure rotation homography of plan section 3.1 is 10 lines of numpy and has an exact test: a point at infinity maps to itself under a rotation about the camera centre; Gen-LaneNet's homography functions are an Apache-2.0 cross-check of the ground-plane variant.

## 5 Radar point encoding / camera-radar fusion
| repo | licence | framework | module | status | marks |
|---|---|---|---|---|---|
| https://github.com/youngskkim/CRN | MIT (sidebar) | mmcv 1.6.0, mmseg 0.28.0, mmdet 2.25.2, pytorch-lightning 1.6.0, torch 1.9.1+cu111, a bundled `mmdetection3d` folder, CUDA ops (`ops/`, GPU needed) | `scripts/gen_radar_bev.py`, `scripts/gen_radar_pv.py`; `layers/backbones/`: `base_lss_fpn.py`, `pts_backbone.py`, `rvt_lss_fpn.py`; `layers/modules/`: `multimodal_deformable_cross_attention.py`, `multi_scale_deformable_attn_function.py`; plus `fuser/`, `heads/` | 3 commits, no dates shown | C (page, `layers` trees) |
| https://github.com/cxmomo/RaCFormer | MIT (sidebar) | torch 2.0.0 / CUDA 11.8, mmcv-full 1.6.0, mmdet 2.28.2, mmseg 0.30.0, mmdet3d 1.0.0rc6 | top level `configs`, `loaders`, `models`, `tools`, `train.py`, `val.py`; the radar depth + RCS embedding file was not identified | 9 commits, no dates shown | C (page) |
| https://github.com/VDIGPKU/RCBEVDet (RCBEVDet) | academic research only; commercial use needs written authorisation | mmdet3d, BEVDet and SparseBEV based | code only as `rcbevdet-master.zip` after signing an application form; weights on Google Drive; RCBEVDet++ code "coming soon" (2024-06-28 note) | 9 commits; RadarBEVNet source not visible | C (page) |
| https://github.com/mrnabati/CenterFusion | MIT (README) | PyTorch 1.2, CUDA 10.0, Python 3.7, needs DCNv2 built separately; based on CenterNet | top folders `data`, `experiments`, `figures`, `models`, `src`; frustum association and pillar expansion files not seen (M: `src/lib/utils/pointcloud.py`) | 6 commits, no dates shown | C (page); file name M |
| https://github.com/phi-wol/hydra (HyDRa) | not found | PyTorch announced | no code: README and `figs` only, "coming soon", placeholder links | 3 commits | C |
| RC-BEVFusion (2305.15883, DFKI / Stellantis) | no repository found by search | - | - | - | search only (digest) |
| CRAFT (2209.06535) | no official repository found by search | - | - | - | search only (digest) |

Note: our radar-side pieces (column of fields, per-sensor polar covariance, sigma points, Doppler normal equations) have no public code. The nuScenes radar field list and loader quirks are in `papers/NUSC_RADAR_nuscenes-radar-fields-and-devkit-quirks.md` (devkit Apache-2.0, above).

## 6 Temporal BEV
| repo | licence | framework | module | status | marks |
|---|---|---|---|---|---|
| BEVDet4D, inside https://github.com/HuangJunJie2017/BEVDet | Apache-2.0 | mmdet3d fork | configs `configs/bevdet/bevdet-r50-4d-*.py` (checked); the BEVDet4D class and feature-alignment code are in the detector file (M: `mmdet3d/models/detectors/bevdet.py`) | as block 2 | C (config names); class path M |
| https://github.com/Divadi/SOLOFusion | Apache-2.0 (sidebar) | mmdet3d fork (version not stated) | configs `configs/solofusion/` (`r50-fp16_phase1.py`, `phase2`, `r50-longonly-fp16_phase2.py`, `r50-shortonly-fp16.py`); sampler `mmdet3d/datasets/samplers/infinite_group_each_sample_in_batch_sampler.py`; two-phase long-term training; the stereo / cost-volume classes were not named on the page | 877 commits; release news 2023-03-14 | C (page) |
| VideoBEV (2303.05970) | no repository found by search | - | - | - | search only (digest) |
| https://github.com/exiawsh/StreamPETR | Apache-2.0 (raw LICENSE read) | mmdet3d plugin style: `projects/mmdet3d_plugin/models/{backbones,dense_heads,detectors,necks,utils}`; version not stated | memory queue and motion-aware LayerNorm are in `dense_heads` and `utils` (file names not seen; M: `streampetr_head.py`) | 81 commits, no dates shown | C (LICENSE, tree); file names M |
| https://github.com/wayveai/fiery | MIT (sidebar + LICENSE) | plain PyTorch | `fiery/layers/temporal.py` and `fiery/layers/convolutions.py` (listing checked); `fiery/utils/` (instance and flow targets: M `instance.py`, `geometry.py`); `losses.py`, `metrics.py`, `trainer.py`; contents of the files not read | 62 commits, no dates shown | C (page, trees); contents M |
| https://github.com/pxiangwu/MotionNet | not an open licence: README says the code is obtained from MERL through a licence request | PyTorch (topics) | page shows only `teaser` and `readme.md` | 4 commits | C |

Note: FIERY is the one plain-PyTorch temporal reference (warp by SE(2), spatial GRU, future flow target); for the plan's section 8 (stationary and moving memory) it is the file to read first.

## 7 BEV trunk + heads
| repo | licence | framework | module | status | marks |
|---|---|---|---|---|---|
| https://github.com/tianweiy/CenterPoint | MIT (README) | forked det3d (not mmdet3d), sparse conv libraries | `det3d/models/bbox_heads/center_head.py` (also `__init__.py`); target helpers `gaussian_radius`, `draw_umich_gaussian` are in `det3d` utils (M, location not seen) | 109 commits; news 2021-12-27 | C (page, tree) |
| https://github.com/zhangyp15/BEVerse | not stated (the repo has no LICENSE file: raw and blob URLs 404) | mmdet3d fork (version not stated) | folders `configs`, `mmdet3d`, `projects`, `tools`; configs `configs/bevdet/bevdet-sttiny.py`, `configs/bevdet4d/bevdet4d-sttiny.py`; heads and temporal module not located | 7 commits, no dates shown | C (page; 404 for LICENSE) |
| https://github.com/Yzichen/FlashOCC | Apache-2.0 (sidebar) | built on BEVDet (plus FB-BEV, RenderOcc, SparseBEV code credited), `projects/mmdet3d_plugin/models/{backbones,dense_heads,detectors,losses,model_utils,necks}`; mmdet3d version not stated | channel-to-height head lives in `dense_heads` (M: `bevocc_head.py`, class `BEVOCCHead2D`; not seen); MMDeploy TensorRT guide; reported M0 197.6 FPS TRT-FP16 and 397.0 FPS INT8 PTQ on an RTX 3090 | 84 commits; news 2025-01-22 | C (page, tree); head names M |

## 8 Multi-task loss balancing
| repo | licence | framework | module | status | marks |
|---|---|---|---|---|---|
| https://github.com/AvivNavon/nash-mtl | MIT (raw LICENSE read) | PyTorch 1.9.0 | `methods/weight_methods.py`: classes `Uncertainty` (one log-variance per task, init 0, loss scaled by exp(-logvar) plus the logvar term, factor 0.5), `NashMTL`, `PCGrad`, `CAGrad`, `MGDA`, `IMTLG`, `DynamicWeightAverage`, `RLW`, `LinearScalarization`, `STL`; GradNorm not included | 27 commits | C (page, raw weight_methods.py) |
| https://github.com/median-research-group/LibMTL | MIT (README) | PyTorch | 16 weighting strategies including UW and GradNorm, chosen by `--weighting`; folder `LibMTL/` | 180 commits; news 2025-04-21 | C (page) |
| https://github.com/lucidrains/gradnorm-pytorch | MIT | PyTorch | class `GradNormLossWeighter`; package `gradnorm_pytorch/` | 46 commits; last push January 2024 (search result, digest) | C (page); date digest |
| https://github.com/LucasBoTang/GradNorm | Apache-2.0 | PyTorch | `gradnorm.py`, function `gradNorm(net, layer, alpha, dataloader, ...)` | 50 commits | C (page) |
| https://github.com/yaringal/multi-task-learning-example | MIT (raw LICENSE read) | notebooks `multi-task-learning-example-pytorch.ipynb` and a second notebook | author's toy example of the uncertainty-weighting paper (1705.07115); code not read | 7 commits | C |

Note for the plan: uncertainty weighting is a 4-line module (log-variance parameters with a clamp); whether the nash-mtl class clamps the log-variance was not seen, the plan asks for a clamp. A spec test needs no repo: two tasks with known noise levels sigma_1, sigma_2 must recover log sigma^2 within a tolerance.

## Gaps and licence flags
- Not usable by copying (read only / re-implement): LSS (non-commercial), HDMapNet (GPL-3), RCBEVDet (academic only), MotionNet (licence request), BEVerse and Fast-BEV (no licence named), GaussianFormer (empty LICENSE), BEV-LaneDet (no code), HyDRa (no code), RC-BEVFusion, CRAFT and VideoBEV (no repository found).
- nuScenes data and its map expansion are non-commercial; the devkit code is Apache-2.0.
- Items still to open on a code pass if we port: BEVDet `view_transformer.py` and `bev_pool_v2` (the cleanest Apache-2.0 reference for pooled lifts), FIERY `temporal.py`, CenterPoint `center_head.py`, Gen-LaneNet `tools/utils.py`, CRN `rvt_lss_fpn.py`.
