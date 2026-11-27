# INV Pretrained camera-to-BEV checkpoints that exist (2026-10-30): what they output, encoder and stride, licence, where

| | |
|---|---|
| source | GitHub README / release pages and Hugging Face cards named per row, all read through a page summariser on 2026-10-30; no file was opened or downloaded (sizes are only given where a page showed them) |
| fact status | checked per row unless marked digest (search snippet) or memory; "licence" = what the page names for code and for weights separately; every checkpoint here was trained on nuScenes (non-commercial data terms) |
| groups | A1: 3 (a donor or teacher for the image side); F4: 3 |
| code / licence | see table |
| verdict | use: one map-output checkpoint with a stride-8 encoder exists with permissive code (BEVFusion camera-only-seg); several vehicle-only stride-8 checkpoints (PointBeV, GaussianLSS, Simple-BEV) with MIT code; nothing small and front-camera-only |

**What it does**: inventory of downloadable models that map camera images to a BEV semantic map or to boxes, as candidate teachers / encoder donors for a stride-8 image side.

**Results that matter** (map or segmentation output first):

| model / checkpoint | output | encoder, stride, input | metric (as stated on the page) | code / weights licence | where, size |
|---|---|---|---|---|---|
| BEVFusion `camera-only-seg.pth` (2205.13542) | 6 map classes (drivable, ped. crossing, walkway, stop line, carpark, divider), 200 x 200 at 0.5 m | Swin-T, FPN 256 ch at stride 8, 256 x 704, six cameras | 57.09 mIoU (README); divider 46.4 (paper) | Apache-2.0 / weights not stated; repo archived 2025-01-12 | Dropbox via README script; size not shown |
| PointBeV (2312.00703) | BEV segmentation (rows not labelled by class; the paper's headline is vehicle), 100 m, 0.5 m | EfficientNet-B4, stride 8 (28 x 60 at 224 x 480, from the paper appendix), 128 ch | 38.69 / 42.09 (224 x 480 / 448 x 800, visibility filter 1), 43.97 / 47.58 (filter 2) | MIT / not stated | GitHub Releases tag "checkpoints" (`38_69.ckpt`, `42_09.ckpt`, `43_97.ckpt`, `47_58.ckpt`); size not shown; no other backbone checkpoint |
| GaussianLSS (2504.01957) | vehicle (class not labelled on the README) | EfficientNet-B4, stride 8, 224 x 480 | 38.3 (filter 1) / 42.8 (filter 2) | MIT / not stated | GitHub Releases `effb4_224_480_1.ckpt`, `effb4_224_480_2.ckpt`; "Add checkpoints" still open in the TODO |
| GaussianCaR `nuscenes_vehicle.ckpt`, `nuscenes_map.ckpt` (2602.08784) | vehicle; map (drivable, divider) | EfficientViT-L2 (timm), stride 8, 448 x 800, plus a radar Point Transformer: needs radar input | vehicle 57.3; drivable 82.9; lane divider 50.1; 75.6 / 81.1 ms on an RTX 4090 | Apache-2.0 (rasteriser: Inria non-commercial) / Hugging Face tag Apache-2.0 | HF `santimontieleu/gaussiancar` (empty model card, sizes not shown); camera-only configuration not offered |
| Simple-BEV (2206.07959) | vehicle | ResNet-101, stride 8 (128 ch), 448 x 800 | camera-only 47.6, camera + radar 55.8 (README) | MIT / not stated | download scripts `get_rgb_model.sh`, `get_rad_model.sh`; URL and size not on the page |
| Cross-View Transformers (2205.02833) | vehicle (and road in the paper) | EfficientNet-B4, features at 1/8 and 1/16 (28 x 60 and 14 x 30), input 224 x 448, paper: 35 FPS on an RTX 2080 Ti, 5M parameters | vehicle 36.0 (setting 2), drivable 74.3 | MIT / MIT per the Qualcomm card | the official README lists NO checkpoint; Qualcomm re-export `qualcomm/CVT` on Hugging Face: `vehicles_50k.pt`, 6 x 3 x 224 x 480 input, 5.18 MB float, 1.33M parameters as listed there |
| BEVal model zoo (arXiv 2408.16322) | vehicle / human / drivable / vehicle + drivable, 100 m at 0.5 m | LSS with EfficientNet-B0 (and LiDAR-depth variants LAPT, FPN) | numbers only in the paper | MIT-derived code / CC BY-NC 4.0 | HF `manutheeng/beval_model_zoo`, 60 models (nuScenes, Lyft, joint); sizes not shown |
| FIERY (2104.10490) | future instance segmentation (static LSS-setting checkpoint also listed) | EfficientNet-B4 | 36.7 IoU (digest from a search snippet) | MIT (digest) | `fiery.ckpt` v1.0 release (digest) |
| Fast-BEV (2301.12511) | 3D boxes | R18 / R34 / R50, FPN 1/4-1/16, 256 x 704 up to 512 x 1408 | M0 0.277 mAP ... M5 0.369 | not named / not stated | Google Drive `epoch_20.pth` per config; plus nuImages-pretrained Cascade Mask R-CNN R18 / R34 / R50 backbones |
| FastBEV++ (2512.08237) | boxes | R50 / R101, 256 x 704 | 0.344 / 0.478 (C), 0.359 / 0.488 (C&D) | Apache-2.0 / not stated | Drive; R50 file 170.5 MB (existing code note) |
| DualBEV (2403.05402) | boxes | R50, stride 16, 256 x 704 | 36.8 / 43.4 (1 frame, README) | Apache-2.0 / not stated | Drive, 316.7 MB (existing code note) |
| SA-BEV (2307.11477) | boxes | R50 | R50 35.5 / 46.7; MSCT 37.0 / 48.8; MSCT + CBGS 38.7 / 51.2 | Apache-2.0 / not stated | Drive + Baidu, same links for all rows |
| BEVNeXt (2312.01696) | boxes | R50; paper: depth network on F1/8 at 256 x 704 (summariser) | Stage 2: 54.8 NDS / 43.7 mAP (ImageNet init), 56.0 / 45.6 (FCOS3D init) | Apache-2.0 / not stated | Hugging Face; stage 1 and 2 files |
| BEVFormer v2 (2211.10439) | boxes | R50 only in the zoo (paper has InternImage-B, V2-99; FPN strides 8-128) | t1 24 ep 45.3 NDS ... t8 55.3 | Apache-2.0 (repo) / not stated | Drive + Baidu; the page shows no resolution, stride or latency |
| mmdet3d nuImages 2D models | 2D instance segmentation, nuImages classes | Mask / Cascade Mask R-CNN R50 / R101 / X101 with FPN (strides 4-64 from memory) | Mask R-CNN R50 box AP 47.8 / mask AP 38.4; Cascade R50 IN + COCO 52.8 / 42.2 | licence not named on the page (Apache-2.0 repository: memory); nuImages data is non-commercial (memory) | `download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/...` (links on the config page) |

**Transferable technique**: (1) the one released model with a map-class output, a stride-8 FPN encoder and an open licence is BEVFusion camera-only-seg; use it as a teacher for the front sector or as a reference for the head layout, not as an encoder: six cameras, Swin-T, mmdet3d. (2) For a vehicle-only stride-8 donor, PointBeV and GaussianLSS give EfficientNet-B4 weights with MIT code; the checkpoints are the full networks, so the encoder has to be cut out by key prefix. (3) For a small student the useful ready-made weights are not BEV models but 2D ones: ImageNet timm weights (see the timm inventory note) or the driving-domain nuImages detectors (Fast-BEV R18 / R34 / R50 backbones, mmdet3d nuImages R50 Mask R-CNN, BEVFusion `swint-nuimages-pretrained.pth`). (4) No downloadable model has a lane-centre or thin-line output that is more than a "divider" mask.

**Cost and fit**: all sizes unknown except CVT (5.18 MB), FastBEV++ R50 (170.5 MB) and DualBEV (316.7 MB). All code stacks except CVT, Simple-BEV, PointBeV, GaussianLSS are mmcv / mmdet3d based and have no CPU path; GaussianLSS and GaussianCaR need a compiled rasteriser.

**Doubts**: class meaning of the PointBeV and GaussianLSS rows is inferred; the CVT parameter count on the Qualcomm card (1.33M) differs from the paper (5M) and the encoder may be excluded (not checked); FIERY and the CVT MIT tag are partly digest; weight licences are nowhere stated explicitly, so research-only is the safe reading; the BEVNeXt stride is the summariser's reading of the paper.
