# SENSORS25 Jun and Lee - comparative study of camera-based BEV segmentation: six image encoders under one LSS, two input sizes, RTX 4090 and AGX Orin latency

| | |
|---|---|
| source | "A Comparative Study and Optimization of Camera-Based BEV Segmentation for Real-Time Autonomous Driving", Sensors 2025, 25(7), 2300: https://pmc.ncbi.nlm.nih.gov/articles/PMC11991255/ (tables read through a page summariser, 2026-10-30; no arXiv id) |
| fact status | checked (summariser; table cells as returned). The summariser lists inconsistencies inside the article (text says LSS ITN-T 448 x 800 = 51.6 and BEVFormer = 49.3, the tables say 53.1 and 50.8; FP16 LSS 224 x 480 loses 2.3 mIoU although the text says "similar"), so use table values and treat 1-2 points as noise |
| groups | F4 (encoder choice vs accuracy vs latency): 3; A1: 1 |
| code / licence | no code or weights mentioned on the page; open access article (licence of the paper not checked); the LSS in it is the original LSS design (NVIDIA research-only licence for the original code, see the LSS note) |
| verdict | use (the only head-to-head of EfficientNet, MiT and InternImage encoders with latency in one BEV pipeline; fits our shortlist question, though the stride is not stated) |

**What it does** [checked]: three BEV methods (LSS, an MLP-based HDMapNet variant, BEVFormer) on nuScenes six-camera, 100 m x 100 m, with image encoders EfficientNet-B0 / B4, MiT-B0 / B2 and InternImage-T / B (ITN) and a BEV decoder of EfficientNet-B0 type; inputs 224 x 480, 448 x 800 and 672 x 1200; FP32 vs FP16; latency on an RTX 4090 and (FP16, ITN-T only) on a Jetson AGX Orin at 30 W and 50 W. Classes: lane, crosswalk, boundary, car (names as in the table). The feature stride of each encoder is not stated.

**Results that matter** [checked; LSS, nuScenes, FP32; mIoU over the four classes; latency ms on RTX 4090; size MB]:

| encoder (params) | input | mIoU | lane | crosswalk | boundary | car | ms | MB |
|---|---|---|---|---|---|---|---|---|
| EN-B0 (5.3M) | 224 x 480 | 46.4 | 53.2 | 42.3 | 57.2 | 32.9 | 21.9 | 59.4 |
| EN-B0 | 448 x 800 | 49.1 | 54.6 | 46.1 | 59.6 | 36.1 | 30.1 | 59.7 |
| EN-B4 (19.0M) | 224 x 480 | 47.7 | 53.9 | 44.1 | 58.6 | 34.3 | 30.5 | 116.1 |
| EN-B4 | 448 x 800 | 48.3 | 55.3 | 45.7 | 60.0 | 38.3 | 50.1 | 116.4 |
| MiT-B0 (3.7M) | 224 x 480 | 44.8 | 51.8 | 39.7 | 55.6 | 32.2 | 19.7 | 51.6 |
| MiT-B0 | 448 x 800 | 47.7 | 54.5 | 43.2 | 58.1 | 35.0 | 28.8 | 51.9 |
| MiT-B2 (25.4M) | 224 x 480 | 46.2 | 53.1 | 40.7 | 56.8 | 34.3 | 25.8 | 138.5 |
| MiT-B2 | 448 x 800 | 48.3 | 55.8 | 45.0 | 59.6 | 33.1 | 50.0 | 138.8 |
| ITN-T (30.0M) | 224 x 480 | 51.3 | 58.8 | 51.5 | 63.0 | 37.9 | 30.6 | 159.2 |
| ITN-T | 448 x 800 | 53.1 | 58.8 | 51.0 | 62.8 | 40.0 | 51.7 | 159.5 |
| ITN-B (97.0M) | 224 x 480 | 52.2 | 59.7 | 52.1 | 63.9 | 39.2 | 37.7 | 427.0 |
| ITN-B | 448 x 800 | 54.9 | 60.5 | 53.6 | 64.5 | 41.2 | 95.5 | 427.3 |

Other tables: with ITN-T at 224 x 480 the three BEV methods give LSS 51.3, HDMapNet-variant 47.1, BEVFormer 48.0 mIoU at 30.6 / 30.4 / 47.5 ms; FP16 halves the model size, mIoU changes by 0 to -2.3; AGX Orin, FP16, ITN-T, LSS: 392.6 ms (30 W) and 216.9 ms (50 W) at 224 x 480, 1167 and 582 ms at 448 x 800. The authors call ITN-T the best accuracy-efficiency trade-off (ITN-B adds 0 to 2 mIoU at 2.7x the size).

**Transferable technique**: (1) in one fixed pipeline, a 5.3M EfficientNet-B0 loses only 1.3 mIoU against the 19M B4 at 224 x 480 and gains more from resolution (+2.7) than B4 does (+0.6) - encoder size is cheap to give up, input size is not; (2) the lane class barely responds to a 4x pixel count with this LSS (EN-B0 53.2 to 54.6, EN-B4 53.9 to 55.3, ITN-T 58.8 to 58.8) while the car class gains +2 to +4: consistent with a lift whose feature stride (16 in the original LSS code, from memory, not stated in the article) is the limit for thin structures - an inference, not a result of the paper; (3) a stronger encoder (InternImage-T, deformable convolution) lifts every class by +3.6 to +7.4 points over EN-B4 at the same latency (224 x 480: 30.6 vs 30.5 ms), so a better 2D prior can be worth more than any lift change; InternImage needs DCNv3 ops, which a 3060 deployment must carry.

**Cost and fit**: the numbers are six cameras at batch size one on a 4090 (FP32 unless stated); one camera is about a sixth of the encoder work. EN-B0 at 224 x 480 is 21.9 ms for the whole six-camera LSS: the encoder is not the bottleneck there. On the Orin the same net is 217-393 ms, so the 4090 numbers do not carry to a small device.

**Doubts**: the dataset labels are the authors' (lane, crosswalk, boundary, car), not the six nuScenes map layers; stride and depth head per encoder not stated; the article's own text contradicts its tables in several places; no code; single seed implied; "Orin" only for FP16 ITN-T.
