# INV Small pretrained image backbones with stride-4/8/16 feature maps (timm): cost, licence, and where one has been used in a BEV model

| | |
|---|---|
| source | Hugging Face timm model cards (URLs below), read through a page summariser on 2026-10-30; BEV usage from the papers cited in each row; GaussianCaR numbers from https://arxiv.org/html/2602.08784v2 |
| fact status | checked for parameters, GMACs, input size, licence tag and the `features_only` output shapes quoted on each card (strides are derived by me from the shapes, the cards do not state them); the column "at 256 x 704" is DERIVED by scaling GMACs with pixel count (ratio 3.59 from 224 x 224, 2.75 from 256 x 256, 1.22 from 384 x 384), ignoring head and non-convolutional effects |
| groups | F4 (small encoders): 3; A1: 2 |
| code / licence | timm code Apache-2.0; per-model licence tag as listed; ImageNet-trained weights inherit the ImageNet terms (not reviewed here) |
| verdict | use (shortlist for the stride-8 student: EfficientNet-B0 / B2, EfficientViT-B1, RepViT-M1.1, MobileNetV4-conv-medium, RegNetY-008; none has a published BEV result at that size, so the choice is ours to measure) |

**What it does**: lists, for backbones named in the brief, what a `timm.create_model(..., features_only=True)` gives and what it costs. timm reports multiply-accumulates; FLOPs are about twice the GMACs (our "5-16 GFLOPs" budget is therefore about 2.5-8 GMACs for the whole image side).

**Results that matter** (card values; strides derived from output shapes):

| backbone (card id) | params | GMACs (ref. size) | at 256 x 704 (derived) | feature channels at stride 4 / 8 / 16 / 32 | licence tag | BEV use found |
|---|---|---|---|---|---|---|
| `resnet18.a1_in1k` | 11.7M | 1.8 (224) | 6.5 | 64 / 128 / 256 / 512 | apache-2.0 | Fast-BEV M0 / M1 (R18), the common small baseline |
| `resnet50.a1_in1k` | 25.6M | 4.1 (224) | 14.7 | 256 / 512 / 1024 / 2048 (standard, memory; the card was not read for shapes) | apache-2.0 | DualBEV, BEVDet, PointBeV, FastBEV++ |
| `efficientnet_b0.ra_in1k` | 5.3M | 0.4 (224) | 1.4 | 24 / 40 / 112 / 320 | apache-2.0 | BEVal LSS, KD360 student, Jun and Lee (46.4 mIoU at 224 x 480 vs 47.7 for B4) |
| `efficientnet_b2.ra_in1k` | 9.1M | 0.9 (256) | 2.5 | 24 / 48 / 120 / 352 | apache-2.0 | none found |
| EfficientNet-B4 (card not read; 19.0M from Jun and Lee) | 19.0M | not read | not read | not read | not read | CVT, PointBeV, GaussianLSS (1/8 maps, 128 ch after the neck) |
| `efficientvit_b1.r256_in1k` | 9.1M | 0.7 (256) | 1.9 | 32 / 64 / 128 / 256 | apache-2.0 (card notes the original repo is MIT) | none found at B1 |
| `efficientvit_l2.r384_in1k` | 63.7M | 20.4 (384) | 24.9 | 64 / 128 / 256 / 512 (96 x 96 to 12 x 12 for a 384 input) | apache-2.0 | GaussianCaR: replacing EfficientNet-B4 in GaussianLSS by L2 took vehicle IoU 46.1 to 47.3 and the camera-only pipeline 53.9 to 56.6 ms (RTX 4090, 448 x 800, six cameras) |
| `mobilenetv4_conv_medium.e500_r256_in1k` | 9.7M | 1.1 (256) | 3.0 | 48 / 80 / 160 / 960 | apache-2.0 | none found |
| `convnext_atto.d2_in1k` | 3.7M | 0.6 (224) | 2.2 | 40 / 80 / 160 / 320 | apache-2.0 | none found |
| `repvit_m1_1.dist_450e_in1k` | 8.8M | 1.4 (224) | 5.0 | 64 / 128 / 256 / 512 | apache-2.0 (original THU-MIG/RepViT) | none found |
| `regnety_008.pycls_in1k` | 6.3M | 0.8 (224) | 2.9 | 64 / 128 / 320 / 768 | MIT | none found |
| `hgnetv2_b0.ssld_stage2_ft_in1k` | 6.0M | 0.3 (224) | 1.1 | 64 / 256 / 512 / 1024 | apache-2.0 (PaddleClas source) | none found (b4: 19.8M listed in comparison tables only) |
| `fastvit_t8.apple_dist_in1k` | 4.0M | 0.7 (256) | 1.9 | 48 / 96 / 192 / 384 | "other": Apple ml-fastvit LICENSE, not an OSI licence | none found; avoid unless the licence is read |
| `swin_tiny_patch4_window7_224.ms_in1k` | 28.3M | 4.5 (224) | 16.2 (window attention: approximate) | 96 / 192 / 384 / 768 (from the BEVFusion config, not the card) | MIT | BEVFusion, BEVDet-Tiny |
| MiT-B0 / B2, InternImage-T / B | 3.7 / 25.4 / 30.0 / 97.0M | not read | not read | not read | not read | Jun and Lee: MiT-B0 44.8, MiT-B2 46.2, ITN-T 51.3, ITN-B 52.2 mIoU (LSS, 224 x 480) |

The EfficientNet cards also list a stride-2 output (16 channels); the table lists strides 4 / 8 / 16 / 32 only.
Card URLs: https://huggingface.co/timm/{resnet18.a1_in1k, resnet50.a1_in1k, efficientnet_b0.ra_in1k, efficientnet_b2.ra_in1k, efficientvit_b1.r256_in1k, efficientvit_l2.r384_in1k, mobilenetv4_conv_medium.e500_r256_in1k, convnext_atto.d2_in1k, repvit_m1_1.dist_450e_in1k, regnety_008.pycls_in1k, hgnetv2_b0.ssld_stage2_ft_in1k, fastvit_t8.apple_dist_in1k, swin_tiny_patch4_window7_224.ms_in1k}. The `hgnetv2_b2.ssld_stage2_ft_in22k_in1k` card returned HTTP 401 (name probably wrong); the HGNetV2 numbers are from the b0 card only.

**Transferable technique**: (1) every candidate delivers stride 4, 8 and 16 maps from one forward pass, so a stride-8 head costs only the 1 x 1 / 3 x 3 neck on top; every row except R18 (6.5), R50 (14.7) and Swin-T (16.2) is at or below 5 GMACs at 256 x 704 (derived), i.e. at or below about 10 GFLOPs for the backbone: the proposed 5-16 GFLOPs budget for backbone plus neck plus heads is realistic. (2) Published BEV evidence ranks encoders only coarsely: B0 within 1.3 mIoU of B4 (Jun and Lee), EfficientViT-L2 +1.2 IoU over B4 for +2.7 ms (GaussianCaR), a stronger 2D prior (InternImage) +4 to +6; there is no head-to-head of the small backbones in the table, so a matched test (same neck, same lift, same loss) is needed and cheap. (3) Compile / export friendliness: plain conv nets (EfficientNet, RegNet, ResNet, MobileNetV4) export without custom ops; RepViT needs its re-parameterisation step before export; EfficientViT uses linear attention (ReLU) and runs in TensorRT per its authors (memory); the MIT EfficientViT repository states it is no longer maintained (checked), timm carries the weights.

**Cost and fit**: parameter and MAC figures are for the classifier trunk; the FPN / neck to 128 channels at stride 8 adds a few percent. nuImages / COCO driving-domain inits exist only for ResNet (Fast-BEV R18 / R34 / R50, mmdet3d R50 / R101 / X101) and Swin-T (BEVFusion); every other row is ImageNet-only.

**Doubts**: GMACs at 256 x 704 are scaled, not measured; latency on a 3060 at batch 1 is not given by any source here (MACs do not predict it for depthwise or attention nets); the "BEV use found" column is from the pages read in this scan, not an exhaustive search; the licence column is the card tag, not the licence of the training data.
