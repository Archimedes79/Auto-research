# RB17 - RoboBEV / nuScenes-C: corruption suite incl. Camera Crash, Frame Lost, Dark, Fog, Motion Blur

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: RoboBEV: Towards Robust Bird's Eye View Perception under Corruptions (Xie, Kong et al., 2023) https://arxiv.org/abs/2304.06719 (seen in search listing; page not opened); Benchmarking and Improving Bird's Eye View Perception Robustness in Autonomous Driving (expanded version, arXiv 2405.17426, May 2024) https://arxiv.org/abs/2405.17426 (opened HTML v2); code github.com/Daniel-xsy/RoBoBEV (from search listing)
- depth: M (for 2405.17426)
- stage: E, D
- idea: Eight corruptions at three severities (Bright, Dark, Fog, Snow, Motion Blur, Color Quant, Camera Crash, Frame Lost) applied to nuScenes val; 33 BEV models; metrics mCE / mRR; extends to multimodal sensor failure and offers a two-stage CLIP-based fix.
- results: map segmentation (CVT, IoU, Table IV): clean 0.348, Camera Crash 0.200, Frame Lost 0.170, Dark 0.200, Fog 0.247, Snow 0.177. Multimodal BEVFusion (Table VI): camera failure mAP 0.6852 -> 0.6247; LiDAR failure NDS 0.7138 -> 0.3340 (mAP drop about 89 %). Corruption-augmented CLIP model mRR 84.32 vs 58.54 for BEVDet baseline (Table X); frozen CLIP alone transfers poorly; end-to-end fine-tuning adds little.
- compute: not reported.
- fit_to_us: baseline - effort S - the corruption definitions (Dark, Fog, Motion Blur, Camera Crash, Frame Lost at 3 severities) can be reproduced in our failure synthesis; our failure suite should report RR = corrupted / clean per severity.
- tried_before: our own synthetic night / blur / radar-off.
- verified: partly (first paper by listing only)
