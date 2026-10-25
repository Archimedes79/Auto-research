# RB21 - Robust BEV segmentation by adapting a frozen DINOv2 with LoRA

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: Robust BEV Segmentation via DINOv2 Adaptation (title as listed in search; arXiv 2409.10228, 2024) https://arxiv.org/abs/2409.10228 (opened HTML)
- depth: M
- stage: B, T
- idea: Keep a DINOv2 image encoder frozen and add LoRA matrices to the attention projections; a SimpleBEV decoder (about 5 M) lifts and decodes to BEV occupancy; evaluated on nuScenes-C.
- training: BCE on BEV occupancy; trainable parameters ViT-B LoRA 1 M, ViT-L LoRA 3 M, ResNet-101 37 M, full ViT-B 86 M.
- results: ViT-L beats SimpleBEV by at least 20 % relative on most corruptions; darkness: SimpleBEV falls below 30 % of its clean value, DINOv2 variants hold much better; fog loses under 20 %; camera crash and frame loss: no better than SimpleBEV. Exact numbers only in a figure.
- compute: latency not reported.
- fit_to_us: reuse_idea - effort L (needs trainable image front end) - evidence that a stronger frozen image foundation helps night and fog but not lost cameras, matching our own fine-tuning finding.
- tried_before: GaussianCaR vs CRN comparison (E27) is a different axis.
- verified: checked
