# RB01 - Post Fusion Stabilizer (PFS): identity-initialised repair module between frozen fusion and head

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: Post Fusion Bird's Eye View Feature Stabilization for Robust Multimodal 3D Detection (Dong, Thakkar, Sargolzaei, Lin, arXiv 2026, v2 3 Sep 2026) https://arxiv.org/abs/2603.05623 (opened as HTML https://arxiv.org/html/2603.05623)
- depth: M
- stage: B, T
- idea: A small module sits between the fused BEV feature map and the unchanged detection head of a frozen host detector. It first normalises distribution shift, then estimates a per-cell reliability map, then runs expert correction / inpainting; every block starts as identity so clean performance is kept. Only the module is trained, in three curriculum stages, on clean and corrupted data.
- formulation: three blocks - shift normalisation (~99 K params), spatial reliability estimation (~664 K), expert correction and inpainting (~2.5 M); trained with detection and reliability losses (exact loss form not stated in what I read). Input and output tensors have the same shape.
- input: fused BEV feature map (shape not stated).
- output: corrected BEV feature map of the same shape (plus an internal reliability map).
- training: host frozen; module trained with a curriculum on clean + synthetic corruptions (camera rain lens, low light, occlusion, 6-camera dropout; LiDAR beam reduction, range / sector dropout, miscalibration; nuScenes-C fog, snow, sunlight).
- differentiable: yes - plain feed-forward on a raster.
- sensors_data: camera + LiDAR, nuScenes val; hosts BEVFusion and UniBEV; a 200-frame physical test with one camera and a 32-beam LiDAR, no fine-tuning.
- results: BEVFusion mAP clean 69.5 vs 69.2 baseline; Low Light 63.2 vs 58.8; Rain Lens 63.6 vs 60.0; 6-camera dropout 63.8 vs 62.6 (only +1.2); 8-beam 17.5 vs 12.1; nuScenes-C Fog 67.9 vs 54.1 (paper text and Table III disagree on snow / sunlight gains). Source: paper tables as summarised.
- compute: BEVFusion 40.8 M -> 44.1 M params; latency 70.9 -> 76.6 ms (+8.1 %, +5.7 ms); UniBEV 417 -> 436 ms (+4.5 %). GPU not stated in what I read.
- code: none found on the page.
- fit_to_us: reimplement - effort M - closest published match to our interface (frozen features in, same-shaped features out); we would train it on cached clean vs corrupted 39-channel rasters, but the gain for lost-camera (+1.2 mAP) is small and it was not tested on map outputs.
- compat_needs: a paired set of corrupted and clean cached features; a loss on the downstream head output or a feature-consistency loss.
- compat_conflicts: cannot restore information that the frozen foundation never encoded (lost camera views, wrong extrinsics inside the foundation).
- tried_before: close to "fine-tuning on failure frames" (helps night / blur, nothing for lost cameras / extrinsics) but PFS puts the capacity in front of the head and adds a reliability map; not tried as such.
- verified: partly

## Correction from the level-2 extraction (2026-10-25)

- Loss form is now known: Stage 2 uses detection loss plus 1.0 x BCE of the reliability map against a target built from LiDAR point-count ratios per cell (corrupted samples) plus 0.2 x BCE against 1 on clean samples; Stages 1 and 3 use the detection loss only.
- Input is not only the fused map: Block 2 also takes the host's LiDAR BEV feature, and the reliability map is a LiDAR-density map. It is never trained to flag camera failures (target 1), which is a likely reason for the small 6-camera dropout gain; the paper gives no reason itself.
- nuScenes-C fog, snow and sunlight are evaluation conditions only; the eight training corruptions are listed in Table I. Text and Table III agree (text gives relative gains, table points); the real text/table mismatch is the UniBEV physical-test row.
- Identity initialisation is approximate (gates 0.0067 and 0.018, reliability bias +4); BEVFusion clean mAP dips to 68.7 / 68.9 before Block 3 restores it (69.5 against 69.2). Quote mAP, not the BEVFusion NDS columns.
- No code, no licence statement anywhere (abstract page, HTML v2, alphaXiv, two searches).

Details: [../deep/RB01_post-fusion-stabilizer.md](../deep/RB01_post-fusion-stabilizer.md).
