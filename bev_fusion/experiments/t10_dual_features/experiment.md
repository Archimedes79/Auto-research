# t10 dual features - can one trunk use the map-trained BEVFusion features AND the detection-trained DualBEV features? (card written 2026-10-31, before any code or run)

Context: t8 (`../t8_bevfeat/experiment.md`, same-code control, two seeds, cold VAL 1000): frozen BEVFusion camera features (map-trained, stride 8, weak depth)
instead of the frozen DualBEV cache (detection-trained, stride 16, sharp depth) nearly double the maps (lane divider@0.5 0.191 -> 0.351 / 0.345, drivable 0.716 -> 0.804 /
0.809, walkway 0.178 -> 0.339 / 0.342, crossing 0.002 -> 0.145 / 0.165) and halve the objects (front mAP2 0.367 -> 0.183 / 0.180, pedestrian AP@2 0.210 -> 0.001 / 0.002).
M6 (the image features limit the maps) is supported; the reverse holds for objects. Neither cache alone gives both. This card is the cheapest test of "both":
the two cached feature sets side by side, no new encoder, no new cache (C: has 27 GB free; both caches exist on D:).
It does not replace t9 (`../t9_encoder_2x2/experiment.md`, written before t8): t9's R arm is the trainable version of the same idea, and its premise
("no layer reaches 60 % of GaussianCaR in 3000 steps") is nearly overtaken by t8 on the maps, with a frozen encoder and no map-loss training (lane divider 0.35 =
51 % of GaussianCaR's 0.691, walkway 0.34 = 57 % of 0.594, t6 report table, same frames and tolerance), so the order of the lane should be re-read after t10.

## 1 Card

- **Question.** With the frozen DualBEV BEV and the frozen BEVFusion BEV both available to the trunk, does one chain keep the maps of t8_bevfeat AND the objects of t8_ctl?
- **Hypothesis.** The two camera BEVs carry complementary information (maps / ground semantics from BEVFusion's map-trained stride-8 features, object depth and
  object features from DualBEV's depth net); the merged BEV (same channel count into the same trunk) is used for both. Competing explanations the arm can tell apart:
  (a) the trunk uses both: maps AND objects recover; (b) the trunk takes the easier source and ignores the other (a shortcut): one family recovers only; (c) the
  loss of the objects with BEVFusion features is not only the weak depth but the missing object information in the map-trained context (then no mixing of depth helps;
  a hybrid of "BEVFusion context x DualBEV depth" would be the next test, t9's R arm is its trainable cousin).
- **Design.** One arm `t10_dual_s0` (seed 1 only if it passes a rule): everything identical to `t8_bevfeat_s0` / `t8_ctl_s0` (3 frames, window memory, radar, 3000 steps,
  Kendall, lr 3e-4, EMA, flip, GT signature 15efbde2d1, the code commit of t8 plus the default-off option below), cold VAL 1000 frames by `bev_suite eval --n 1000`.
  Controls are the existing same-code runs `t8_ctl_s0/s1` (DualBEV only) and `t8_bevfeat_s0/s1` (BEVFusion only): no new control run. Cost ~100 min per seed
  (two lifts, two reductions; the data loader reads both caches: 1.4 MB per frame instead of 0.3 / 1.1 MB).
- **The option (default off, bit-identical defaults).** `ModelConfig.camera_features = "dual"`:
  - dataset (`feature_cache.CachedFrontRadarDataset`): a second `FeatureStore` from `feature_cache2` (= `data/cache/cam_feats_bevfusion`; the first is
    `data/cache/cam_feats`); each frame gets `prob2` (118, 32, 88) / `ctx2` (80, 32, 88) next to `prob` / `ctx`; the item carries `ida2 = bevfusion_ida()`
    beside the DualBEV `ida`; `collate_cached` stacks `prob2`, `ctx2`, `ida2`.
  - model (`BEVFusionModel.__init__` and `CachedBEVFusionModel._Cached.camera_bevs`): branch 1 as today (`self.lift` with `depth_centres()` on the stride-16 cache);
    branch 2 = `self.lift2 = LiftPD(grid, depth_centres=bevfusion_camera.DEPTHS, frustum_cells=True, oversample=cfg.oversample)` with its own `self.cam_reduce2`
    (same type as `cam_reduce`); the two reduced BEVs (c_cam each) are concatenated and merged by `self.cam_merge = Conv2d(2 c_cam, c_cam, 1) + BN + ReLU`, so
    everything downstream (temporal window memory, aux head, fusion, trunk, heads) sees the same shapes as in the control. `camera_visibility` keeps using branch 1 (same
    camera geometry; the visibility is calibration only). `camera_off` still returns zeros. `lift == "s"` stays unsupported.
  - config `t10_dual_s0.json` = `t8_bevfeat_s0.json` + `"model": {"camera_features": "dual"}` + `"feature_cache2": "data/cache/cam_feats_bevfusion"` (and `feature_cache` back to
    `data/cache/cam_feats`).
- **Gate (layer 0, before the run; synthetic tensors, CPU, a few seconds; I write `temp/t10/gate.py` as soon as the option exists).**
  G1 shapes: the camera BEV is (B, F, c_cam, X, Y) as in the control; G2 branch select: with `cam_merge` set to [I, 0] the dual BEV equals the DualBEV-only camera BEV
  (same weights, same inputs), with [0, I] it equals the BEVFusion-only camera BEV (each branch uses its own depth centres, frustum convention and `ida`); G3 gradients: a loss on the
  merged BEV gives non-zero gradient to both reductions and the merge, and none to the cached inputs; G4 `camera_off` gives zeros without reading the features; G5 the
  default configs reproduce their own output bit for bit (existing selftests of model.py / feature_cache.py pass); G6 the first real step: peak GPU memory and seconds per step
  (the card is shared: peak below 7.5 GB, else micro-batch 1 with accum 3).
- **Prediction (before the run; cold VAL 1000, dt 0).**

  | layer | ctl (DualBEV) | bevfeat | predicted dual |
  |---|---|---|---|
  | lane divider@0.5 | 0.191 | 0.351 / 0.345 | 0.34 (0.30 - 0.37) |
  | drivable@0.5 | 0.716 | 0.804 / 0.809 | 0.80 |
  | walkway | 0.178 | 0.339 / 0.342 | 0.33 |
  | front mAP2 (0-50 m) | 0.367 | 0.183 / 0.180 | 0.33 (0.25 - 0.38) |
  | vehicle AP@2 | 0.670 | 0.455 / 0.447 | 0.62 |
  | pedestrian AP@2 | 0.210 | 0.001 / 0.002 | 0.15 |

  Reason for "most of both": the DualBEV BEV reaches the trunk unchanged, so the object path of the control stays available; the maps need the stride-8 BEV, and t8 shows the trunk can use it.
  The weak spot is the shortcut (b): the pedestrian AP is the sensitive number.
- **Decision rule (written now; bands from the t8 / t9 cards: maps max(0.02, 2 |s0 - s1|), walkway 0.054; front mAP2 the paired bootstrap of `temp/t4/compare_front.py`, 60 draws).**
  - E1 both: lane divider and drivable >= bevfeat - 0.02 and walkway >= bevfeat - 0.054, and front mAP2 >= ctl - 0.054 -> run the seed twin (s1); if it agrees, the chain with both feature
    sources is the new reference and the target for the single encoder (E2E, plan 2.8 step 1 / t9 U and UR, initialised from BEVFusion for the maps and distilled/trained
    against the DualBEV depth for objects); report ms per frame of both image sides (dual frozen is a diagnostic arm, not the product: Swin-T + FPN + ResNet-50 + DepthNet).
  - E2 maps kept, objects between (front mAP2 in [0.22, 0.31]) -> the shared trunk dilutes the object path: next arm = branch-specific heads (object head reads the trunk of branch 1 only, or
    a stop-gradient on branch 2 for the object loss), seed 1 first only if the gain over bevfeat on mAP2 exceeds the band.
  - E3 objects back (>= ctl - 0.054) but maps <= ctl + 0.05 -> shortcut: the trunk ignores branch 2. Check the gradient norm of `cam_reduce2` / `cam_merge`, add an auxiliary map head on branch 2's
    BEV, and rerun once; if it persists, the DualBEV BEV dominates by convergence speed, and the mixing has to be at the feature level (hybrid H or t9 R).
  - E4 both mid (maps and objects each lose more than the band against their best arm) -> capacity or optimisation, not information: longer run (6000 steps) before any conclusion.
  - INCONCLUSIVE (a metric inside its band): add s1, do not adopt.
- **Kill / validity criteria.** (1) the loss rule of t7 at step 1000 (group.lines mean over steps 900 - 1000 above 1.48 while the references are 1.22 - 1.25): stop and report; (2) a NaN or a
  CUDA unknown error: resume from the last checkpoint once (as for t8); (3) a missing frame in either cache (the token sets are identical by construction: BEVFusion cache = built from
  the tokens of `cam_feats`): the dataset raises, the run stops.
- **What it does not test.** Speed of a single product encoder, a better depth for the BEVFusion side, the hybrid (BEVFusion context x DualBEV depth on one grid), lens/roll behaviour, longer training.
- **Cost.** One seed ~100 min of lane time after the build (the lane is free except `t7_rgb_s1`); the build is about 60 lines in three files owned by the lane owner (`feature_cache.py`, `model.py`,
  the config) and the gate is mine.

## 2 Log

- 2026-10-31: card written after the t8 result; nothing built or queued. Needs the lane owner's go for the option and its place in the queue relative to t9 (see the first paragraph).
