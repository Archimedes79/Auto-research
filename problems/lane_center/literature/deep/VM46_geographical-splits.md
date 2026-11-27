# VM46 - Geographically disjoint splits (Lilja et al.): full-text extraction

| field | value |
|---|---|
| paper | Localization Is All You Evaluate: Data Leakage in Online Mapping Datasets and How to Fix It. Adam Lilja, Junsheng Fu, Erik Stenborg, Lars Hammarstrand. CVPR 2024 (venue as given in the repository README; the arXiv page names none). arXiv 2312.06420, v1 2023-12-11, v2 2024-04-05 (current). Links opened: https://arxiv.org/abs/2312.06420, https://arxiv.org/html/2312.06420v2, https://github.com/LiljaAdam/geographical-splits |
| axis, tier, K ids | A9 (evaluation, data, splits), tier 1 core; no K id yet (a location-overlap audit is proposed in section 5, to be registered before use) |
| level-1 note(s) | [VM46](../papers/VM46_geographically-disjoint-splits.md) (home), [SN03](../papers/SN03_localization-is-all-you-evaluate.md); split context also in [VM01](../papers/VM01_streammapnet.md) and its extraction [VM01](VM01_streammapnet.md) |
| code extraction | [geographical-splits](../code/geographical-splits.md) (Apache-2.0: adaptation allowed with attribution) |
| sources read | arXiv abstract page (v2); arXiv HTML v2 in nine separate queries (structure and tables list, leakage definition and Table 2, Section 3, Section 4 and Table 4, Appendix A and Table 10, Tables 1/3/6, Section 4.1 and Appendix B, Tables 5/7/8/9/12 and conclusion, wording on the original test set); repository pages: README (root and `near_extrapolation_splits`), file trees of the root, `src`, `src/nuscenes`, `src/argoverse2`, both `zone_coordinates` / `conversions` folders, the split folders, `requirements.txt`, `LICENSE`, commit list; raw files `src/nuscenes/generate_geo_split.py`, `utils.py`, `constants.py`, `generate_city_splits.py`, `convert_pkls.py`, `zone_coordinates/nusc_geo.yaml`, `src/argoverse2/generate_geo_split.py`, and the three nuScenes scene lists (`scenes` version). Local files: `data/splits.json`, `data/README.md`, `deep/VM01_streammapnet.md`. All 2026-10-25. The arXiv PDF was not used (binary); no supplement exists beyond the appendices A-D of the HTML |
| reading marks | every statement about the paper and the repository is **digest**: all pages came through the summarising fetch tool; where the tool returned code lines verbatim this is said, and those excerpts are still tool output, not a raw page. The HTML text had doubled digits (the tool saw "55 m" for 5 m, "8080 %" for 80 %); I read each value once and the abstract page and the README both say 5 m. Statements about our own splits (`data/splits.json`) are **checked** (file read). The cross-tabulation of our scenes against Lilja's scene lists in section 5.3 is **my own computation on digest-level lists** (lists as transcribed by the fetch tool, train list only for scene numbers up to 400, the only ones that can meet our scenes); recompute it from the raw files or from the polygons before it enters a report. Statements marked (memory) were not seen on any page |
| extracted by, date | extraction sub-agent (Claude Sonnet 5.5) of the literature session, 2026-10-25 |

## 1. Problem and claim
- Online map estimation (vector or raster maps from onboard sensors, no pre-built map) is trained and tested on nuScenes and Argoverse 2 (AV2) with the official splits. Those splits reuse the same places in train, val and test, so a model can score by recalling a memorised map of the area (the paper's word is localisation) instead of reading the scene.
- Measure of the problem: the share of val / test samples that lie within 5 m of some training sample. Original splits: nuScenes 79.4 % (val) and 85.5 % (test), AV2 45.0 % and 41.9 %. Proposed fix: geographically disjoint splits, a "Near Extrapolation" split (same cities, disjoint regions, usable as a drop-in) and a "Far Extrapolation" cross-validation (whole cities / map areas held out).
- The claim that matters for us: on the Near split every method falls sharply (test mAP of MapTRv2 on nuScenes 72.7 to 26.2, an average drop above 35 mAP on nuScenes and 12 mAP on AV2), the method ranking is largely kept, and conclusions about lifting methods and depth supervision shrink. Nothing in the paper concerns a frozen pre-trained foundation or lane graphs.

## 2. Method as built
This is an evaluation protocol; there is no model, loss or inference rule of the authors'. The rows of the template are filled accordingly.

- **Pipeline / architecture (protocol chain):**
```
  positions of all samples (ego pose, global x y, per map)
    -> leakage ratio r(tau) of the official split  (val / test samples with a train sample within tau = 5 m)
    -> hand-drawn zone polygons per map  ->  every sample labelled train / val / test by position  (Near)
       or whole cities / map areas as train and val folds  (Far, cross-validation, no test set)
    -> re-train the published methods with their original configs on the new pkl files, report mAP / IoU
    -> re-check design choices (lifter, depth supervision, sample density, buffer of 60 m)
```
- **Inputs, outputs:** input = sample positions (the repository takes the x, y of the ego pose attached to the LIDAR_TOP sample data of each keyframe, global frame of the map). Output = split lists (txt and pkl), see "Labels".
- **Equations:** the paper prints none for the ratio. The natural reading, which I use in section 5 and which is **my formulation, not the paper's**: r(tau) = |{ q in VAL : min over p in TRAIN of ||x_q - x_p|| <= tau }| / |VAL|, with tau = 5 m and x the 2-D ego position; one value per val / test set, per sample, not per area. Which positions are compared (ego pose or sample centre), the metric (Euclidean presumably) and whether heading enters are **not stated** in the text (checked twice); the repository only has helper code that loads precomputed "closest mapping" pickles with a range threshold in metres and a heading threshold in degrees, no script that creates them (see the code extraction).
- **Losses:** none of their own. The methods keep their original losses (Hungarian matching inside the vector methods).
- **Labels / ground truth (the split itself):**
  - *nuScenes Near:* 70 / 15 / 15 train / val / test; sizes 28k / 6k / 6k samples (Table 1, same as the original counts). Regions are drawn by hand: per map (singapore-onenorth, singapore-hollandvillage, singapore-queenstown, boston-seaport) a few polygons in map x, y metres for val and for test, train = everything else (code: val polygons are tested first, then test, else train). Val: 1 + 2 + 1 + 2 = 6 polygons, test: 2 + 1 + 2 + 3 = 8 polygons, 4 to 6 vertices each (derived from the per-map counts of the digest). The text says the regions were chosen from map attributes so that zone classes (residential, commercial, industrial), road-object class frequencies, weather (clear / rain) and time of day (day / night) are kept equal across the sets and every city appears in every set; the code does not use any map attribute (map layers are loaded only to draw figures), so the balancing was done by eye when drawing the polygons. The number of regions and the balancing procedure are not described beyond this.
  - *Pool:* the code pools the keyframes of v1.0-trainval and v1.0-test (about 40,000 samples, 1,000 scenes, Boston and Singapore) and assigns each by position; the original test scenes are therefore spread over all three new sets, they are not the new test set (see "Corrections"). A switch removes the original test scenes (the paper mentions this for multi-task networks that need object labels).
  - *Sequences that cross a zone border:* two published versions. `samples`: every sample carries its own assignment, so a scene is cut into two partial sequences at the border (cut across the driving direction, as few cuts as possible; temporal models stay usable). `scenes`: the crossing scenes are removed and the lists hold scene names (digest counts: train 657, val 124, test 137 = 918 of 1,000 scenes).
  - *Buffer:* none in the construction. Appendix A argues that predictions cover about 30 m ahead and behind, so maps overlap near borders, and shows that removing val / test samples within 60 m of any training sample changes the result negligibly (Table 10).
  - *AV2 Near:* 110k / 24k / 24k samples (Table 1), assigned per log: a log goes to val or test only if its whole trajectory lies inside a polygon of its city, else to train (code, digest). No weather or time-of-day attributes exist in AV2, so the balancing there is qualitative (zone classes). Intra-set correlation (consecutive frames, standing at junctions) is acknowledged and left to future work.
  - *Far Extrapolation:* whole cities (nuScenes: map areas) as folds, train about 70 %. nuScenes Far-A 30k train / 9k val, Far-B 31k / 9k; AV2 Far-A 110k / 46k, Far-B 101k / 55k, Far-C 101k / 55k; no test set, report the cross-validation mean. Boston holds 55 % of the nuScenes samples, Miami and Pittsburgh 35 % each of AV2 (digest wording). The city assignment of Table 3 was not readable in the digest; the code defines eight map groups (hollandsvillage, queenstown, boston, singapore = all three areas, boston+onenorth, hollandsvillage+queenstown, boston+hollandsvillage+queenstown, onenorth; the key is spelled with an s), and together with the counts this suggests nuScenes fold A = train Boston + One North, val Holland Village + Queenstown, fold B = train Boston + Holland Village + Queenstown, val One North (my inference, unconfirmed).
- **Training recipe:** the authors re-run HDMapNet, VectorMapNet, MapTR, MapTRv2 (camera and camera + lidar), StreamMapNet and the segmentation baselines GKT, CVT, IPM with the original configs and only the split files replaced (README: replace the pkl paths). Epochs, seeds, GPUs and the evaluation range per method are not stated in the text I could read. Backbones ResNet-50 and ResNet-18 (vector methods), EfficientNet-B4 (segmentation), SECOND for the lidar row. Metric: mAP over Chamfer thresholds 0.5 / 1.0 / 1.5 m for vector methods, IoU for segmentation. One run per row, no seeds shown.
- **Inference:** not applicable.

## 3. Numbers
All values are author-reported (digest of the HTML tables, table numbers as printed). They are on the authors' protocol: do not compare them with our F1 / S, nor with other papers' splits.

**Table 2, share of val / test samples within 5 m of a training sample**

| split | nuScenes val | nuScenes test | AV2 val | AV2 test |
|---|---|---|---|---|
| Original | 79.4 % | 85.5 % | 45.0 % | 41.9 % |
| Near Extrapolation | 0.9 % | 1.1 % | 0.0 % | 0.0 % |

Figures 6 and 7 plot the same ratio against the range (values not readable in the digest). The remaining nuScenes overlap of the Near split comes from samples next to the zone borders.

**Table 4, vector methods, mean mAP over the three classes (V = val, T = test; per-class cells were read and are in the paper)**

| dataset | method (sensor, backbone, lifter) | Orig V | Orig T | Near V | Near T |
|---|---|---|---|---|---|
| nuScenes | VectorMapNet (cam, R50, IPM) | 43.2 | 54.8 | 14.0 | 18.2 |
| nuScenes | MapTR (cam, R18, GKT) | 33.1 | 42.4 | 9.8 | 11.1 |
| nuScenes | MapTR (cam, R50, GKT) | 48.8 | 59.9 | 19.0 | 19.7 |
| nuScenes | MapTRv2 (cam, R50, LSS) | 61.5 | 72.7 | 26.7 | 26.2 |
| nuScenes | MapTRv2 (cam + lidar, R50 + SECOND, LSS) | 54.0 | 65.8 | 20.0 | 20.1 |
| nuScenes | StreamMapNet (cam, R50, BEVFormer) | 62.6 | 74.4 | 26.1 | 27.9 |
| AV2 | VectorMapNet 2D | 44.0 | 42.0 | 32.7 | 32.9 |
| AV2 | MapTR 2D | 63.6 | 62.1 | 48.0 | 48.2 |
| AV2 | MapTRv2 2D | 67.7 | 65.3 | 53.1 | 55.2 |
| AV2 | MapTRv2 3D | 64.2 | 62.1 | 50.1 | 52.6 |
| AV2 | StreamMapNet 2D | 61.7 | 60.8 | 50.7 | 52.6 |

Reading: the largest nuScenes test drop is 46.5 points (MapTRv2 72.7 to 26.2; StreamMapNet 74.4 to 27.9 gives the same difference); the pedestrian-crossing class collapses most on nuScenes test (MapTR R50: 53.3 to 5.9; cells read in the digest). The AV2 drops (derived from the means above) are 8.2 to 13.9 points on test and 11.0 to 15.6 on val. Table 5 (segmentation IoU, mean val / test): nuScenes GKT 19.2 / 17.7 to 8.8 / 12.3, CVT 24.4 / 21.1 to 12.8 / 7.7, IPM 41.2 / 44.2 to 27.4 / 24.2, HDMapNet 31.3 / 38.8 to 17.1 / 21.2; AV2 GKT 26.3 / 20.7 to 22.5 / 17.9, CVT 28.4 / 21.9 to 24.2 / 20.5, IPM 47.5 / 42.7 to 41.5 / 37.7.

**Table 6, Far Extrapolation, cross-validation mean mAP:** nuScenes VectorMapNet 8.7, MapTR 14.9, MapTRv2 21.4, StreamMapNet 21.3 (folds A / B for MapTRv2: 21.8 / 20.9); AV2 VectorMapNet 24.0, MapTR 41.8, MapTRv2 45.5, StreamMapNet 46.6 (the VectorMapNet fold-A cells in the digest are not self-consistent, treat that row with care).

**Ablations that decide transfer**
- Sample density (Table 7, MapTRv2): dense sampling instead of keyframes only lifts nuScenes Original test mAP from 72.7 to 87.5 and val from 61.5 to 86.4, but on Near only from 26.2 to 26.9 (val 26.7 to 27.2). More frames of the same places help only when places repeat.
- Lifting method (Table 8, MapTR val, nuScenes Orig / Near): GKT 48.8 / 19.0, BEVFormer 47.8 / 20.6, LSS 50.0 / 21.0. The ordering of GKT and BEVFormer flips and all differences on Near are about 2 points.
- Auxiliary tasks (Table 9, MapTRv2 val, nuScenes Orig / Near): from none to all three tasks 56.6 / 25.3 to 61.5 / 26.7, i.e. +4.9 on the original split and +1.4 on Near (the assignment of the check marks to tasks was not readable).
- Buffer (Table 10, nuScenes Near, val / test, removing samples closer than 60 m to a training sample): MapTRv2 26.7 / 26.2 to 26.5 / 25.8, MapTR 19.0 / 19.7 to 19.0 / 19.6, VectorMapNet 14.0 / 18.2 to 14.5 / 18.3, HDMapNet IoU 17.1 / 21.2 to 17.0 / 21.4. The number of removed samples is not given.
- Extra data (Table 12, test mAP, Near to Near plus the validation samples): MapTRv2 nuScenes 26.2 to 28.7, MapTR 19.7 to 21.6, HDMapNet 13.9 to 21.8; the caption says 15 %, the body text 20 % (inconsistent).
- Qualitative (Fig. 4): a model trained on the original split draws dividers in occluded road regions that no image shows, which the authors take as map recall.

**Other numbers about the same question (not from this paper; from the StreamMapNet extraction [VM01](VM01_streammapnet.md), digest, not re-read here):** StreamMapNet's Table 6 measures overlap by area (union of 30 m discs around ego positions): nuScenes original 85 % of the val area overlaps train, StreamMapNet's own nuScenes split 11 %; AV2 54 % to 0 %. MapTR on nuScenes: 48.7 (original split, 30 m) against 20.9 (StreamMapNet split, 60 x 30 m). These are StreamMapNet's measure and splits, not Lilja's.

**Compute:** not applicable.

**Caveats:** the new splits are not "OOD by city" (Near keeps all cities); intra-set correlation is untouched; the 5 m measure ignores the width of the predicted window (they argue about 30 m ahead and behind), so a ratio near zero does not mean no map overlap near the borders (Table 10 shows this is harmless for them); mAP in the Near rows comes from single runs.

## 4. Paper versus code
Repository state: see the code extraction. Differences found:
1. **Regions.** Paper: defined manually from map attributes (zone classes, object classes, weather, time of day). Code: hand-drawn polygons in map coordinates in `zone_coordinates/nusc_geo.yaml`; no attribute is read.
2. **Pool and test set.** Paper wording (digest, ambiguous): the original test data is used because its map is available; code: v1.0-trainval and v1.0-test are pooled, the original test scenes are scattered over the new train / val / test, and a flag removes them. The level-1 note's "test = original test" is wrong (section "Corrections").
3. **The 5 m ratio.** Paper: Table 2 and Figures 6 / 7. Code: no script that computes it; only a loader for precomputed closest-mapping pickles with range and heading thresholds. As read (listings may be truncated), the evidence of the ratio is not reproducible from the repository.
4. **Argoverse 2.** Per log: the whole trajectory must lie inside a polygon (digest); the paper speaks of samples.
5. **Far Extrapolation.** Paper: city folds in Table 3 (cells unreadable in my digest); code: eight map-group keys, `singapore` among them; the output is one scene-name file per group.
6. **Sizes.** Paper: nuScenes 28k / 6k / 6k; `scenes` lists: 657 / 124 / 137 scenes (918 of 1,000), five scenes are blacklisted in the code without a stated reason.
7. **Command-line flags** declared as booleans accept any non-empty string as true (so `--f_no_original_test_scenes False` switches the option on).

## 5. Transfer to our pipeline

| question | answer |
|---|---|
| what it needs from us | Per keyframe: scene, map (location of the log) and the global 2-D ego position (x, y); nothing else. Our frames are in `data/splits.json` (scenes) with frame indices into `train_tokens.json` / `val_tokens.json`; poses are cached (registry in `data/README.md`) or readable from `v1.0-trainval_meta` that is already in `downloads/` (checked in `data/README.md`), no new download. No BEV, no image, no GPU. |
| what it gives us | A9: (1) a leakage measure comparable to a published number (5 m, per sample); (2) published polygons and lists that label every nuScenes keyframe by region, Apache-2.0; (3) the evidence that scene-disjoint splits do not remove location overlap and what that does to end-to-end map models (46.5 points on nuScenes test). It does not tell what happens to a frozen map-pretrained foundation plus a light head. |
| conflicts | None in code (pure numpy / shapely). Substantive limits: (a) the frozen foundation GaussianCaR was trained by its authors on nuScenes scenes (it scores 0.932 on its own training scenes against 0.820 on our TEST, OVERVIEW A1) and cannot be made location-disjoint without retraining it; an audit measures and stratifies the exposure, it cannot remove it. (b) Lilja's zones need a retrain of every head to be used as a split (section 5.3). (c) Their pool contains v1.0-test, which we do not have. |
| reuse mode | Idea + data: re-implement the ratio (20 lines); reuse the polygons of `nusc_geo.yaml` as data for an alternative zone label; Apache-2.0 allows adapting code with attribution. |
| effort | S for the audit (hours, section 5.1); M for a zone re-split with retrained heads (days, 2 seeds per arm); L for anything that needs a location-disjoint foundation. |
| expected gain and risk | Closes open item "location-overlap check" of OVERVIEW section 2 A9. Gain: a number for the share of VAL / TEST frames near TRAIN frames and a performance split by distance, which tells whether the seen-to-unseen gap is a location effect. Risk: the far bin may hold few scenes, so intervals are wide (frames of one scene are not independent); a small ratio does not prove absence of leakage through the foundation. |
| feeds which experiment / K id | Proposed new technique id (register first): "location-overlap audit of splits" (nearest-TRAIN distance per frame, per map, threshold curve, stratified evaluation). Related proposals already named in [VM01](VM01_streammapnet.md) section 5. |

### 5.1 Minimal audit recipe for our scene-disjoint splits (proposal; not run)
Tier A, hours (about 3 h in total: 1 h data, 1 h function plus self-test, 1 h table and note):
1. **Collect.** For all 255 scenes build a table: frame id, scene name, split, map name (from the log of the scene), x, y (global ego position of the keyframe; Lilja use the ego pose of the LIDAR_TOP sample data, our cached pose should differ by less than the motion in 50 ms, check once). Sources: `data/splits.json` plus the cached poses or the nuScenes meta tables. Optional yaw for a heading variant.
2. **Compute.** For each map separately (global coordinates of different maps are unrelated: never mix Boston and Singapore), build a k-d tree over the TRAIN positions and take the nearest-TRAIN distance d(q) for every VAL and TEST frame. Do the same for TEST against VAL, and TEST against TRAIN plus VAL, because tuning happens on VAL and our VAL and TEST scenes alternate scene numbers (checked in `data/splits.json`: 0092, 0094 ... in VAL, 0093, 0095 ... in TEST, similarly 0268 to 0278), and consecutive nuScenes scene numbers are usually consecutive clips of one drive (memory); the TEST-to-VAL ratio is likely high. The 19 Boston scenes of the file (`locations`: boston-seaport 19, singapore-onenorth 24) are by count the scene numbers 0092 to 0110 (derived, not checked in the nuScenes metadata).
3. **Overlap measure.** Per sample, as the paper: r(5 m) = share of frames with d <= 5 m. Add r at 25, 50 and 100 m because our BEV window is 100 m wide (Lilja's 60 m cell and their 60 m buffer correspond to a 60 m window; the equivalent buffer for us is about 100 m) and the median and the 90th percentile of d. Report per map.
4. **Self-test (layer 0 / 1).** A synthetic line of points with known distances; a two-map case that must give no cross-map neighbour; and a reproduction of a stored number: apply the same function to the official nuScenes 700 / 150 scene split with ego poses from `v1.0-trainval_meta` and compare with 79.4 % (val, 5 m, Table 2). A deviation of a few points would be explained by the pose or sample definition; a large one means the paper measures something else (heading, range) and the audit must say so.
5. **Stratified result (cheap, no retraining).** Take the stored per-frame metrics of the current champion and control on VAL and TEST, assign every frame to a distance bin (for example d <= 25 m, 25 to 100 m, > 100 m, bins chosen before looking), and report S, Q and F1@1 per bin with the paired scene bootstrap (`evaluation/sparse_lanes/bootstrap.py`) and the noise band 0.010. TEST only inside the existing once-per-finalist rule and its log; use VAL for the exploratory look.
6. **Decision rule to write before computing:** if the far bin (d > 100 m) is worse than the near bin by more than 0.010 in S with an interval that excludes zero, report the far-bin number next to the headline and consider tier C; if the difference is inside the band, state "no location effect detectable at this sample size" together with the bin sizes. Also write down a predicted value of r(5 m) first (project rule); the paper's 79 % / 85 % were for 700 training scenes and cannot be transferred, because our TRAIN has 212 scenes.

Tier B, about +4 h: **ground-truth overlap**. For each VAL frame, share (by length) of its ground-truth lane centre-lines that lies within 1.0 m of a ground-truth centre-line seen in any TRAIN frame (transform the cached `graph_gt` graphs into the map frame with the poses; one k-d tree over resampled TRAIN centre-line points per map). This is closer to "the map could have been memorised" than a pose distance and is independent of the window size.

Tier C, days: a location-disjoint re-split with retrained heads (section 5.3).

**What to report:** one table (pair, number of frames, r at 5 / 25 / 50 / 100 m, median and p90 of d, per map), a CDF figure of d, the list of scenes with d > 100 m, the per-bin S / Q / F1 with intervals, the Lilja Table 2 values labelled as a different split and size, and a sentence that the foundation's own exposure is outside the audit. Say VAL or TEST and the denominator in every table row.

### 5.2 What of the paper's other findings carries over
- End-to-end vector models (here the image encoder, decoder and matching are trained on 28k samples) lose 20 to 46 mAP; whether a frozen foundation with a 0.7 M-parameter route head and a light graph builder behaves similarly is open. Memorising a map needs capacity and exposure; the heads have little capacity, the foundation has its exposure.
- The paper's ranking-preserved result (rankings stay largely, conclusions about lifters and depth shrink) argues for comparing our arms inside one split, as we do, and for treating small differences with care; it also says an ablation gain measured on a leaky split can shrink on a clean one, relevant for the gains of a few thousandths of S that we use as decisions.
- The dense-sampling result (Table 7) is a caution for any plan to add non-keyframe sweeps to training: on a leaky split they look like a large gain.

### 5.3 Can the published lists be reused for our 255 scenes?
Yes as labels, partly as a split. What I computed from digest-level lists (own computation, recompute before use): of the scenes of `data/splits.json`, Lilja's `scenes` version puts

| our set | scenes | in Lilja train zone | in Lilja val zone | in Lilja test zone | not listed (crossing, blacklisted) |
|---|---|---|---|---|---|
| TRAIN | 212 | 143 | 22 | 29 | 18 |
| VAL | 22 | 16 | 1 | 1 | 4 |
| TEST | 21 | 13 | 5 | 1 | 2 |

- 29 of our 43 VAL / TEST scenes lie in the zone that Lilja assigns to training, 8 in a zone they hold out, 6 are not listed. This is not a measure of overlap with our own TRAIN (that is the audit of section 5.1); it shows that most of our VAL / TEST scenes are in areas that a region-disjoint evaluation would not hold out.
- Using the zones as a split of our 255 scenes would give about 172 / 28 / 31 scenes (train / val / test zone; 24 not listed), roughly 40 frames per scene (8,414 / 212 = 39.7). The held-out zones consist mostly of scenes that are TRAIN today (51 of the 59 listed scenes in the two zones), so every head, builder and route model would be retrained, with two seeds, and the foundation would still have been trained on those scenes. Hence: use for stratification and audit now (label all frames by applying the 14 polygons of `nusc_geo.yaml` to our ego positions, which gives frame-level labels and avoids the scene lists); use as a split only as a deliberate, separately budgeted experiment.
- The sample-level lists (`samples` version: token to set) and the pkl files need a download (small files, the names are in the code extraction); ask first. The polygons can be read from the YAML page and typed in without any download.
- Boundary scenes: 24 of our scenes are absent from the `scenes` lists (cut or blacklisted; scene-0064 is among the blacklisted ones and is in our TRAIN), which the polygon route avoids.

## 6. Not found / open
- Which positions and which distance the 5 m ratio uses, whether the same scene is excluded, whether heading enters; the script that computes it; the values behind Figures 6 and 7.
- Training details of the re-runs (epochs, seeds, GPUs, evaluation range), the number of samples removed by the 60 m filter, the number and the exact selection of regions, the balancing procedure.
- Table 3 (city assignment of the folds) and one row of Table 6 were not readable; the Table 9 task labels were not readable.
- The sample-level txt and pkl files and the Argoverse polygons were not read. The PDF and any supplement beyond the HTML appendices were not seen.
- Why five scenes are blacklisted in the code.
- Not re-read here: MapLightning's 2.1 % (StreamMapNet split) and 0.9 % (Near) under the 5 m criterion and the list of best results on the StreamMapNet split quoted in the level-1 note.

## 7. Our experiments that touch it
None. Our splits are scene-disjoint by design; no location-overlap number exists. The nearest related number is the seen-to-unseen gap of the foundation (0.932 on its own training scenes, 0.820 on TEST, OVERVIEW A1), which mixes frame exposure and location exposure. The audit of section 5.1 would split the two.

## Corrections to the level-1 note (and to wording that cites it)
1. **VM46 and SN03: "test = original test set / original test set kept".** In the code the keyframes of v1.0-trainval and v1.0-test are pooled and assigned by position; the original test scenes land in all three new sets (their scene numbers are spread over the new train, val and test lists in the repository; memory for which numbers are official test scenes). The new test set is not the original one and the new val set is not "carved from the original training data". Only the paper's own sentence is ambiguous (digest).
2. **VM46: "regions defined by hand from map attributes".** They are hand-drawn polygons in map coordinates (14 polygons on 4 maps); the code reads no map attribute.
3. **VM46: "sample counts of the Far folds not read".** Now read: nuScenes Far-A 30k / 9k, Far-B 31k / 9k; AV2 110k / 46k, 101k / 55k, 101k / 55k (no test set).
4. **SN03: repository "named in a search result, not opened".** Opened: it contains the lists (txt and pkl), the polygons, the generation scripts, converters for MapTR, MapTRv2 and the MARS benchmark (VectorMapNet / HDMapNet); licence Apache-2.0 (the LICENSE file was added on 2025-07-12, after the paper).
5. **The 85 % claim.** VM46's own wording (79.4 % val / 85.5 % test per sample within 5 m) is right. The shorthand "85 % of the nuScenes validation locations also occur in training" (OVERVIEW A9, curation `why`) mixes two measures: 85.5 % is Lilja's test figure (val 79.4 %), and the 85 % of validation area is StreamMapNet's disc-union measure (30 m radius). Use "79 % of val and 86 % of test samples within 5 m of a training sample (Lilja, original 700 / 150 / 150 scenes)".
6. **The 48.7 to 20.9 claim** (OVERVIEW A9, MapTR) is not in this paper. It is StreamMapNet's pair: 48.7 (original split, 30 m, its Table 4) and 20.9 (its own split, 60 x 30 m, Table 2), and that split still has 11 % area overlap. Lilja's MapTR (R50, GKT) values are 48.8 to 19.0 on val and 59.9 to 19.7 on test.
7. **SN03: "average drop > 35 mAP on nuScenes"** is right (the paper adds 12 mAP for AV2); "lifter ranking flips (GKT 19.0, BEVFormer 20.6, LSS 21.0)" is right for the Near val numbers, and only the GKT / BEVFormer order flips (Original: LSS 50.0, GKT 48.8, BEVFormer 47.8).
8. **Verified as stated in the notes:** the Table 2 values, the Near overlaps 0.9 % / 1.1 % / 0 %, the Table 4 and Table 5 values quoted, Far CV mean 21.4 (MapTRv2, nuScenes), 28k / 6k / 6k, 70 / 15 / 15, Boston 55 %, no buffer and the 60 m check (26.2 to 25.8 for MapTRv2 test).
