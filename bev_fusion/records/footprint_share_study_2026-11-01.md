# Footprint share of the GT objects in the fusion area: the threshold study for D38 (b) (2026-11-01)

Question: D38 (b) (`sensor_coverage_2026-11-01.md`) decides that an object counts as observed by the share f of its footprint inside the fusion area, with two thresholds that "come from the data (histogram of f over the objects, a sensitivity table)". This is that study, made by the BEV system chat on stored data (no GPU, no model, no change of any code). Status: a proposal for Mirko; nothing is decided or built.

## Method

- f = the share of an object's footprint (the four corners of its GT box, ego frame at the camera time) that lies inside the fusion area. The fusion area = the cells of the GT layers `camera` (ground-plane projection into CAM_FRONT) x `radar` (the fan) x not `occluded` (hidden behind another box: autogt's 2D ray rule, which takes every box as infinitely high). Samples on a 0.125 m grid inside the footprint are looked up in the 0.5 m masks; samples outside the 60 x 52 m grid count as not observed.
- Data: the GT cache of signature 15efbde2d1 (GT version v5): the 1744 frames with objects of the official validation scenes it holds (they lie in the 82 scenes of the cold VAL protocol, which scores 1000 frames of them), objects with a part inside the grid: 18700 (11451 vehicles, 6575 pedestrians, 674 two-wheelers), 2670 of them cut by the grid edge.
- Check (layer 0): known answers on a synthetic mask (inside 1, outside 0, half over the border 0.5, cut by the far edge 0.5, a rotated box 1); a pedestrian sliding over a mask border has a share error of at most 0.10 (mean 0.04; the sample grid gives 5 samples across 0.6 m), a car at most 0.03. Not yet checked against a second implementation: the GT chat's v6 helper for the footprint share must reproduce these tables.
- The tables were made by a throw-away script that is not kept in main; the production function is that v6 helper, and this study is to be re-run with it.

## The tables (made from the object list of the extraction; f = the share inside the fusion area, all objects with a part inside the grid)


18700 objects: pedestrian 6575, two_wheeler 674, vehicle 11451; frames in the csv: 1744
objects fully inside the grid (in_grid = 1): 16030, cut by the grid edge: 2670

### Histogram of f (share of the footprint inside the fusion area; samples outside the grid count as not observed)

| f | all | pedestrian | two_wheeler | vehicle |
|---|---|---|---|---|
| 0 | 9393 | 4044 | 357 | 4992 |
| (0, .1) | 733 | 72 | 11 | 650 |
| [.1, .2) | 610 | 86 | 9 | 515 |
| [.2, .3) | 508 | 92 | 13 | 403 |
| [.3, .4) | 462 | 102 | 5 | 355 |
| [.4, .5) | 479 | 92 | 7 | 380 |
| [.5, .6) | 515 | 138 | 9 | 368 |
| [.6, .7) | 449 | 175 | 10 | 264 |
| [.7, .8) | 642 | 309 | 38 | 295 |
| [.8, .9) | 1058 | 518 | 81 | 459 |
| [.9, 1) | 2724 | 302 | 77 | 2345 |
| 1 | 1127 | 645 | 57 | 425 |

exactly 0 or exactly 1: 10520 of 18700 (56.3 %); strictly between: 8180 (43.7 %)

### One threshold t (the hard rule: care when f >= t, else don't care): objects kept

| t | all | pedestrian | two_wheeler | vehicle | change vs the neighbour t |
|---|---|---|---|---|---|
| 0.05 | 8870 (47.4 %) | 2500 | 312 | 6058 |  |
| 0.1 | 8574 (45.9 %) | 2459 | 306 | 5809 | 296 objects leave |
| 0.25 | 7688 (41.1 %) | 2327 | 291 | 5070 | 886 objects leave |
| 0.4 | 6994 (37.4 %) | 2179 | 279 | 4536 | 694 objects leave |
| 0.5 | 6515 (34.8 %) | 2087 | 272 | 4156 | 479 objects leave |
| 0.6 | 6000 (32.1 %) | 1949 | 263 | 3788 | 515 objects leave |
| 0.75 | 5256 (28.1 %) | 1634 | 236 | 3386 | 744 objects leave |
| 0.9 | 3851 (20.6 %) | 947 | 134 | 2770 | 1405 objects leave |
| 0.99 | 1184 (6.3 %) | 645 | 57 | 482 | 2667 objects leave |

### Two thresholds: don't care below t_low, partial between, care from t_high (counts of all objects)

| t_low | t_high | don't care | partial | care |
|---|---|---|---|---|
| 0.1 | 0.5 | 10126 | 2059 | 6515 |
| 0.1 | 0.7 | 10126 | 3023 | 5551 |
| 0.1 | 0.8 | 10126 | 3665 | 4909 |
| 0.1 | 0.9 | 10126 | 4723 | 3851 |
| 0.1 | 0.95 | 10126 | 5729 | 2845 |
| 0.2 | 0.5 | 10736 | 1449 | 6515 |
| 0.2 | 0.7 | 10736 | 2413 | 5551 |
| 0.2 | 0.8 | 10736 | 3055 | 4909 |
| 0.2 | 0.9 | 10736 | 4113 | 3851 |
| 0.2 | 0.95 | 10736 | 5119 | 2845 |
| 0.3 | 0.5 | 11244 | 941 | 6515 |
| 0.3 | 0.7 | 11244 | 1905 | 5551 |
| 0.3 | 0.8 | 11244 | 2547 | 4909 |
| 0.3 | 0.9 | 11244 | 3605 | 3851 |
| 0.3 | 0.95 | 11244 | 4611 | 2845 |
| 0.5 | 0.7 | 12185 | 964 | 5551 |
| 0.5 | 0.8 | 12185 | 1606 | 4909 |
| 0.5 | 0.9 | 12185 | 2664 | 3851 |
| 0.5 | 0.95 | 12185 | 3670 | 2845 |

### The rules in use today against f

- centre cell in the fusion area (c_fus = 1): 6646 objects; of them with f < 0.5: 259, with f < 0.1: 1
- centre cell NOT in the fusion area: 12054 objects; of them with f >= 0.5: 128, with f >= 0.9: 1
- centre cell in the camera cells (c_cam = 1): 9649; f counts: 3147 below 0.5
- `observed` flag (a lidar or radar point in the box): 17153 observed, 1547 unobserved; unobserved with f >= 0.5: 350; observed with f < 0.5: 10988
- hidden behind a box (f_fusnoocc - f_fus): objects that lose at least 0.1 of f by the occlusion: 5542; at least 0.5: 3005

### By range (care at t = 0.5, share of the objects of the bin)

| range m | objects | f >= 0.5 | f = 0 | 0 < f < 0.5 | 0.5 <= f < 1 | f = 1 |
|---|---|---|---|---|---|---|
| 0-20 | 6193 | 18 % | 73 % | 9 % | 15 % | 3 % |
| 20-40 | 8345 | 37 % | 49 % | 14 % | 31 % | 6 % |
| 40-60 | 3842 | 58 % | 20 % | 22 % | 46 % | 11 % |
| 60-200 | 320 | 22 % | 18 % | 60 % | 18 % | 3 % |

## What the tables say

1. **f is broad, not U-shaped, and has no gap.** Exactly 0: 9393 objects (50.2 %); exactly 1: 1127 (6.0 %); strictly between: 8180 (43.7 %). From f = 0.1 to 0.7 every 0.1 bin holds 449 - 610 objects, then the bins rise (642 at [0.7, 0.8), 1058 at [0.8, 0.9), 2724 at [0.9, 1)). A threshold cannot be read from a valley of the histogram; it needs a rule, or the detection quality as a function of f (not measured here).
2. **The centre cell and f >= 0.5 are almost the same rule.** Of the 6646 objects whose centre cell is in the fusion area, 259 have f < 0.5; of the 12054 whose centre cell is outside, 128 have f >= 0.5: 387 objects (2.1 %) differ. A straight border splits an object in half exactly when its centre lies on it. The footprint share matters at other thresholds and where the area is not a straight edge, not at 0.5.
3. **Occlusion is the large effect.** 5542 objects (29.6 %) lose at least 0.1 of f because other boxes hide them, 3005 (16.1 %) lose at least 0.5. The occlusion rule is autogt's 2D one with unlimited box height: it overstates the hiding of a pedestrian behind a low car.
4. **Near objects are mostly outside.** The share of objects with f >= 0.5 is 17.9 % at 0 - 20 m (6193 objects), 37.3 % at 20 - 40 m (8345) and 57.9 % at 40 - 60 m (3842): the narrow wedge of the camera close to the car leaves parked cars and pedestrians beside it out. By class (f = 0 / f >= 0.5): pedestrians 61.5 % / 31.7 %, two-wheelers 53.0 % / 40.4 %, vehicles 43.6 % / 36.3 %.
5. **The `observed` flag is not a visibility rule for this rig.** 10988 objects with a lidar or radar point have f < 0.5, and 350 of the 1547 unobserved objects have f >= 0.5 (inside the fusion area, but no point in the box: their position rests on the annotation alone).
6. **The cost of moving a threshold is moderate.** Around 0.5 a step of 0.1 moves 479 (0.4 -> 0.5) or 515 (0.5 -> 0.6) objects, 2.6 - 2.8 % of all; the steps get larger above 0.6 (744 up to 0.75, 1405 up to 0.9) because the mass of the nearly visible objects sits there.

## Proposal for the simple technique now (not decided)

- **Three states by f:** care f >= 0.5 (6515 objects, 34.8 %), partial 0.1 <= f < 0.5 (2059, 11.0 %), don't care f < 0.1 (10126, 54.1 %). 0.5 is the "centre in view" rule (it agrees with it for 97.9 % of the objects) and the KITTI overlap rule; 0.1 drops grazing contacts (733 objects with 0 < f < 0.1) that nobody can be asked to find. The data offers no other natural value, so these are a convention backed by the sensitivity table, not a result.
- **Evaluation:** the headline counts care objects only, scored on the fusion area (D37); the partial band is reported next to it, never inside; a detection is ignored when half or more of its own footprint lies in a don't-care area (the same f computed on the detection); the stored numbers keep their camera-field-of-view label and are not mixed with new ones.
- **Training, simple version:** objects with f < 0.5 get no loss (neither positive nor negative), as unobserved boxes already do today; the fused map and object heads get no loss on cells outside the fusion area (D38: neither positive nor negative); the aux heads keep their own sensor area and become switchable (D38 reading a). That is a change of regime: the model never sees a label for lanes outside the camera wedge, so it needs a matched control (arm A today's loss, arm B this hard rule, same budget, two seeds, scored on the fusion area).
- **Later, as decided:** the soft weight in f, the radar class prior, the three-arm test.
- **Order of work after Mirko's go:** (1) the footprint-share helper in autogt (the GT chat's v6) reproduces these tables; (2) evaluation: the three states in `metrics.py` and the two-sided ignore, the headline on the fusion area; (3) training: loss masks as an option of `LossConfig` (default off, so every existing run is reproduced), then the matched pair of arms at the 3000-step recipe.
- **Before the thresholds are frozen (not done):** the recall of a trained model as a function of f, a GPU diagnostic run over the cold VAL frames, about 15 minutes. If a model already finds most objects at f = 0.3, the care threshold is too high; if it finds almost none at f = 0.7, too low.

## Open for Mirko

1. Accept (0.1, 0.5) as the first thresholds, or first the recall-by-f check?
2. Keep autogt's 2D occlusion rule (unlimited box height) in the fusion area, or soften it for low boxes?
3. Training version: hard rule in training at f >= 0.5 (above), or the evaluation states only first and the training masks after the matched control is planned?
