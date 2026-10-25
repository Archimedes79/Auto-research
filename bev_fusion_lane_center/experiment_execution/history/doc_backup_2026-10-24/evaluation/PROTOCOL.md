# Evaluation protocol

Setting of every number: exact ego poses, simulated ADASIS-like SD map, nuScenes (frozen GaussianCaR features), scene-disjoint splits. State this whenever numbers leave the project.

## Data and splits
- Frames: TRAIN 8,414, VAL 873, TEST 833 (`training/data/p16_common.py`: `N_TRAIN, N_VAL, N_TEST`); scene lists in `data/splits.json`. TEST = 21 unseen scenes. The splits are scene-disjoint, **not** location-disjoint (open check: overlap of VAL / TEST locations with TRAIN, 5 m rule).
- VAL subset for early rungs of a search: 250 frames, `p16_common.val_subset()`; the full VAL (873) from rung 2 on. Never compare numbers from different frame sets.
- Out-of-fold dense heads (`e37_fold0/1/2`) give the inputs for perturbed TRAIN rows (no leakage of the head's own training frames).

## Selection and noise
- Select on VAL only. Differences below the noise band (0.010 in S) are ties.
- Any comparison of route heads needs **two seeds**; the seed spread of the route head is 0.05-0.09 in route success, larger than most effects.
- Intervals: scene-level bootstrap (`boot_eval.py`, `p11_eval.py`, 2000 draws, paired differences against a reference).

## TEST discipline
- TEST is used once per finished design: the winner and its control, with the user's explicit OK, and the use is logged in `TEST_USAGE_LOG.md`. Nothing is tuned on TEST. Exploration after a freeze is validation-only (loop 16b).

## Metric set (see `sparse_lanes/README.md`, `lane_grid/README.md`)
Graph: F1 at 0.5 / 1.0 / 1.5 m, J (split / merge F1 at 6 m), vertices per 100 m, FL (false length), S (composite), lanes found, direction accuracy. Frontier: F1@Sp17 (best F1@1 with <= 17 vertices per 100 m), FL@R85. Route: route success of the route head (assigned lane on the route), route F. Planner: consumer L2 error at 1-3 s, deviation rate. Dense maps: point P / R / F1 at 0.5 / 1.0 / 1.5 m.

## Failure and map protocols
- Failure series (loop 5 E57, loop 17 E193+): per scene, first half clean features, second half features of one failure type, 1 s between frames. Types: night (`dark`), `blur`, `cam_front3_off` (three front cameras off), `fog`, `radar_off`, `extr3` (extrinsic error), `cam_front_off`; E209 holds out night and blur from the training of the builder.
- SD map: ADASIS-like simulator (E77); map error = scene-constant world-frame translation with sigma (m) and yaw 2 sigma (deg) plus per-frame noise (E61 / E87); modes exact, none, wrong, error 1-2 m. Map benefit is judged against the matched no-map reference (E208).

## Timing protocol
Milliseconds per frame on an idle GPU (RTX 3060), batch 1, cached foundation features, after warm-up, median of the frames (`p16_timing.py`, `p17_latency.py`); an optimised path must reproduce the graphs of the reference path (`p16_fast_check.py`: identical graphs in N / N frames).
