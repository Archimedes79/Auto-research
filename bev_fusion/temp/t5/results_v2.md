# t5 lane probe v2 - results (VAL: 300 fresh keyframes of 82 scenes; TRAIN: 600 keyframes; measures at 2-30 m flat-ground distance)

Recall@1 of the labelled line cells, corridor precision@1 (inside 4 cells of a label), F1 of the two (evaluation/metrics.SegMeter counts, 1 cell = 4 x 4 input pixels); threshold = best TRAIN F1; 50 epochs; difference of F1 to `ctx16` with the paired scene bootstrap (1000 draws, 95 %).

| arm | features | recall@1 | corridor precision@1 | F1 | F1 at 0 cells | d F1 vs ctx16 [95 %] | thr | TRAIN F1 |
|---|---|---|---|---|---|---|---|---|
| prior | per-pixel prior (null) | 0.499 | 0.389 | 0.437 | 0.211 | -0.012 [-0.034, +0.010] | 0.2 | 0.410 |
| ctx16 | context 80 ch, stride 16 (lift input) | 0.551 | 0.380 | 0.450 | 0.217 | - | 0.2 | 0.495 |
| fpn16 | neck 512 ch, stride 16 | 0.498 | 0.419 | 0.455 | 0.227 | +0.005 [-0.007, +0.017] | 0.2 | 0.560 |
| res8 | ResNet layer 2, stride 8 | 0.430 | 0.464 | 0.446 | 0.228 | -0.003 [-0.025, +0.017] | 0.3 | 0.642 |
| res4 | ResNet layer 1, stride 4 | 0.594 | 0.421 | 0.493 | 0.232 | +0.043 [+0.024, +0.061] | 0.2 | 0.545 |
| rgb | RGB CNN from scratch (control) | 0.557 | 0.434 | 0.488 | 0.229 | +0.038 [+0.017, +0.058] | 0.2 | 0.563 |

Control gate (rgb F1 - prior F1 >= 0.05 with the interval excluding 0): +0.050 [+0.021, +0.077] -> PASSES.
Labelled line cells at 2-30 m: 25937 in 300 frames (mean 86 per frame); frames without one: 30.
