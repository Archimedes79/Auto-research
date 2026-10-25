# E55: How large is the training-seed noise? Repeated runs of the dense head recipe and of the key-point head

Status: completed

**Question:** E54 covers the sampling of the evaluation scenes but not the randomness of the training. The foundation comparisons of this loop (E49, E50, E52: differences of -0.002 ... -0.006 F1) and the main node result (one key-point head, one seed) rest on single runs. How large is the seed-to-seed variation of (a) the dense head recipe and (b) the key-point head and the node-guided graph built from it?

## Setup
- (a) Dense head (pixel-based): the reference recipe of E30 / E49 / E50 / E52 (39 frozen channels = decoder PCA 32 + 7 map logits, 4,207 training rows, 3,000 steps, the stabilised recipe) repeated with two further seeds (22, 23; the reference is seed 21). The seed changes the initialisation and the sample order only; data, features and the evaluation are identical, the threshold is chosen on VAL for every run.
- (b) Key-point head (pixel-based node heat maps) of E44 (5,000 steps, all 8,414 rows) repeated with seed 1 (E44 = seed 0). The default node insertion (conservative cut + prune, vector-level) is applied to the same graphs of the previous system with the key-point threshold fixed at 0.5 (no re-tuning) - so the second table shows how much of the E44 result is seed noise.

![architecture](E55_architecture.png)

## Numbers

**Dense head (pixel-based), reference recipe of E30 with three seeds (4,207 training rows, 3,000 steps)**

| run | seed | TRAIN_EVAL F1@1m | VAL F1@1m | TEST F1@1m | VAL P / R @1m | tau (VAL) |
|---|---|---|---|---|---|---|
| reference (E30) (e30_base2) | 21 | 0.933 | 0.872 | 0.838 | 0.902 / 0.844 | 0.5 |
| repeat (e55_seed22) | 22 | 0.932 | 0.867 | 0.837 | 0.898 / 0.838 | 0.5 |
| repeat (e55_seed23) | 23 | 0.932 | 0.868 | 0.838 | 0.898 / 0.840 | 0.5 |
| mean of 3 seeds | - | 0.933 | 0.869 | 0.838 | - | - |
| standard deviation (n = 3) | - | 0.0004 | 0.0025 | 0.0006 | - | - |
| range (max - min) | - | 0.0009 | 0.0048 | 0.0012 | - | - |
| ensemble: average of the three runs (heat and direction maps) | 21 + 22 + 23 | - | 0.871 | 0.841 | 0.902 / 0.842 | 0.5 |

**The single-seed foundation variants against the seed spread of the reference**

| variant (one seed, 21) | experiment | VAL F1@1m | TEST F1@1m | difference to the seed mean VAL / TEST | inside the seed range of the reference? |
|---|---|---|---|---|---|
| wider decoder read-out (96 PCA components) | E49 | 0.864 | 0.834 | -0.0055 / -0.0036 | no (VAL below, TEST below) |
| instance normalisation | E50 | 0.869 | 0.840 | -0.0004 / +0.0019 | no (VAL inside, TEST above) |
| GaussianCaR + CRN fusion | E52 | 0.865 | 0.836 | -0.0037 / -0.0015 | no (VAL below, TEST below) |

**Key-point head (pixel-based node heat maps), two seeds**

| key-point head | seed | VAL split / merge F1 @6 m | TEST split / merge F1 @6 m | TRAIN_EVAL split / merge | tau (VAL) |
|---|---|---|---|---|---|
| E44 (e44_nodes) | 0 | 0.786 / 0.833 | 0.792 / 0.837 | 0.955 / 0.961 | 0.6 |
| repeat (e55_nodes_s1) | 1 | 0.794 / 0.825 | 0.794 / 0.844 | 0.953 / 0.956 | 0.6 |

**Effect of the key-point seed on the graph (vector level; default cut + prune, key-point threshold 0.5 fixed)**

| graph with default cut + prune | key-point head seed | set | F1@1m | route F | split F1 3 / 6 m | merge F1 3 / 6 m | split P / R @6 m | merge P / R @6 m |
|---|---|---|---|---|---|---|---|---|
| default node insertion (e44_cons_p) | 0 (E44) | VAL | 0.899 | 0.793 | 0.454 / 0.497 | 0.545 / 0.603 | 0.739 / 0.375 | 0.821 / 0.477 |
| default node insertion (e44_cons_p) | 0 (E44) | TEST | 0.876 | 0.733 | 0.502 / 0.552 | 0.537 / 0.607 | 0.825 / 0.415 | 0.875 / 0.464 |
| default node insertion (e55_cons_s1) | 1 (repeat) | VAL | 0.899 | 0.792 | 0.457 / 0.500 | 0.533 / 0.592 | 0.745 / 0.377 | 0.810 / 0.467 |
| default node insertion (e55_cons_s1) | 1 (repeat) | TEST | 0.876 | 0.731 | 0.507 / 0.558 | 0.530 / 0.610 | 0.832 / 0.420 | 0.876 / 0.468 |

## Result images

![Pixel F1 @1 m of the dense head on VAL and TEST: three seeds of the reference recipe (red) and the single-seed foundation variants.](E55_e55_seeds.png)
*Pixel F1 @1 m of the dense head on VAL and TEST: three seeds of the reference recipe (red) and the single-seed foundation variants.*

## Findings
- The dense head recipe is reproducible to a few thousandths: VAL F1 @1 m 0.872 / 0.867 / 0.868 (mean 0.869, standard deviation 0.0025, range 0.005), TEST 0.838 / 0.837 / 0.838 (standard deviation 0.0006), TRAIN_EVAL 0.933 / 0.932 / 0.932. With only three runs the standard deviations are crude, but the stabilised recipe of E37 does what it was built for: no seed derailed and the spread is small compared with the differences discussed in this loop. Averaging the heat and direction maps of the three runs (a seed ensemble, the cheapest form of variance reduction) gives VAL 0.871 / TEST 0.841, i.e. +0.002 / +0.003 over the mean of the single runs and +0.003 over the best single run on TEST: a gain of the size of the seed noise that does nothing about the seen-to-unseen shift.
- The single-seed foundation variants in perspective (second table): the wider read-out of E49 is -0.0055 (VAL) / -0.0036 (TEST) below the three-seed mean, which is outside the seed spread (a small but real loss); instance normalisation (E50) is -0.0004 / +0.0019 (inside on VAL, slightly above on TEST); the CRN fusion (E52) is -0.0037 / -0.0015. None of the three improves the reference by more than 0.002, and the largest effect is a 0.006 loss: the conclusion 'no gain from a better read-out of the frozen foundation' holds with the seed noise taken into account. The resolution of the single-seed comparisons of this loop is about 0.005 F1 on VAL.
- Key-point head: the second seed gives VAL / TEST split F1 (6 m) 0.794 / 0.794 and merge F1 0.825 / 0.844 against 0.786 / 0.792 and 0.833 / 0.837 for seed 0 (spread at most 0.008). The default node insertion on the graph reproduces the E44 result: TEST split / merge F1 0.558 / 0.610 against 0.552 / 0.607, route F 0.731 against 0.733, point F1 0.876 in both (VAL 0.500 / 0.592 against 0.497 / 0.603, route F 0.792 against 0.793). The difference between the two seeds (0.006 / 0.003 on TEST) is about one thirtieth of the gain over the previous system (+0.21 / +0.23): the headline of E44 is not a seed artefact.

## Limits
- Two repeats only (three runs for the dense head, two for the key-point head): the standard deviations are crude. The dense-head repetition is the reduced recipe of E30 (4,207 rows, 3,000 steps), not the final dense head of E37 (8,414 rows, 6,000 steps), and the refiner, the connectors (E53 / E56), the registration (E48) and the link head (E45) were not repeated. The seeds change the initialisation and the sample order but not the data or the scenes; the foundation (frozen) is the same in all runs.
