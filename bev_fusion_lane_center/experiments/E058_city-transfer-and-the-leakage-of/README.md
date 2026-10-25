# E58: City transfer and the leakage of near-duplicate frames: how much does a new place cost?

Status: completed

**Question:** nuScenes-trained systems are known to look better than they are because validation and test frames lie close to training frames (Lilja et al.: > 80 % of the nuScenes frames within 5 m of a training frame), and a colleague reports that models trained on nuScenes break down elsewhere. How much of this dense head's score is memorisation of places, and what does it cost to run it in a city it was not trained in (Boston <-> Singapore, left- against right-hand traffic)? Does a few target-domain data close the gap?

## Setup
- Heads (all on the frozen GaussianCaR features, same recipe as the dense head of loop 3 - lr 6e-4, beta2 0.98, adaptive clipping, roll-back guard, 3,000 steps, batch 8, direction loss; the same number of training rows for every head so that only the city differs): B = 2,200 randomly chosen TRAIN rows from Boston only (4,569 available), S = 2,200 rows from Singapore only (One-North and Queenstown; 3,845 available), M = 2,200 rows from both cities in their natural proportions. The reference is the all-data head e37_final (8,414 rows, both cities).
- Evaluation: VAL and TEST split by the city of the scene (VAL Boston 396 / Singapore 477 frames, TEST Boston 355 / Singapore 478); pixel F1 at 1 m with tau chosen per head on VAL (mean of the two VAL cities). Fit diagnosis: F1 on the head's own training rows and on the TRAIN-eval rows of each city.
- Leakage check (Lilja et al.): for every VAL / TEST frame the horizontal distance of its global ego position to the nearest TRAIN frame of the SAME city (poses of the nuScenes ego_pose); frames are grouped into <= 5 m, 5-10, 10-25, 25-100 and > 100 m, and the F1 of the heads is computed per group.
- E58b: the Boston head B is fine-tuned on N = 100 / 400 / 1,600 randomly chosen Singapore TRAIN rows (400 / 800 / 1,500 steps, lr 2e-4, one cycle), to see how little target-domain data is needed; the Boston score after the fine-tuning shows forgetting.

![architecture](architecture.png)

## Numbers

**Pixel F1 @1 m of the equal-size heads (2,200 training rows each; tau chosen on VAL per head)**

| head (training rows; tau chosen on VAL) | F1 on its own training rows | F1 on TRAIN-eval rows: Boston / Singapore | VAL Boston | VAL Singapore | TEST Boston | TEST Singapore |
|---|---|---|---|---|---|---|
| B: Boston only (2,200 rows) | 0.978 | 0.974 / 0.841 | 0.904 | 0.815 | 0.878 | 0.777 |
| S: Singapore only (2,200 rows) | 0.929 | 0.843 / 0.925 | 0.812 | 0.861 | 0.793 | 0.837 |
| M: both cities (2,200 rows) | 0.931 | 0.941 / 0.908 | 0.877 | 0.850 | 0.848 | 0.817 |
| (reference) all 8,414 training rows, both cities (e37_final) | - | - | 0.903 | 0.866 | 0.879 | 0.835 |

**City-transfer gap**

| head | set | F1 in the training city | F1 in the other city | gap (other - own) |
|---|---|---|---|---|
| B: Boston only (2,200 rows) | VAL | 0.904 | 0.815 | -0.089 |
| B: Boston only (2,200 rows) | TEST | 0.878 | 0.777 | -0.101 |
| S: Singapore only (2,200 rows) | VAL | 0.861 | 0.812 | -0.049 |
| S: Singapore only (2,200 rows) | TEST | 0.837 | 0.793 | -0.044 |
| M: both cities (2,200 rows) | VAL | 0.877 (B) / 0.850 (S) | - | - |
| M: both cities (2,200 rows) | TEST | 0.848 (B) / 0.817 (S) | - | - |

**E58b - a few target-domain rows: fine-tuning the Boston head on Singapore rows (pixel F1 @1 m)**

| head | VAL Singapore | TEST Singapore | VAL Boston | TEST Boston |
|---|---|---|---|---|
| head B as trained (no Singapore data) | 0.815 | 0.777 | 0.904 | 0.878 |
| + fine-tune on 100 Singapore rows (400 steps) | 0.839 | 0.795 | 0.846 | 0.822 |
| + fine-tune on 400 Singapore rows (800 steps) | 0.849 | 0.815 | 0.833 | 0.819 |
| + fine-tune on 1,600 Singapore rows (1,500 steps) | 0.861 | 0.831 | 0.839 | 0.820 |
| (reference) head S: 2,200 Singapore rows from scratch | 0.861 | 0.837 | 0.812 | 0.793 |
| (reference) all 8,414 rows, both cities | 0.866 | 0.835 | 0.903 | 0.879 |

**Distance of the evaluation frames to the nearest training frame (Lilja et al. leakage check) and F1 per distance class**

| head / evaluation set | frames within 5 m / 5-10 / 10-25 / 25-100 / > 100 m of a training frame | F1 @1 m, 0-5 m | F1 @1 m, 5-10 m | F1 @1 m, 10-25 m | F1 @1 m, 25-100 m | F1 @1 m, 100-inf m | F1 all |
|---|---|---|---|---|---|---|---|
| all-data head/val | 582 / 92 / 73 / 93 / 33 | 0.889 | 0.872 | 0.860 | 0.869 | 0.930 | 0.885 |
| all-data head/test | 566 / 83 / 67 / 117 / 0 | 0.877 | 0.750 | 0.843 | 0.813 | - | 0.856 |
| head B/test_b | 315 / 1 / 8 / 31 / 0 | 0.900 | 0.650 | 0.688 | 0.583 | - | 0.878 |
| head B/test_s | 251 / 82 / 59 / 86 / 0 | 0.761 | 0.715 | 0.837 | 0.862 | - | 0.777 |
| head S/test_b | 315 / 1 / 8 / 31 / 0 | 0.791 | 0.533 | 0.484 | 0.527 | - | 0.769 |
| head S/test_s | 251 / 82 / 59 / 86 / 0 | 0.843 | 0.747 | 0.856 | 0.864 | - | 0.831 |
| head M/test_b | 315 / 1 / 8 / 31 / 0 | 0.852 | 0.597 | 0.574 | 0.576 | - | 0.831 |
| head M/test_s | 251 / 82 / 59 / 86 / 0 | 0.799 | 0.732 | 0.858 | 0.851 | - | 0.802 |

## Result images

![Pixel F1 of the three equal-size heads on the two cities (VAL, TEST): the matched-city score is always the highest; the mixed head is close to the matched ones in both cities.](e58_bars.png)
*Pixel F1 of the three equal-size heads on the two cities (VAL, TEST): the matched-city score is always the highest; the mixed head is close to the matched ones in both cities.*

![F1 against the distance of an evaluation frame to its nearest training frame: left - all-data head; right - city heads.](e58_bins.png)
*F1 against the distance of an evaluation frame to its nearest training frame: left - all-data head; right - city heads.*

![The same TEST frames under the three equal-size heads: the head of the OTHER city misses lanes and invents few; the mixed head is close to the matched head (heat map in colour, ground-truth centerlines in cyan).](e58_examples.png)
*The same TEST frames under the three equal-size heads: the head of the OTHER city misses lanes and invents few; the mixed head is close to the matched head (heat map in colour, ground-truth centerlines in cyan).*

![Fine-tuning the Boston head on a few Singapore rows: F1 on Singapore against the number of target-domain rows.](e58b_ft.png)
*Fine-tuning the Boston head on a few Singapore rows: F1 on Singapore against the number of target-domain rows.*

## Findings
- Leakage in the splits is real but small for this model: 66.7 % of the VAL and 67.9 % of the TEST frames lie within 5 m of a training frame of the same city (median distance 1.8 m / 1.3 m; 86 % within 25 m) - 85-89 % of the Boston frames, 51-53 % of the Singapore frames. Yet the F1 of the all-data head depends only weakly on that distance: VAL 0.889 within 5 m against 0.869 at 25-100 m, TEST 0.877 against 0.813 (n = 117). Only in Boston TEST the few frames more than 10 m from any training frame (about 40 of 355) are clearly worse for every head (0.48-0.69 against 0.79-0.90 for the near frames, depending on the head), while in Singapore the far frames are better (0.86 against 0.76); with 21 scenes this is confounded with what kind of road is far from the training drives (junction-rich Boston areas against plain Singapore stretches). The dense head on frozen features does not memorise places the way a map-memorising model would.
- A new city costs 0.04-0.10 F1: the Boston head scores 0.904 / 0.878 (VAL / TEST) in Boston but 0.815 / 0.777 in Singapore (gap 0.089 / 0.101); the Singapore head 0.861 / 0.837 in Singapore and 0.812 / 0.793 in Boston (gap 0.049 / 0.044). The Boston-trained head transfers worse - Boston scenes are easier (0.904 against 0.861 for the matched heads) and have no left-hand traffic.
- The mixed head gets both: with the same 2,200 rows it scores 0.877 / 0.848 in Boston and 0.850 / 0.817 in Singapore, i.e. within 0.03 (Boston) and 0.02 (Singapore) of the matched specialists and 0.03-0.06 above the other city's specialist. Data from the target city, not the amount of data, decides: the matched specialists with 2,200 rows already equal the all-data head (8,414 rows) in their own city (Boston 0.904 / 0.878 against 0.903 / 0.879; Singapore 0.861 / 0.837 against 0.866 / 0.835) - the scaling with the training rows has saturated at the seen-to-unseen shift of the foundation (loop 3).
- Train-vs-val diagnosis: with 2,200 rows the heads fit their own training rows to 0.978 (B), 0.929 (S) and 0.931 (M) and score 0.904 / 0.861 / 0.863 (mean of both cities) on the unseen VAL scenes of their own domain and 0.878 / 0.837 / 0.832 on TEST - a train-to-unseen gap of 0.07 (VAL) and 0.09-0.10 (TEST) that is of the same size as the city gap; and on the TRAIN rows of the OTHER city the heads score like on VAL there (B on Singapore train rows 0.841, S on Boston train rows 0.843), so the penalty of a new city is a property of the city, not only of unseen scenes.
- A little target-domain data closes most of the gap (E58b): fine-tuning the Boston head on 100 / 400 / 1,600 Singapore rows lifts the Singapore score from 0.777 to 0.795 / 0.815 / 0.831 on TEST (0.815 -> 0.839 / 0.849 / 0.861 on VAL) - with 1,600 rows the Boston-trained head comes within 0.006 of the Singapore specialist trained from scratch on 2,200 rows (0.837 TEST, 0.861 VAL), with 400 rows it recovers 63 % of the TEST gap to the Singapore specialist (0.060; 38 % of the 0.101 transfer gap). The price is forgetting: the Boston score falls from 0.878 / 0.904 to about 0.82 (TEST) and 0.83-0.85 (VAL) because no Boston rows were replayed - the mixed head M shows that training on both avoids this (Boston 0.848). This matches the finding of the campus-deployment study of the literature (a nuScenes-trained model recovers from 13.0 mAP on the campus to 56.9 mAP with campus data): a few hundred target-domain frames are the realistic remedy, and replay of the source domain is part of it.
- Reading for the real world: a model that is tuned and evaluated on one city can lose 0.04-0.10 F1 in another before any sensor or weather change; the geometric safeguards of this loop (map association with registration, monitors) are not expected to depend on the city (not measured per city here), the lane extraction does. The distance-to-train analysis suggests that the nuScenes split leak inflates this model's TEST F1 by at most a few hundredths, while the city shift is the larger effect.

## Limits
- Two cities only (Singapore's One-North / Queenstown treated as one domain); TEST has 355 / 478 frames from 21 scenes, so the per-distance-class numbers (n = 1-40 in the far classes) are anecdotal; one training run per head (seed noise of the dense head: 0.003 F1).
- The distance to the nearest training FRAME is a proxy for 'place seen before'; it does not capture seen road layouts at different places.
- Heads trained on 2,200 rows are weaker than the all-data head in the other city; the comparison across heads isolates the city, not the best attainable score.
