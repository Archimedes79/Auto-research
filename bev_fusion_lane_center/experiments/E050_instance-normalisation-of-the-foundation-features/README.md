# E50: Instance normalisation of the foundation features: is the seen-to-unseen shift a change of feature statistics?

Status: completed

**Question:** E30 showed that the train-to-validation gap of the dense head is dominated by a seen-to-unseen-scene shift of the frozen foundation features (0.05 F1 on VAL, 0.08 on TEST for the same heads) and that head-side regularisation and mirror augmentation do not remove it. If the shift were a global change of the feature statistics (for example more confident logits on scenes the foundation was trained on), normalising every sample's channels (mean and standard deviation over the grid) at the head's input should remove most of it. Does it?

## Setup
- Control and test are identical to the adjustment runs of E30: every second training row (4,207 samples), 3,000 steps, the stabilised recipe, the same seed (21); only the input changes: after the global channel normalisation each sample's 39 channels are standardised by their own mean and standard deviation over the 200 x 200 grid (--inorm 1; statistic floor 0.01).
- Reference: e30_base2 (same rows, steps, seed; global normalisation only): F1@1 m TRAIN_EVAL 0.933 / VAL 0.872 / TEST 0.838.

![architecture](architecture.png)

## Numbers

**Dense head comparison (threshold chosen on VAL, applied to TRAIN_EVAL / TEST)**

| head (4,207 training rows, 3,000 steps, seed 21) | TRAIN_EVAL F1@1m | VAL F1@0.5m | VAL F1@1m | VAL F1@1.5m | VAL P / R @1m | TEST F1@1m | TEST P / R @1m | gap TRAIN_EVAL - VAL | tau (VAL) |
|---|---|---|---|---|---|---|---|---|---|
| reference: dec32 + logits (39 ch), global normalisation | 0.933 | 0.675 | 0.872 | 0.941 | 0.902 / 0.844 | 0.838 | 0.885 / 0.796 | 0.061 | 0.5 |
| instance-normalised inputs (39 ch) | 0.932 | 0.673 | 0.869 | 0.937 | 0.901 / 0.838 | 0.840 | 0.886 / 0.799 | 0.063 | 0.5 |

## Result images

![Precision-recall over the threshold sweep (x = VAL-chosen threshold).](e50_pr.png)
*Precision-recall over the threshold sweep (x = VAL-chosen threshold).*

## Findings
- No effect: with instance normalisation TRAIN_EVAL / VAL / TEST are 0.932 / 0.869 / 0.840 against 0.933 / 0.872 / 0.838 for the reference (all differences at most 0.003); the train-validation gap stays 0.063 (reference 0.061). Against the three-seed mean of the reference recipe (E55: VAL 0.869, TEST 0.838) the differences are -0.0004 (VAL, inside the seed spread) and +0.0019 (TEST, slightly above it, not seen on VAL): no gain.
- The seen-to-unseen shift is therefore NOT a global offset or scale of the feature statistics (calibration of confidence); it is a difference in CONTENT - which lane structures and layouts the foundation encodes at all on scenes it was not trained on - and cannot be removed by normalising the input. This agrees with E30 (heads trained on unseen features do not do better) and with E40 (external information, the map prior, shrinks the gap: 0.075 -> 0.022 at an exact pose).
- Consequence for the foundation question: a better foundation has to be one that has not memorised its training scenes' map labels (or one trained on more scenes); post-hoc normalisation, regularisation and augmentation of the head do not help.

## Limits
- Single seed and the reduced setting (4,207 rows, 3,000 steps) of the E30 adjustment runs; per-sample normalisation of the 7 logit channels also removes their absolute calibration, which could hurt as much as it helps.
