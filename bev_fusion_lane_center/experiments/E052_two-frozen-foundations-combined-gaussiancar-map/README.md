# E52: Two frozen foundations combined: GaussianCaR (map-pretrained) + CRN (detection-pretrained) features

Status: completed

**Question:** E27 showed that the detection-pretrained CRN BEV feature carries lane structure far worse than the map-pretrained GaussianCaR feature on its own (F1@1 m 0.624 against 0.867 on VAL). Does it still add COMPLEMENTARY information - for example vehicle positions as lane cues, or a second camera-radar encoding - when both frozen foundations are concatenated at the input of the dense head? (A cheap 'better foundation by fusion' test with the features that already exist on disk; no new weights.)

## Setup
- Input = the 39 GaussianCaR channels (decoder PCA 32 + 7 map-layer logits, 200 x 200) plus the 32 CRN channels (PCA of the detection network's BEV feature, 128 x 128 cells of 0.8 m, bilinearly up-sampled to 200 x 200 as in E27) = 71 channels in the same ego frame.
- Identical to the adjustment runs of E30 / E49 / E50: every second training row (4,207 samples), 3,000 steps, the stabilised recipe (lr 6e-4, beta2 0.98, clipping, roll-back guard), seed 21; reference = e30_base2 (39 channels).

![architecture](architecture.png)

## Numbers

**Dense head comparison (threshold chosen on VAL, applied to TRAIN_EVAL / TEST)**

| head (4,207 training rows, 3,000 steps, seed 21) | TRAIN_EVAL F1@1m | VAL F1@0.5m | VAL F1@1m | VAL F1@1.5m | VAL P / R @1m | TEST F1@1m | TEST P / R @1m | gap TRAIN_EVAL - VAL | tau (VAL) |
|---|---|---|---|---|---|---|---|---|---|
| reference: dec32 + logits (39 ch), global normalisation | 0.933 | 0.675 | 0.872 | 0.941 | 0.902 / 0.844 | 0.838 | 0.885 / 0.796 | 0.061 | 0.5 |
| GaussianCaR (39 ch) + CRN (32 ch) frozen features (71 ch) | 0.927 | 0.668 | 0.865 | 0.938 | 0.895 / 0.838 | 0.836 | 0.883 / 0.795 | 0.062 | 0.5 |

## Result images

![Precision-recall over the threshold sweep (x = VAL-chosen threshold).](e52_pr.png)
*Precision-recall over the threshold sweep (x = VAL-chosen threshold).*

## Findings
- No gain from the fusion: F1@1 m TRAIN_EVAL / VAL / TEST 0.927 / 0.865 / 0.836 against 0.933 / 0.872 / 0.838 for the GaussianCaR-only reference (VAL precision / recall 0.895 / 0.838 against 0.902 / 0.844) - a small loss on VAL (-0.007) and none on TEST (-0.002); the train-validation gap is unchanged (0.062 against 0.061). Against the three-seed mean of the reference recipe (E55: VAL 0.869, TEST 0.838) the differences are -0.0037 (VAL) and -0.0015 (TEST): a small loss of the order of one to two seed standard deviations, no gain.
- The detection-pretrained BEV feature therefore adds nothing the map-pretrained one does not already carry for lane centerlines: its lane information is much weaker (E27) and whatever else it encodes (objects) is either not useful as a lane cue for a 2.7 M-parameter head at this data size or already part of GaussianCaR's own camera-radar fusion. Together with E49 (wider read-out of the same decoder) and E50 (instance normalisation) this closes the cheap routes to a 'better foundation' on the head side: the lane information that the frozen features can give to this head is saturated; what limits the result is the seen-to-unseen shift of the foundation itself (E30).

## Limits
- Single seed, 4,207 training rows; the CRN features are a 32-channel PCA of the detection BEV feature on a 128 x 128 grid whose extent (+-51.2 m) differs slightly from the 200 x 200 grid (+-50 m) - the bilinear up-sampling of E27 was kept. A deeper fusion (feature-level attention) or the full 256 / 128 channels of the CRN BEV feature was not tried.
