# E39: Cascade refiner on top of the grid ('pooling + re-estimation, blocks one behind another')

Status: completed

**Question:** Does a learned sparse stage that starts from the grid's polylines - sampling the feature maps across each line, re-estimating the points block by block, with a loss that penalises false alarms more than omissions - improve geometry and topology over the grid alone, and is it better than a sparse extractor without the grid (E32)?

## Setup
- Input: proposals of E38 (relaxed tracer, 1 m spacing; at most 48 pieces x 100 points per sample), the 39 foundation channels, the dense head's heat map and direction field. The refiner is trained ONLY on out-of-fold dense predictions of all 8,414 TRAIN samples (E37 fold heads) and applied to the final head's VAL / TEST predictions.
- Block (3 in the main model): sample a 2-D feature pyramid (fine 200x200 and coarse 50x50 branch of a small encoder) at 7 lateral offsets (+-2.5 m) and 5 coarse offsets (+-6 m) along the normal of every point ('pooling'), embed, mix along the line with dilated 1-D residual convolutions (context along the lane), and 're-estimate' per point a lateral offset (+-2 m), a longitudinal offset (+-0.6 m) and an existence logit; points move and the next block samples again at the new positions. Per piece an orientation head decides the travel direction. 0.76 M parameters (in the spirit of iterative vertex deformation with 1-D convolutions along the contour, e.g. Deep Snake, but without any matching).
- Loss without any matching: the exact ground-truth distance field is sampled (differentiably) at the moved points (Huber), existence is supervised by 'point within 1 m of GT' with the FALSE-ALARM class weighted by lambda_FP = 3 (errors cost more than omissions), plus point-spacing and smoothness regularisers and an orientation loss from the GT direction field. Proposals are jittered laterally (0.5 m) in training so that the refiner also learns to cope with worse proposals. 5,000 steps, batch 8, AdamW, one-cycle LR 1e-3.
- Post-processing: cut pieces where existence < tau_e (0.3, chosen on VAL; the sweep is flat between 0.3 and 0.5), orient, suppress duplicates (shorter pieces lying within 0.75 m of a longer one for >= 70 % of their length), link pieces: edges inherited from the grid graph plus geometric end-to-start links.
- Ablations (one change at a time, same data and steps): number of blocks (1 vs 3), false-alarm weight (lambda_FP 1 vs 3), input regularisation (channel dropout 0.2 + Gaussian noise 0.3 on the foundation features). Not run for lack of GPU time: 2 / 4 blocks, lambda_FP 6, no jitter, 2 m point spacing (the point-spacing question is covered by E33 for the tracer).
- Comparison systems: the grid graph of E31 (strict tracer + assembler, now with the E37 head), the unrefined relaxed proposals with the same linking, the sparse-only extractor of E32 (no grid), and the oracle ceiling of the rule-based assembler (E38).

![architecture](E39_architecture.png)

## Numbers

**System comparison (point and graph metrics; thresholds fixed on VAL)**

| system | set | F1@0.5m | F1@1m | F1@1.5m | P@1m | R@1m | lanes found | false length | split F1 3m | split F1 6m | merge F1 3m | merge F1 6m | route rec | route prec | route F | dir acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sparse only (E32 LaneFormer, no grid) | TRAIN (eval subset) | 0.735 | 0.880 | 0.943 | 0.893 | 0.868 | 0.756 | 0.046 | 0.137 | 0.225 | 0.143 | 0.244 | 0.625 | 0.566 | 0.594 | 0.943 |
| sparse only (E32 LaneFormer, no grid) | VAL | 0.636 | 0.821 | 0.909 | 0.839 | 0.804 | 0.649 | 0.072 | 0.083 | 0.169 | 0.101 | 0.198 | 0.611 | 0.565 | 0.587 | 0.941 |
| sparse only (E32 LaneFormer, no grid) | TEST | 0.601 | 0.791 | 0.884 | 0.803 | 0.779 | 0.603 | 0.099 | 0.086 | 0.176 | 0.112 | 0.192 | 0.564 | 0.480 | 0.519 | 0.923 |
| grid graph (E37 head + strict tracer + assembler) | TRAIN (eval subset) | 0.864 | 0.965 | 0.987 | 0.979 | 0.952 | 0.909 | 0.007 | 0.244 | 0.399 | 0.264 | 0.421 | 0.778 | 0.797 | 0.788 | 0.945 |
| grid graph (E37 head + strict tracer + assembler) | VAL | 0.721 | 0.891 | 0.950 | 0.909 | 0.874 | 0.772 | 0.039 | 0.185 | 0.286 | 0.230 | 0.390 | 0.776 | 0.789 | 0.782 | 0.944 |
| grid graph (E37 head + strict tracer + assembler) | TEST | 0.693 | 0.867 | 0.932 | 0.892 | 0.844 | 0.720 | 0.048 | 0.199 | 0.322 | 0.207 | 0.362 | 0.711 | 0.733 | 0.722 | 0.932 |
| grid proposals unrefined (relaxed tracer, same linking, tau_e 0.3) | VAL | 0.716 | 0.886 | 0.946 | 0.898 | 0.874 | 0.771 | 0.050 | 0.190 | 0.290 | 0.234 | 0.393 | 0.772 | 0.775 | 0.773 | 0.942 |
| grid proposals unrefined (relaxed tracer, same linking, tau_e 0.3) | TEST | 0.686 | 0.860 | 0.926 | 0.873 | 0.847 | 0.721 | 0.066 | 0.200 | 0.324 | 0.207 | 0.358 | 0.707 | 0.712 | 0.709 | 0.929 |
| grid proposals unrefined (relaxed tracer, same linking, tau_e 0.3) | TRAIN scenes, held out from the dense head (OOF fold 0; seen by the refiner) | 0.790 | 0.924 | 0.964 | 0.935 | 0.913 | 0.822 | 0.032 | 0.231 | 0.369 | 0.256 | 0.399 | 0.748 | 0.756 | 0.752 | 0.928 |
| grid + cascade refiner (E39 main, tau_e 0.3) | VAL | 0.730 | 0.892 | 0.949 | 0.919 | 0.867 | 0.759 | 0.032 | 0.198 | 0.297 | 0.235 | 0.401 | 0.769 | 0.824 | 0.796 | 0.963 |
| grid + cascade refiner (E39 main, tau_e 0.3) | TEST | 0.706 | 0.869 | 0.931 | 0.905 | 0.837 | 0.711 | 0.039 | 0.205 | 0.329 | 0.212 | 0.357 | 0.703 | 0.774 | 0.736 | 0.950 |
| grid + cascade refiner (E39 main, tau_e 0.3) | TRAIN scenes, held out from the dense head (OOF fold 0; seen by the refiner) | 0.828 | 0.941 | 0.972 | 0.967 | 0.917 | 0.822 | 0.011 | 0.236 | 0.376 | 0.262 | 0.398 | 0.757 | 0.816 | 0.785 | 0.959 |
| oracle field -> assembler (ceiling of the rule-based builder) | VAL | 0.951 | 0.993 | 0.997 | 0.998 | 0.987 | 0.984 | 0.000 | 0.236 | 0.340 | 0.242 | 0.450 | 0.838 | 0.837 | 0.838 | 0.960 |
| oracle field -> assembler (ceiling of the rule-based builder) | TEST | 0.943 | 0.993 | 0.997 | 0.998 | 0.988 | 0.986 | 0.001 | 0.249 | 0.368 | 0.271 | 0.449 | 0.823 | 0.823 | 0.823 | 0.957 |

**Ablations of the cascade refiner**

| variant | VAL F1@1m | VAL P@1m | VAL R@1m | VAL route F | VAL split F1 6m | VAL merge F1 6m | TEST F1@1m | TEST route F | OOF-A F1@1m (seen scenes) | train loss (last 100 steps) |
|---|---|---|---|---|---|---|---|---|---|---|
| main: 3 blocks, lambda_FP 3, jitter 0.5 m, 1 m spacing | 0.892 | 0.919 | 0.867 | 0.796 | 0.297 | 0.401 | 0.869 | 0.736 | 0.941 | 0.478 |
| 1 block | 0.892 | 0.918 | 0.868 | 0.793 | 0.292 | 0.406 | 0.868 | 0.737 | 0.941 | 0.224 |
| lambda_FP 1 (false alarms not up-weighted) | 0.891 | 0.911 | 0.873 | 0.788 | 0.296 | 0.403 | 0.867 | 0.730 | 0.940 | 0.358 |
| regularised: input channel dropout 0.2 + feature noise 0.3 | 0.894 | 0.919 | 0.871 | 0.796 | 0.295 | 0.401 | 0.871 | 0.740 | 0.937 | 0.515 |

**Existence threshold and de-duplication (VAL, with TEST for the threshold sweep)**

| tau_e | VAL P@1m | VAL R@1m | VAL F1@1m | VAL route F | VAL split F1 6m | VAL merge F1 6m | VAL false length | TEST F1@1m | TEST route F |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.913 | 0.870 | 0.891 | 0.791 | 0.300 | 0.401 | 0.036 | 0.868 | 0.732 |
| 0.3 | 0.919 | 0.867 | 0.892 | 0.796 | 0.297 | 0.401 | 0.032 | 0.869 | 0.736 |
| 0.4 | 0.927 | 0.862 | 0.893 | 0.797 | 0.295 | 0.400 | 0.027 | 0.869 | 0.739 |
| 0.5 | 0.934 | 0.853 | 0.892 | 0.798 | 0.292 | 0.398 | 0.022 | 0.866 | 0.739 |
| 0.6 | 0.941 | 0.840 | 0.888 | 0.794 | 0.288 | 0.396 | 0.019 | 0.860 | 0.738 |
| 0.7 | 0.947 | 0.823 | 0.880 | 0.790 | 0.282 | 0.390 | 0.016 | 0.850 | 0.729 |
| tau_e 0.3, dedupe 0.0 | 0.918 | 0.868 | 0.893 | 0.793 | 0.290 | 0.390 | 0.032 | - | - |
| tau_e 0.3, dedupe 0.5 | 0.920 | 0.863 | 0.891 | 0.795 | 0.285 | 0.398 | 0.032 | - | - |
| tau_e 0.3, dedupe 0.85 | 0.919 | 0.868 | 0.893 | 0.793 | 0.295 | 0.399 | 0.032 | - | - |
| tau_e 0.3, reduce edges (transitive reduction of the edge set) | 0.919 | 0.867 | 0.892 | 0.795 | 0.304 | 0.406 | 0.032 | - | - |

## Result images

![Ablations of the refiner (same proposals, same training data).](E39_e39_ablation.png)
*Ablations of the refiner (same proposals, same training data).*

![Operating point of the refiner: the existence threshold plays the role of the dense head's threshold.](E39_e39_tau.png)
*Operating point of the refiner: the existence threshold plays the role of the dense head's threshold.*

![Block-by-block movement of the proposal points on three VAL samples (the same input to all blocks is the proposal set; later blocks re-sample the maps at the new positions).](E39_e39_cascade.png)
*Block-by-block movement of the proposal points on three VAL samples (the same input to all blocks is the proposal set; later blocks re-sample the maps at the new positions).*

![VAL: grid graph vs grid + refiner vs ground truth.](E39_e39_system_val.png)
*VAL: grid graph vs grid + refiner vs ground truth.*

![TEST (untouched by all decisions): grid graph vs grid + refiner vs ground truth.](E39_e39_system_test.png)
*TEST (untouched by all decisions): grid graph vs grid + refiner vs ground truth.*

## Findings
- Headline: starting from the grid's polylines, the cascade refiner reaches VAL F1@1 m 0.892 (P 0.919, R 0.867), route F 0.796, split / merge F1 at 6 m 0.297 / 0.401 and direction accuracy 0.963; TEST 0.869, route F 0.736. That is +0.006 F1 and +0.023 route F over the UNREFINED proposals (0.886 / 0.773) and +0.001 F1, +0.014 route F, +0.019 direction accuracy over the grid graph of E38 (0.891 / 0.782 / 0.944). The sparse-only extractor of E32 (no grid) is far behind (VAL 0.821, route F 0.587): building the sparse stage ON the grid is clearly better than extracting vectors from scratch.
- The gain is small because the refiner mostly VERIFIES and SMOOTHS rather than relocates: of the 415,000 VAL proposal points, 89.8 % were already within 1 m of a ground-truth centerline and 90.1 % are afterwards (median point error 0.27 -> 0.26 m); of the points that start 1.0-1.5 m off only 28 % end within 1 m, of those 1.5-2.5 m off only 10 %. The cascade figure shows what changes: the scattered proposal points become smooth, evenly spaced lines after the FIRST block and the existence head separates reliable (green) from doubtful (red) parts. Existence is the useful signal: false lines farther than 5 m from the ground truth keep existence >= 0.3 for only 28 % of their points, and the precision of the points that are kept rises from 0.898 to 0.918 (tau_e 0.3) - 0.933 (tau_e 0.5), at the price of recall (0.874 -> 0.867 / 0.853).
- Why not more? The ceiling analysis (E38) leaves room (about +0.04 to +0.08 F1 if every covered GT point were snapped), but the 10 % of points that are 1-2.5 m off are exactly those where the frozen features carry no sub-metre cue that the dense head has not already used: junction interiors, overlapping connector lanes, far range. A second 'look' at the same features through a different head cannot recover information that is not in them.
- Train versus validation (overfitting diagnosis): on the out-of-fold TRAIN scenes that are IN the refiner's training data the same model gains +0.017 F1@1 m (0.924 -> 0.941) and +0.033 route F, pulling 51 % of the 1.0-1.5 m points and 36 % of the 1.5-2.5 m points within 1 m; on unseen VAL scenes it gains +0.006 and +0.023. So the refiner fits the training scenes better than it generalises: its training loss plateaus at about 0.47 after 4,000 steps while only 212 distinct scenes (strongly correlated frames) are available, plus the foundation-seen shift of E37. Regularising the input (channel dropout 0.2 + feature noise 0.3) narrows the gap a little (OOF-A 0.937, VAL 0.894, TEST 0.871 / route F 0.740, against 0.941 / 0.892 / 0.869 / 0.736) - an improvement of about +0.002, i.e. within noise. The refiner is therefore neither under-fitted (it has room on the training scenes) nor strongly over-fitted (the validation gain is the same size with and without regularisation); it is information-limited.
- Cascade depth does not matter: one block gives the same result as three (VAL 0.892 / route F 0.793 vs 0.892 / 0.796, TEST 0.868 / 0.737 vs 0.869 / 0.736, OOF-A 0.941 both). The 'pooling + re-estimation, blocks one behind another' scheme works, but a single block already extracts what the features offer.
- False alarms vs omissions (lambda_FP): weighting the false-alarm class of the existence loss three times as high (lambda_FP 3 vs 1, otherwise identical training) moves the refiner towards precision: VAL P 0.911 -> 0.919 with R 0.873 -> 0.867, TEST P 0.892 -> 0.905 with R 0.843 -> 0.837; F1@1 m changes by only +0.001 / +0.002 but route F rises by +0.008 / +0.006 (VAL / TEST) and the direction of the effect is the intended one - fewer false lines, slightly more omissions. The effect is small: with the same information in the features the weight mostly shifts the operating point along one precision-recall curve (as for the dense head in E29), and the existence threshold tau_e does the same job after training.
- Post-processing sweeps (VAL): the existence threshold trades precision for recall monotonically (tau_e 0.2 -> 0.7: P 0.913 -> 0.947, R 0.870 -> 0.823, false length 3.6 % -> 1.6 %), F1 is flat between 0.3 and 0.5 (0.892-0.893) and route F peaks at 0.5 (0.798); duplicate suppression (0 / 0.5 / 0.7 / 0.85 overlap) changes F1 by less than 0.002; a transitive reduction of the edge set (dropping shortcut edges i->j when i->k->j exists) raises split / merge F1 at 6 m slightly (0.297 / 0.401 -> 0.304 / 0.406) at equal route F.
- Remaining errors (system figure): spurious short branches (e.g. a false side lane in the second example), lanes that are partly invisible at far range, and junction interiors where lanes overlap - the topology (split / merge F1 at 6 m only 0.30 / 0.40) is bounded by the rule-based assembly and the ground-truth conventions (E38 oracle: 0.34 / 0.45), which is why the learned link head (E42) is tried next.

## Limits
- Single seed per configuration; differences below about 0.01 F1 are not interpreted (all ablation differences here are below that).
- The refiner can only reach what the proposals cover (E38): lines the grid misses entirely are not created by it.
- Training uses out-of-fold predictions of TRAIN scenes, which the foundation has seen; the validation scenes' maps are less accurate, so the refiner meets worse inputs at test time than it was trained on.
- Only 5,000 steps and a 0.76 M-parameter model were tried; the plateau of the training loss and the 1-block result suggest that more capacity or steps would not help without new information.
