# E252 Map offset of the lane ground truth: metric by city and lateral bias of correct predictions

The card (read by `experiment.py` and `decide.py`; schema `autoresearch/schemas/experiment.schema.json`):

```json card
{
 "id": "E252",
 "card": "H20.G1",
 "title": "the map offset of the lane ground truth (0.3 - 1 m per element and drive, worst in Boston) lowers the metric by city and shows up as a per-scene lateral bias of correct predictions",
 "kind": "variant",
 "deterministic": true,
 "metric": "Q",
 "direction": "higher",
 "band": 0.01,
 "prediction": {
  "value": -0.03,
  "vs": "c"
 },
 "variant": "a",
 "control": "c",
 "arms": {
  "a": {
   "recipe": "analysis script temp/loop20/e252_gt_offset.py on the cached raw outputs of champion, e24_j1eq_s0, e24_j1rc_s0 (VAL 873): the frames of boston-seaport",
   "base": "e123_D7_r3_f1",
   "names": [
    "champion"
   ],
   "note": "per-city pooling inside the script; the arms of this card name the two groups of frames, not two checkpoints"
  },
  "c": {
   "recipe": "the same models, the frames of the three Singapore locations",
   "base": "e123_D7_r3_f1",
   "names": [
    "champion"
   ]
  }
 },
 "rule": [
  {
   "metric": "Q",
   "min_effect": 0.01,
   "note": "city gap Boston - Singapore, read from the printout (the tool cannot split by city): CAP when Q(Boston) is lower by >= 0.03 for all three models with non-overlapping 95 % scene intervals AND the share of correct points with |lateral offset| > 0.5 m is >= 1.5 x that of Singapore; NO EVIDENCE when the Q gap is < 0.01"
  }
 ],
 "budget_gpu_h": 0.0,
 "kill": "none (CPU, 1 thread, about 10 min); void if the frame count per city does not add up to 873 or the identity check (offset of the GT against itself) is not exactly 0",
 "evaluate": {
  "split": "val"
 }
}
```


## 0. What the tools run (the fields of `experiment.json`)
| field | value |
|---|---|
| id / card / title | E252 / H20.G1 / map offset of the ground truth, metric by city |
| kind | variant (analysis of existing checkpoints, no training, no shared code) |
| deterministic | true (cached raw outputs; the scene bootstrap gives the interval) |
| metric, direction, band | Q, higher, 0.010 (also F1@0.5 / 1.0 / 2.0, lanes found, R@1_far, vert100 per city) |
| prediction | Boston minus Singapore in Q: -0.03 (champion, j1eq, j1rc alike) |
| arms | a: Boston frames, c: Singapore frames of VAL 873 (three models each) |
| control | c |
| rule | see section 1 |
| budget_gpu_h, kill | 0, see section 1 |
| evaluate | split val (873 frames), cached raw outputs |

## 1. Card
| | |
|---|---|
| lens | system review (GT quality), past critic |
| group | A9 evaluation, data quality |
| sources | bev_fusion/plan.md (map-image alignment, per-location check, 2026-10-26), D10 (distance map), E238 (assumed 2 m scene-constant map error), E092 (duplicate junction annotations) |
| status | proposed |

- **Statement**: the nuScenes map is not registered to the images with the 1 m tolerance of our metric; the lane centrelines inherit an offset of 0.3 - 1 m that varies by element and by drive and is worst in Boston, so (a) Q and F1@1.0 are lower in Boston than in Singapore for every model, and (b) correct predictions carry a per-scene lateral bias (signed offset to the nearest ground-truth point) in Boston that Singapore scenes do not have.
- **Mechanism**: the other chat measured the paint against the map lines (share of line elements within 0.25 m of the paint: Boston lane dividers 41 %, Singapore-queenstown 82 %); lane centrelines come from the same map. A prediction that follows the real lane then sits at the offset from the ground truth. A model trained on this ground truth can learn a systematic part of the offset, not the per-drive part, so the residual is a lower bound.
- **Prediction**: Q(Boston) - Q(Singapore pooled) = -0.03 for each of champion, e24_j1eq_s0, e24_j1rc_s0; share of correct points with |lateral offset| > 0.5 m at least 1.5 x higher in Boston; per-scene median signed offset spread (IQR) twice as wide in Boston. Reference class: city effects in perception metrics are typically 0.02 - 0.05; the weakest part is (b), because scene difficulty differs by city too.
- **Decision rule**: CAP (the ground truth caps the metric in Boston): all three Q gaps <= -0.03 with 95 % scene-bootstrap intervals that do not overlap AND the |offset| > 0.5 m share ratio >= 1.5 -> report every metric by city from now on and write a DECISIONS entry for a tolerance-aware variant of the protocol. NO EVIDENCE: Q gap within +-0.010 -> the offset does not show in our metric; record and close. Anything between: report by city only. A Q gap alone is not proof (scene difficulty), the residual reading (b) must agree.
- **Test design**: three checkpoints (champion f1 + time recursion, j1eq seed 0, j1rc seed 0), all registry names, standard settings; layer 0 check inside the script (an identity: offset of the ground truth against itself is 0); one run, 1 thread.
- **Cost / kill**: CPU about 10 minutes; void if frame counts per city do not add to 873.
- **Inputs**: ready (cached raw outputs, `poses_val.npz` loc and scene, scene bootstrap helpers).
- **Referee** (2026-10-26, ar-referee): REVISE (cheap, CPU, not tried before, worth running once the points below are in; the card as written cannot separate "GT offset" from "Boston is harder"). Revise: (1) CONFOUND: add a permutation null: shuffle the city label over scenes (>= 1000 draws, same scene counts) and report where the observed Boston - Singapore Q gap sits in that distribution; and the offset signature: an offset acts at tight tolerances, difficulty also at loose ones, so make gap(F1@0.5) - gap(F1@2.0) and the Q gap against the F1@2.0 gap part of the rule (offset: gap shrinks by >= half from F1@0.5 to F1@2.0; difficulty: stays), plus a recall-vs-precision split (lanes found vs vert100 / FL). (2) CALIBRATION: add a synthetic-shift control on the Singapore frames (shift the GT laterally by 0.3 / 0.5 / 1.0 m, re-score): it says what Q loss a given offset costs; without it -0.03 cannot be read as an offset in metres. (3) PREDICTION as {value, tol, class}: -0.03 with tolerance +-0.02 is not a decision threshold: the rule "<= -0.03" is passed by 50 % of the outcomes if the prediction is right; set the CAP threshold at the lower end (<= -0.02) AND the 95 % interval of the DIFFERENCE (scene bootstrap of both cities, not "intervals do not overlap") excludes -0.010, and add the IQR clause (2x) of the prediction to the rule or drop it from the prediction. The three models are not independent (two fine-tunes of one base, same train data): name the champion the primary test, the other two as replication, and say so. (4) STATISTIC (b): the plan's own finding is that map error is per ELEMENT (within-frame std 0.31 m > frame-mean movement 0.18 m), so per-scene medians average it away; compute the median signed offset per ground-truth lane polyline (and its spread within a scene) as well; the <= 1.5 m gating truncates exactly the large offsets (report the share of predicted points with no GT within 1.5 m per city); |s| > 0.5 share also grows with model noise at intersections / wide lanes, so stratify by straight single-lane segments or report beside GT curvature. (5) Power and locations: the script prints scenes per city; void (not "no evidence") if Boston or Singapore has < 8 VAL scenes, and report the four locations separately (boston-seaport vs queenstown / hollandvillage as the best-aligned pair; onenorth 53 % is a mid case) since 'Singapore pooled' mixes 53 % and 82 % alignment. (6) MECHANISM gap: the 41 % / 82 % numbers are for divider / boundary lines from 15 scenes x 4 keyframes with 'clear peaks only'; centrelines are a different layer, so 'inherit' is an assumption: say so in the Statement; the CAP outcome cannot by itself justify a DECISIONS entry for a protocol change (that needs the multi-log estimate). (7) Kill: a sanity check is not a kill criterion; keep 'void if counts <> 873' and add 'void if the identity check fails' (the script already asserts it).

## 3. Summary (2026-10-26, run E252.a0; VAL 873 frames, cached raw outputs, standard settings; output `gt_offset.txt`, script `temp/loop20/e252_gt_offset.py`)
- **Decision (by the rule as written): the cap is NOT supported; prediction missed in sign.** The Q gap Boston - Singapore is +0.052 (champion), +0.049 (j1eq seed 0), +0.070 (j1rc seed 0): Boston scores higher, not lower; the CAP clause (all gaps <= -0.03, ratio >= 1.5) failed on both parts. The gap lies outside the NO EVIDENCE band (+-0.010), so the rule says "report by city only"; it is reported below.
- **VAL 873 has only two cities**: boston-seaport 396 frames / 10 scenes and singapore-onenorth 477 frames / 12 scenes. Queenstown and hollandvillage (where the other chat found the offsets small and the drivable-area bug) are not in VAL, so the metric says nothing about them.
- **Numbers (Q, 95 % scene-bootstrap interval, B = 500; F1@1.0)**: champion Boston 0.7625 [0.7155, 0.8032], F1@1.0 0.8899; Singapore 0.7101 [0.6898, 0.7360], 0.8502. j1eq: 0.7609 [0.7236, 0.7999], 0.8801 against 0.7118 [0.6909, 0.7335], 0.8483. j1rc: 0.7757 [0.7383, 0.8176], 0.8884 against 0.7061 [0.6862, 0.7294], 0.8447. The intervals overlap for ten Boston scenes; the sign is the same for all three models and for F1@0.5 (champion 0.718 against 0.666).
- **Lateral residual of correct predictions (points within 1.5 m of the ground truth; champion; 0.41 / 0.43 million points)**: median |offset| 0.19 m Boston, 0.26 m Singapore; share with |offset| > 0.5 m 0.224 against 0.267 (ratio 0.84; j1eq 0.84, j1rc 0.83); share > 1.0 m 0.052 against 0.074. Per-scene median signed offset: IQR [-0.029, +0.004] m Boston, [-0.023, +0.043] m Singapore; 1 of 10 and 0 of 12 scenes have a median beyond 0.15 m (j1eq 0 / 0, j1rc 1 / 0). Layer 0: the offset of the ground truth against itself is exactly 0 on 20 frames; frame counts add to 873.
- **What it says**: the divider offsets of 0.3 - 1 m that the BEV-system chat measured against the paint do not appear as a scene-constant lateral bias of correct predictions on the lane centrelines of these two cities: the per-scene medians are within a few centimetres. Either the centrelines are better registered than the dividers, or a per-drive error is random along the scene and averages out in the median, or the model has learned the systematic part. The 1 m tolerance of the metric absorbs offsets of this size; Boston scores higher on our metric for a reason this experiment cannot identify (scene difficulty, fewer lanes per frame, other), so it is not capped by the offset. This lowers the weight of the GT-offset argument for the lane project and fits the other chat's own finding that the elements differ within one frame (they are per element, not per scene).
- **Weaknesses**: only two cities, ten Boston scenes; the residual uses points within 1.5 m (a larger offset is excluded and counted as false length); a learned systematic offset hides in the model (lower bound); one checkpoint per kind; scene difficulty differs by city, so the Q gap is not a GT statement either way. The pooled R@1_far and F1@2.0 are not available in this reading (nan in the printout, not used).
- **Next**: the per-element reading needs the multi-log estimate of the other chat; for the lane project report metrics by city next to the pooled number; use the b456 UNSEEN scenes (hollandvillage) for a third city once the loader exists (S19.9).

### Follow-up after the referee (run E252.b0, champion only, `gt_offset_controls.txt`, script `temp/loop20/e252b_controls.py`)
The referee (REVISE) asked for a null, an offset signature, a synthetic-shift scale and a per-polyline statistic. All four were run after the first verdict; the verdict did not change.
- **Permutation null** (22 scenes, labels permuted 10 / 12, 2000 draws): the observed Q gap +0.0524 has a null SD of 0.0274; share of draws with |gap| >= |observed| 0.056 (two-sided), with gap >= observed 0.032. Boston is higher at the edge of chance with ten scenes; the cap hypothesis needed a gap of -0.03 or lower and is on the wrong side of the null.
- **Offset signature**: gap F1@0.5 +0.0527, F1@1.0 +0.0398 (ratio 0.75): the gap shrinks with the tolerance but has the wrong sign for an offset effect (Boston is higher, not lower).
- **Synthetic shift of the ground truth** (all frames, champion): Q 0.7350 -> 0.7125 (0.3 m) -> 0.6652 (0.5 m) -> 0.5203 (1.0 m); F1@1.0 0.8696 -> 0.8488 -> 0.8229 -> 0.6025. So a lateral GT offset of 0.3 - 0.5 m would cost 0.02 - 0.07 of Q: the metric is very sensitive to it, and the observed city gap (+0.05) is the size of a 0.4 m shift in the opposite direction of the hypothesis.
- **Per ground-truth polyline** (predicted points within 1.5 m, at least 20 points): median |polyline median offset| 0.157 m Boston against 0.221 m Singapore; share above 0.3 m 0.283 against 0.380; above 0.5 m 0.149 against 0.187; predicted points with no ground-truth point within 1.5 m 0.026 against 0.056. Boston is better on every count; the per-element offsets that the other chat found on dividers do not make the Boston centrelines worse for correct predictions.
- **Remaining limits** (referee items 6, 7): only two of four locations are in VAL; the 41 % / 82 % alignment figures belong to dividers and boundaries, and that centrelines inherit that error remains an assumption that this experiment weakens but cannot test (a model trained on the ground truth can hide a systematic offset). A third city needs the b456 loader.
