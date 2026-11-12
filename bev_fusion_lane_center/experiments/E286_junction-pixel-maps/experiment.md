# E286 Junction pixel maps: entry / exit heat with heading, junction zone and conflict core, painted stop lines and crossings (D82, step 2 of the thread)

Card of 2026-11-11, written by the lane chat on Mirko's word (D82: "Dann sehr gute PixelMaps ... Und bleibe differenzierbar."). Needs the GT layer of E285 (`jgt.py`, `qjunction.py`, the paint sidecars). Everything on the training path is a soft map (BCE + dice, a cosine loss for the heading); the keypoint extraction (3 x 3 peaks, soft-argmax) is the one hard step and is a READ-OUT for scoring and the delivered output, named here and in the report.

## 0. The card (what the tools run)
```json card
{
 "id": "E286",
 "card": "",
 "title": "junction maps head on the frozen pixel stage of B (1 m: zone, conflict core, entry / exit heat and heading; 0.5 m: stop lines, pedestrian crossings), three context depths, against B's own stop / exit event heat maps read the same way",
 "kind": "block",
 "deterministic": false,
 "metric": "kp_f1",
 "direction": "higher",
 "band": 0.05,
 "prediction": {"value": 0.15, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = the stop / exit event heat maps of B's pyramid (runs e273_p_s0 / e273_p_s1, channels 8 / 9 of centre1), 3 x 3 peaks above 0.3, distance only (B has no heading); no new training; `jmaps_eval.py score` computes it next to every arm"},
  "j0": {"recipe": "`jmaps_train.py --ctx none` (frozen pyramid of B, 1 x 1 / 3 x 3 convolutions only, 3000 steps, batch 8, lr 1e-3)"},
  "j1": {"recipe": "`jmaps_train.py --ctx dil` (+ four dilated residual blocks at 2 m: a receptive field of about 60 m)"},
  "j2": {"recipe": "`jmaps_train.py --ctx attn` (+ one self-attention layer over 25 x 25 tokens at 4 m, learned position)"},
  "jj": {"recipe": "`jmaps_train.py --joint` with the context depth of the best of j0 / j1 / j2 (the pyramid of B trained together with the head, B's own pixel loss kept so that the maps of B do not drift; judged on the maps only, B's nodes / links no longer match such a pixel stage)"}
 },
 "control": "c",
 "variant": "j1",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "kp_f1", "vs": "c", "min_effect": 0.05}
 ],
 "budget_gpu_h": 3.0,
 "kill": "a loss that is not finite; the paint sidecar missing for a TRAIN frame (the loader counts `paint_missing`); the score of an arm below the control by more than 0.05 on both seeds after 3000 steps (CUT of that arm)",
 "evaluate": {"split": "val"}
}
```
**By hand (the tools cannot read keypoint cards; the numbers come from `evaluation/jmaps_eval.py score`, label "by hand: keypoint scorer"):** `kp_f1` = the mean of the entry F1 and the exit F1 at tolerance 3.0 m, threshold 0.3, DISTANCE ONLY (the control has no heading), VAL 873 frames. KEEP of an arm: `kp_f1` above c by more than 0.05 on BOTH seeds. TIE: inside +-0.05: **not discarded (D82)**: the arm stays the base and is refined in a follow-up card; CUT: below c by more than 0.05 on both seeds. The base for the later steps is the arm with the highest mean `kp_f1`; on a tie inside 0.01 the cheaper one (fewer layers). Reported, not judged: entry / exit recall and precision at 1.5 m and 3 m, with and without the 30 degree heading condition, the median heading error, zone and core IoU, stop-line / crossing precision and recall (tolerance one 0.5 m cell), ms of the head.

## 1. Why this step first (D82)
E283: the junction blur is the unknown pair of entry and exit; the delivered head's flags miss 25 - 33 % of the GT ends (E284: end recall 0.58 on node tokens that were trained for nodes). The pixel evidence that tells an entry from a plain lane is the painted stop line (nuScenes map expansion), the crossing, the lane markings ending and the widening of the road; none of it was a training target. E285 measured: only 38.5 % of the entries are painted (a stop line other than a turn stop within 3 m, or a crossing within 4 m; exits 28.3 %), so about 61 % of the entries have to be inferred from context: hence the wide context arms; the scorer reports painted and unpainted recall separately. The conflict core (about 40 % of the zone) is the tighter definition of "inside the junction" for the node suppression later.

## 2. Design
Three arms on the frozen B pyramid (so the maps are read from B's own features; the nodes / links of B stay valid), two seeds each (the seed of B's pyramid and of the head are the same: `--base runs/e273_p_s<S>/model.pt`). TRAIN 8414 frames (the paint sidecars exist for all), VAL 873 for the score; TEST never. A joint arm (the pyramid trained together with the head, B's pixel loss kept) is the follow-up when the frozen arms plateau (`jmaps_train.py --joint`; it changes the pixel features, so B's later stages would have to be retrained: judged on the maps only).
**Prediction (a number):** `kp_f1` +0.15 over the control for the dilated arm (the control's event heat maps were trained with ev_w 1 on a coarser target and without paint, their F1 on the E283 ends is about 0.45 - 0.55; unknown on this scorer: the score of c comes with the first run).

## 3. Command and records
`wsl python3 ../autoresearch/tools/run.py --id E286.j1_s0 --hyp ... --pred ... --rule ... --require-card -- bash tools/env/_py14.sh jmaps_train.py --name e286_j1_s0 --seed 0 --ctx dil` (and j0 / j2, seeds 0 / 1), then `jmaps_eval.py score --run e286_j1_s0`. Result files: `runs/e286_<arm>_s<S>/jmaps_val.json`. GPU: nights 22 - 07 and weekdays 08:30 - 17:30 only (D57 / D61).

## 4. Pictures
`tools/viz/junction_view.py frame <scene:idx> --out DIR --pred e286_j1_s0`: the GT layer next to the head's heat / keypoints / zone / core / stop lines.

## 5. The control c, measured before the first run (2026-11-11, CPU, all 873 VAL frames, `jmaps_eval.py score` on an untrained head: its B columns are the control)
B's own stop / exit event heat maps read as keypoints (3 x 3 peaks above 0.3, tolerance 3.0 m, distance only): **entry recall / precision 0.548 / 0.323 (F1 0.406), exit 0.546 / 0.307 (F1 0.393): `kp_f1` of c = 0.400** (threshold 0.5: 0.527 / 0.344 and 0.526 / 0.326). Painted ends are found far more often than unpainted ones: entries 0.651 against 0.483, exits 0.692 against 0.488 (17 - 20 points): B's maps lean on the paint. The prediction of the card (+0.15) means `kp_f1` about 0.55 for the dilated arm; KEEP needs above 0.45 on both seeds. The scorer also reports recall by distance from the ego (< 20 / 20 - 35 / > 35 m) and by city (E300: the far range and Singapore are the holes of the E284 head).

## 6. First result: arm j1 (dilated context), seed 0 (2026-11-11, CPU score of the final checkpoint, VAL 873, `runs/e286_j1_s0/jmaps_val.json`; seed 1 and the arms j0 / j2 / jj follow from the night chain and are scored by it)
| VAL 873, tolerance 3.0 m, thr 0.3 | entry R / P (F1) | exit R / P (F1) | `kp_f1` (distance only) |
|---|---|---|---|
| **c: B's stop / exit event heat** | 0.548 / 0.323 (0.406) | 0.546 / 0.307 (0.393) | **0.400** |
| **j1 s0, with the 30 degree heading condition** | 0.615 / 0.610 (0.612) | 0.637 / 0.616 (0.626) | - |
| **j1 s0, distance only (the card's number)** | 0.625 / 0.620 (0.622) | 0.649 / 0.628 (0.638) | **0.630** |
Median heading error at the matched keypoints 3.3 degrees. Maps: zone IoU 0.760 (F1 0.864), conflict core IoU 0.571 (F1 0.727), stop line P / R 0.723 / 0.848, crossing P / R 0.869 / 0.906 (one 0.5 m cell of tolerance).
**By the rule (seed 0): KEEP-level, +0.230 over c (the KEEP line was +0.05, the prediction +0.15); the verdict needs seed 1 and the other arms.** Where the gain is (recall of the GT ends, thr 0.3, 3 m): painted entries 0.622 against B's 0.651, **unpainted 0.610 against B's 0.483 (+13 points)**: the head no longer depends on paint (painted / unpainted differ by 1 - 4 points, B's by 17 - 20); far range (> 35 m) 0.551 / 0.577 against B's 0.423 / 0.430 (+13 / +15 points), within 20 m 0.707 / 0.718 against 0.710 / 0.684 (equal); Boston 0.672 / 0.693 against 0.629 / 0.592, Singapore 0.564 / 0.588 against 0.474 / 0.505 (+9 / +8 points); the far range and Singapore stay the weakest.
**Honest limits:** (a) c is a weak control (B's event maps were never trained for this); against the E284 end head (which reads node tokens; its own count at p >= 0.5, 3 m / 30 degrees with its matching rule: entry recall 0.57 / 0.58, precision 0.77 / 0.79) the maps head has a little more recall (0.608 / 0.631) and less precision (0.645 / 0.652 at thr 0.5): the definitions of the two counts are not identical (one-to-one greedy at 3 m here), so this is a hint, not a comparison; E287's keypoints from these maps will be scored by the head's own counter (`jends_eval.py dump`) on the same footing; (b) the end recall of about 0.62 is a plateau: the lever is not the paint but the far range / city; (c) the heading is learned well (3.3 degrees).

## 7. Result of the three context arms, two seeds each (2026-11-12 01:44, `jmaps_eval.py score` of the final checkpoints with the FIXED scorer of the critic pass, VAL 873, `runs/e286_<arm>_s<S>/jmaps_val.json`; arm jj follows from the night chain)
| arm | kp_f1 (thr 0.3, 3 m, distance only) s0 / s1 | mean | best-threshold F1 s0 / s1 | zone IoU s0 / s1 | core IoU s0 / s1 | entry recall > 35 m s0 / s1 | entry recall Singapore s0 / s1 | entry recall crowded / isolated (s0) |
|---|---|---|---|---|---|---|---|---|
| **c: B's event heat** | 0.400 / 0.377 | **0.388** | 0.416 / 0.388 | - | - | 0.423 / 0.388 | 0.474 / 0.450 | 0.612 / 0.510 |
| **j0** | 0.580 / 0.554 | **0.567** | 0.617 / 0.606 | 0.721 / 0.718 | 0.483 / 0.475 | 0.574 / 0.567 | 0.570 / 0.571 | 0.660 / 0.608 |
| **j1** | 0.630 / 0.642 | **0.636** | 0.653 / 0.666 | 0.760 / 0.759 | 0.571 / 0.578 | 0.551 / 0.562 | 0.564 / 0.592 | 0.542 / 0.657 |
| **j2** | 0.640 / 0.636 | **0.638** | 0.662 / 0.663 | 0.764 / 0.762 | 0.589 / 0.583 | 0.559 / 0.565 | 0.591 / 0.577 | 0.561 / 0.653 |
Gain of `kp_f1` over c (B's event heat on the same seed): **j0 +0.180 / +0.177, j1 +0.231 / +0.265, j2 +0.240 / +0.259.** The best-threshold F1 (both maps at their own best threshold, the calibration question of the critic) gives the same order: j0 0.61, j1 0.66, j2 0.66, B 0.40.
**By the rule (written before the run, "by hand: keypoint scorer"): KEEP for all three arms on both seeds** (the KEEP line was +0.05, the prediction +0.15). **Base for the later steps: j1** - j2 has the highest mean (0.638 against 0.636) but the card says a tie inside 0.01 goes to the cheaper arm: the attention layer over 25 x 25 tokens adds nothing measurable (+0.002; zone IoU +0.003, core IoU +0.008) for its cost. Context does matter: j0 (1 x 1 / 3 x 3 convolutions only) loses 0.07 against j1 and 0.1 of zone IoU.
**What the splits add (not part of the rule):** (a) the far range (> 35 m) and Singapore stay the weakest (entry recall 0.55 - 0.56 and 0.56 - 0.59) for every arm; (b) **context costs the crowded ends**: j0 finds entries whose neighbour lane's entry is within 4 m / 30 degrees at 0.660 (isolated 0.608), j1 / j2 at 0.54 - 0.56 (isolated 0.65 - 0.66): the dilated context smooths the heat so that two ends of neighbouring lanes merge into one peak; and j0 is better in the far range (0.574 against 0.551); this is the question of E307 (narrower Gaussian targets) and a reason to read the arms by split; (c) the head no longer depends on paint (painted / unpainted recall differ by 1 - 4 points, B's by 17 - 20); (d) the numbers of section 6 (first scorer version) were within 0.01 of these: the critic's fixes did not change the verdict, only the splits (exits painted by a crossing only, crowded split, threshold sweep).
**Honest limits:** c is a weak control (B's event maps were never trained for this); the targets are the legacy half-cell shifted polygons (E306 asks what the exact ones change); zone / core IoU is measured on those targets too.

## 8. The joint arm jj: stopped at step 1000 of 3000 (2026-11-12 07:55, by hand)
`jmaps_train.py --joint` (pyramid of B trained together with the head, B's pixel loss kept), two lanes in parallel, started 04:44: **8 s / step at step 100, 11.1 - 11.3 s / step from step 700** (the frozen arms: 0.84 s / step), the GPU at 11.3 GB with both lanes and 100 % busy. At step 900 (07:34) the loss was 4.45 (s0) / 4.39 (s1); the remaining time at that speed was about 6.4 h per lane, and the day chain (E304, E298, E303, E307, E306, E305, E302) waited behind it. I stopped both lanes (SIGTERM, exit 143) right after the step-1000 checkpoint (`runs/e286_jj_s0/ckpt.pt`, `runs/e286_jj_s1/ckpt.pt`, 07:54); **no score, no verdict: the arm is not judged**. Nothing in the card changes: jj stays the follow-up "when the frozen arms plateau". If it is wanted again: one lane alone (not two on the GPU at once), resumed from the checkpoint, and look at the speed first (a joint step is 13 times a frozen step here, the cause is not measured). q2 (E287's head on the jj maps) did not run for the same reason.
