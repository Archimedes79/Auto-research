# E307 Narrower end heat for crowded lanes: Gaussian targets of 1.0 m instead of 1.5 m (found in the first fixed scorer output of E286 j1, D82 thread)

Card of 2026-11-12 (plan card). The scorer of E286 (after the critic's fixes) splits the GT ends into CROWDED ones (another end of the same kind within 4 m and 30 degrees of heading: the neighbouring lane) and isolated ones. E286 j1 seed 0, keypoints at threshold 0.5, tolerance 3 m, distance only: entry recall **0.542 crowded against 0.657 isolated**, exit 0.511 against 0.689 (n 3140 / 5394 entries, 2383 / 5730 exits); B's own event heat, which has no such targets, is the other way round on the entries (0.612 crowded, 0.510 isolated). A lane is 3.5 m wide and the end Gaussians have sigma 1.5 m (E285 `jgt.SIG`): two neighbours 3.5 m apart are 2.3 sigma apart, their sum has a shallow saddle, and the 3 x 3 peak read-out finds one keypoint where there are two. The lever is cheap: the width of the target.

## 0. The card (what the tools run)
```json card
{
 "id": "E307",
 "card": "",
 "title": "the junction maps head (E286 j1) trained with entry / exit heat targets of sigma 1.0 m (the heading is supervised within 2 sigma of an end) against sigma 1.5 m",
 "kind": "block",
 "deterministic": false,
 "metric": "kp_f1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.02, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E286 j1 of the same seed (sigma 1.5 m), the stored scorer numbers"},
  "n": {"recipe": "`jmaps_train.py --ctx dil --sigma 1.0`: the same head and budget as j1, the heat targets of the entry and exit ends narrower (1.0 m), the heading cells within 2 sigma = 2 m"}
 },
 "control": "c",
 "variant": "n",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "kp_f1", "vs": "c", "min_effect": 0.01},
  {"metric": "kp_recall_crowded", "vs": "c", "min_effect": 0.04},
  {"metric": "zone_iou", "vs": "c", "max_loss": 0.01}
 ],
 "budget_gpu_h": 1.0,
 "kill": "a loss that is not finite; the entry recall at threshold 0.5 falls below 0.5 on a seed (the narrow targets are not found at all)",
 "evaluate": {"split": "val"}
}
```
By hand (numbers from `jmaps_eval.py score`: `kp_ours_tol3.0_noang_*` at threshold 0.5, the crowded split, the best-threshold F1; label "by hand: maps scorer"): KEEP when the mean entry / exit keypoint F1 rises by more than 0.01 AND the crowded recall by more than 0.04 on both seeds with the zone IoU held; TIE kept and refined (D82; sigma 1.25 is the refinement); CUT below. Not judged by the best-threshold F1 alone: a narrower Gaussian changes the calibration of the heat, so the fixed-threshold numbers and the sweep are both reported.

## 1. What changes (default off)
`jgt.fields` reads the width from `BFLC_SIG` (default 1.5), `jmaps_train.py --sigma` sets it before the loader exists (the workers inherit it). The pair head reads the heat through its soft-argmax over 3 x 3 cells: a narrower peak means a larger share of the mass in the centre cell, the position is as before. Self-test: `python jgt.py` (a narrower Gaussian is lower two metres from the end).

## 2. Design
Two seeds, 3000 steps, lr 1e-3, batch 8, the context arm of j1, VAL 873 frames, TEST untouched. **Prediction (a number):** kp F1 +0.02 and crowded entry recall +0.06 (0.54 -> 0.60) over c; the isolated ends lose at most 0.01. If the saddle is not the cause (the crowded ends are also the far ones), the crowded recall stays and the card is a CUT that says so.

## 3. Command and records
`wsl bash temp/e307/start_narrow.sh <seed>` (scratch wrapper around `run.py --require-card`), then `jmaps_eval.py score --run e307_n_s<S>`. GPU: nights 22 - 07 and weekdays 08:30 - 17:30 only.

## 2. Result (2026-11-12 11:56; trained 10:57 - 11:48, both seeds, 3000 steps; `jmaps_eval.py score --run e307_n_s<S>` -> `runs/e307_n_s<S>/jmaps_val.json`, VAL 873, the control c = E286 j1 of the same seed (frozen pyramid, dilated context); comparison script `temp/gt2/maps_cmp.py`)
| seed | arm | keypoint F1 (3 m, thr 0.3) | best-threshold F1 | entry recall crowded / isolated | entry recall > 35 m | zone IoU | core IoU |
|---|---|---|---|---|---|---|---|
| 0 | c = E286 j1 | 0.630 | 0.653 | 0.542 / 0.657 | 0.551 | 0.760 | 0.571 |
| 0 | **E307 sigma 1.0** | **0.723** (+0.093) | 0.737 (+0.084) | **0.789** (+0.247) / 0.714 (+0.057) | 0.652 (+0.101) | 0.767 (+0.007) | 0.584 |
| 1 | c = E286 j1 | 0.642 | 0.666 | 0.576 / 0.674 | 0.562 | 0.759 | 0.578 |
| 1 | **E307 sigma 1.0** | **0.705** (+0.063) | 0.727 (+0.061) | **0.789** (+0.213) / 0.710 (+0.036) | 0.651 (+0.089) | 0.761 (+0.002) | 0.581 |
**Verdict by the rule: KEEP** (keypoint F1 +0.093 / +0.063 against the +0.01 of the rule, crowded entry recall +0.247 / +0.213 against +0.04, zone IoU within 0.01 on both seeds). The prediction (+0.02 kp F1, crowded +0.06, isolated within 0.01) is missed by a wide margin in the good direction. The far range improves as well (+0.10 / +0.09 entry recall beyond 35 m) although the arm has no far-range weight: with a 1.5 m Gaussian the heat of far, blurred ends is a plateau that falls below the peak read-out, a narrower target gives a peak. **Base for the later steps: the E307 maps** (E312 builds the pair head on them; the far-range weight of E304 was stopped at step 200 for that reason). Not yet known: whether the better ends also lift the pairs and the delivered graph (E312).
