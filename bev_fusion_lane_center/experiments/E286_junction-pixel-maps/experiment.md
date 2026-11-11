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
