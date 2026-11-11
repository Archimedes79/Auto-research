# E306 Exact polygon targets for the junction maps: the area rule instead of the half-cell shifted fill (critic finding, D82 thread)

Card of 2026-11-12 (plan card). The critic of 2026-11-11 found that `cv2.fillPoly` with rounded vertices puts integer coordinates on pixel CENTRES while a cell of our grids spans [k, k + 1): every zone, conflict-core, stop-line and crossing target is shifted by half a cell (0.5 m on the 1 m grid, 0.25 m on the 0.5 m paint grid) and its outline is jagged. Measured on 60 random convex polygons (`temp/gt2/t_raster_num.py`): of 3125 cells that lie more than half inside the polygon, the legacy fill gets 1122 wrong (36 %), the new area rule (`jgt.fill_poly`, `BFLC_RASTER=exact`, 8 x 8 supersampling) 71 (2.3 %). The maps head learns the shifted target, so its zone edge is where the legacy fill puts it, not where the road is; the evidence along a pair curve (zone and core samples of E287) reads the same shifted maps. The change is opt-in (`--raster exact`, default `legacy`): the stored numbers of E285 - E304 stay reproducible.

## 0. The card (what the tools run)
```json card
{
 "id": "E306",
 "card": "",
 "title": "the junction maps head (E286 j1) trained and scored on exact polygon targets (zone, conflict core, stop lines, crossings) against the legacy half-cell shifted ones",
 "kind": "block",
 "deterministic": false,
 "metric": "zone_iou",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.03, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E286 j1 of the same seed (legacy targets), scored on the EXACT targets (`jmaps_eval.py score --run e286_j1_s<S> --raster exact`): what the legacy-trained head gets on the corrected exam"},
  "x": {"recipe": "`jmaps_train.py --ctx dil --raster exact`: the same head and budget as j1, trained on the exact targets and scored on them"}
 },
 "control": "c",
 "variant": "x",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "zone_iou", "vs": "c", "min_effect": 0.02},
  {"metric": "core_iou", "vs": "c", "max_loss": 0.02},
  {"metric": "kp_f1", "vs": "c", "max_loss": 0.01}
 ],
 "budget_gpu_h": 1.0,
 "kill": "a loss that is not finite; the exact-target scorer gives a zone IoU of the control below 0.5 (the targets are broken, not the head)",
 "evaluate": {"split": "val"}
}
```
By hand (the verdict numbers come from `jmaps_eval.py score`; label "by hand: maps scorer", D79-style guards): KEEP when the zone IoU on the exact targets rises by more than 0.02 on both seeds with the core IoU and the keypoint F1 held; TIE kept and refined (D82): the exact raster is then the default for every later maps run because it is the same cost and the targets are right; CUT below.

## 1. What changes (default off)
`jgt.fill_poly` is the one polygon rasteriser (`jgt.fields` for the 1 m zone / core, `qjunction._fill` for the 0.5 m stop lines / crossings); `BFLC_RASTER=exact` selects the area rule (a cell is in when more than half of it is inside, thin paint: more than a quarter). `jmaps_train.py` / `jends_train.py --raster exact` set it before the loader exists (the workers inherit it), `jmaps_eval.py score` takes the run's own raster or `--raster`. Self-test: `python jgt.py` (a box, 20 random convex polygons against shapely).

## 2. Design
Two seeds, 3000 steps, lr 1e-3, batch 8, the context arm of j1; VAL 873 frames; TEST untouched. **Prediction (a number):** zone IoU +0.03 over c (the control is scored on targets it was not trained on, its edge is half a cell off by construction); core IoU about the same; keypoint F1 unchanged (the end heat does not use the polygon fill). It is a cheap correctness arm: if the exact-trained head only matches the legacy one on its own exam, the legacy numbers of E286 are a fair read of the head and the shift only cost the evidence features a little.

## 3. Command and records
`wsl bash temp/e306/start_exact.sh <seed>` (scratch wrapper around `run.py --require-card`), then `jmaps_eval.py score --run e306_x_s<S>` and `score --run e286_j1_s<S> --raster exact` (written next to the E286 result as `jmaps_val_exact.json`, the run's own `jmaps_val.json` is never overwritten). GPU: nights 22 - 07 and weekdays 08:30 - 17:30 only.
