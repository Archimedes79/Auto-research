# E313 - heat sigma 0.7 m for the end maps

E307 (2026-11-12): the narrower end heat, sigma 1.0 m instead of 1.5 m, lifted the keypoint F1 of the junction maps from 0.630 / 0.642 to 0.723 / 0.705, the crowded entry recall from 0.54 / 0.58 to 0.79 and the far entry recall by +0.10, far more than predicted (+0.02). The question this card asks: where does it end? One more step, sigma 0.7 m (1.0 m: 3.5 m apart lanes keep a valley of 0.06; 0.7 m: a single cell). CPU / GPU as E307, maps only (the pair head follows on the winner).

## 0. The card (what the tools run)
```json card
{
 "id": "E313",
 "card": "",
 "title": "junction maps head (dilated context, frozen pyramid of B) trained with end heat targets of sigma 0.7 m, against the same head with sigma 1.0 m (E307)",
 "kind": "block",
 "deterministic": false,
 "metric": "kp_f1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.02, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E307 of the same seed (`jmaps_train.py --ctx dil --sigma 1.0`): keypoint F1 0.723 / 0.705"},
  "s": {"recipe": "`jmaps_train.py --name e313_s_s<S> --ctx dil --sigma 0.7 --steps 3000` (scored by `jmaps_eval.py score`, thr 0.3, 3 m, distance only)"}
 },
 "control": "c",
 "variant": "s",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "kp_f1", "vs": "c", "min_effect": 0.01},
  {"metric": "crowded_entry_recall", "vs": "c", "max_loss": 0.03},
  {"metric": "zone_iou", "vs": "c", "max_loss": 0.01}
 ],
 "budget_gpu_h": 2.0,
 "kill": "a loss that is not finite; the entry heat beyond the first 300 steps stays at the initial value (the narrow target does not train)",
 "evaluate": {"split": "val"}
}
```
By hand (`maps_cmp.py e313_s e307_n`, the numbers of `jmaps_eval.py score`): KEEP when the keypoint F1 rises by more than +0.01 on BOTH seeds with the crowded entry recall not lower by more than 0.03 and the zone IoU within 0.01; TIE (kept; D82: the narrower or the wider arm is kept by the cost: equal, E307) when inside the band; CUT when it falls by more than 0.01. The keypoint F1 is read at a fixed threshold 0.3: a narrower heat is lower one metre from the end, so the best-threshold F1 is reported next to it (a fair read).
