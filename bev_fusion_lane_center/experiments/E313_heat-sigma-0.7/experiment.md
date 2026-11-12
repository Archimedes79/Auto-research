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
By hand (`jdiag.py maps e313_s e307_n`, the numbers of `jmaps_eval.py score`): KEEP when the keypoint F1 rises by more than +0.01 on BOTH seeds with the crowded entry recall not lower by more than 0.03 and the zone IoU within 0.01; TIE (kept; D82: the narrower or the wider arm is kept by the cost: equal, E307) when inside the band; CUT when it falls by more than 0.01. The keypoint F1 is read at a fixed threshold 0.3: a narrower heat is lower one metre from the end, so the best-threshold F1 is reported next to it (a fair read).

## 1. Result (2026-11-12 14:10; trained 13:08 - 14:02, both seeds; `jmaps_eval.py score --run e313_s_s<S>`, VAL 873; `jdiag.py maps e313_s e307_n`)
| seed | arm | keypoint F1 | best-threshold F1 | entry recall crowded / isolated | entry recall > 35 m | zone IoU | core IoU |
|---|---|---|---|---|---|---|---|
| 0 | c = E307 (sigma 1.0) | 0.723 | 0.737 | 0.789 / 0.714 | 0.652 | 0.767 | 0.584 |
| 0 | **E313 (sigma 0.7)** | **0.743** (+0.020) | 0.749 (+0.012) | 0.856 (+0.068) / 0.691 (-0.023) | 0.652 (+0.000) | 0.766 (-0.001) | 0.591 |
| 1 | c = E307 (sigma 1.0) | 0.705 | 0.727 | 0.789 / 0.710 | 0.651 | 0.761 | 0.581 |
| 1 | **E313 (sigma 0.7)** | **0.732** (+0.027) | 0.737 (+0.010) | 0.844 (+0.054) / 0.676 (-0.034) | 0.639 (-0.011) | 0.761 (+0.000) | 0.584 |
**Verdict by the rule: KEEP** (keypoint F1 +0.020 / +0.027 against +0.01, crowded entry recall +0.068 / +0.054 (not lower), zone IoU within 0.01). Honest limits: the best-threshold F1 gains only +0.012 / +0.010 (part of the fixed-threshold gain is calibration of the narrower heat), the recall of ISOLATED entries falls by 0.023 / 0.034 and the far range does not move: the narrower target separates crowded ends and costs some isolated ones. Whether this lifts the delivered graph is the question of E314 (the E312 pair head on these maps).
