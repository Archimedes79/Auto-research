# E316 - the E312 pair head trained twice as long (6000 steps) on the E307 maps

Card of 2026-11-12 21:30, written by the lane chat. The pair head of E312 is still improving at step 3000 (total loss 1.69 -> 1.59 over the last 500 steps; see E315 for the curves and the reason). E315 asks the maps stage, this card the pair stage, single factor: the same maps (`runs/e307_n_s<S>`), the same recipe, only `--steps 6000`.

## 0. The card (what the tools run)
```json card
{
 "id": "E316",
 "card": "",
 "title": "pair head `--e2e --scales --maps-w 1.0` trained 6000 steps on the E307 maps (BFLC_SIG=1.0), snapped read-out at p >= 0.1, against the same recipe at 3000 steps (E312)",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.01, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E312 of the same seed (3000 steps): snapped p >= 0.1 topo F1 over the plain graph +0.0566 / +0.0615 (0.6825 / 0.6864)"},
  "l": {"recipe": "`jends_train.py --e2e --scales --maps-w 1.0 --steps 6000` on `runs/e307_n_s<S>/model.pt`, BFLC_SIG=1.0 (run e316_p_s<S>); same read-out"}
 },
 "control": "c",
 "variant": "l",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "min_effect": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.01},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 7.0,
 "kill": "the loss is not finite, or the maps after the joint training are worse than E307 by more than 0.05 keypoint F1 on either seed",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: replay"): `jends_eval.py score --snap --thr 0.1`, differences to the plain graph of the seed, then to E312. KEEP when topo F1 over E312 rises by more than +0.01 on BOTH seeds with the guards held; TIE kept (the shorter recipe stays the cheaper one); CUT below -0.01. Reported next to it: entry / exit recall, pair recall / precision, the maneuver coverage (`jmaneuver.py`, p0.1s), the lenient topo F1 (tolerance 2 m, `temp/e309/sens.py`, scratch).

## 1. Design and prediction
Two seeds, VAL 873 frames, TEST untouched. **Prediction (a number):** topo F1 +0.010 over E312 (edge of the band); pair precision at the recall of E312 +0.02. It runs after E315 in the same two GPU slots (3.2 h at about 1.9 s per step); E315 and E316 do not share a checkpoint.
