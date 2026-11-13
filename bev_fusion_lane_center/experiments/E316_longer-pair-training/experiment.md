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

## 2. Result (VAL 873 frames, two seeds; read from the result files on 2026-11-13 04:55 by the lane lead)
Runs: `E316.p_s<S>` (`jends_train.py --e2e --scales --maps-w 1.0 --steps 6000` on `runs/e307_n_s<S>/model.pt`, 01:32 - 04:01, 1.7 - 1.9 s/step in the two slots next to each other); post scoring by `temp/day11` (snapped p >= 0.1, plain, pair PR, maneuver), the not snapped p >= 0.1 read-out afterwards (`temp/e316/unsnap_p01.sh`), the kill check on the maps afterwards (`temp/e316/kill_maps.sh`, `kill_check.py`). Control = E312 of the same seed (the same E307 maps, 3000 steps); every E312 number is read from `runs/e312_p_s<S>/` (cards, `ends_head.json`, `maneuver_val.json`).

| VAL | E312 s0 | **E316 s0** | E312 s1 | **E316 s1** |
|---|---|---|---|---|
| **snapped p >= 0.1: topo F1** | 0.6825 | **0.6898** | 0.6864 | **0.6938** |
| over the plain graph | +0.0566 | +0.0639 | +0.0615 | +0.0689 |
| **E316 - E312 (the rule)** | | **+0.0073** | | **+0.0074** |
| topo precision / recall | 0.6804 / 0.6845 | 0.6888 / 0.6908 | 0.6863 / 0.6866 | 0.6979 / 0.6897 |
| lanes found | 0.7376 | 0.7423 | 0.7423 | 0.7408 |
| junction F1 (guard -0.03) | 0.3194 | 0.3222 | 0.3143 | 0.3196 |
| dup_frac (guard +0.01) | 0.1747 | 0.1678 (-0.0069) | 0.1557 | 0.1532 (-0.0025) |
| F1@1.0 (guard -0.01) | 0.8669 | 0.8703 | 0.8685 | 0.8714 |
| vertices / 100 m (guard +1.0) | 19.55 | 19.51 | 20.10 | 19.99 |
| not snapped p >= 0.1: topo F1 | 0.6438 | 0.6503 (+0.0065) | 0.6481 | 0.6588 (+0.0107) |
| maneuvers realised, p0.1 snapped | 0.7069 | 0.7064 | 0.7008 | 0.7070 |
| entry / exit recall of the keypoints (pair run) | 0.745 / 0.771 | 0.750 / 0.763 | 0.752 / 0.767 | 0.744 / 0.765 |
| pair recall (logit >= 0) / precision | 0.298 / 0.743 | 0.355 / 0.758 | 0.328 / 0.724 | 0.328 / 0.742 |
| precision at E312's recall (`jpair_pr.py`) | | 0.784 (+0.041) | | 0.742 (+0.018) |
| maps after the joint training: keypoint F1 (3 m, thr 0.3) | 0.6965 (E307 0.7233) | 0.6861 (-0.0372 vs E307) | 0.6868 (E307 0.7051) | 0.6832 (-0.0219 vs E307) |

**Verdict (by hand: replay, the rule of section 0): TIE.** KEEP needed more than +0.01 topo F1 over E312 on both seeds; the result is +0.0073 / +0.0074 (CUT would be below -0.01), all guards held (dup_frac is lower than E312's on both seeds, F1@1.0 higher). The kill criterion did not fire: the loss stayed finite and the maps after the joint training are 0.037 / 0.022 keypoint F1 below E307 (kill at 0.05). The shorter E312 recipe (pair stage 3000 steps) stays the system of record, because on a tie the cheaper variant wins; the E316 checkpoints are kept (TIE = kept and refined).

**What it shows.** The gain has the same sign and size on both seeds (+0.007) and comes from the precision side (topo precision +0.008 / +0.012, pair precision at the recall of E312 +0.041 / +0.018); the prediction of the card was right for the pair precision (+0.02 predicted) and just below the edge for the graph (+0.010 predicted, +0.0073 / +0.0074 measured). The not snapped read-out gains about as much (+0.0065 / +0.0107); the snapped minus not snapped gap (0.0395 / 0.0350 for E316, 0.0387 / 0.0383 for E312) stays about the same, so the longer pair stage does not remove the dependence on the snap's wiring (E320 asks that). A side effect: the joint training lowers the keypoint F1 of the maps against E307 (E312 -0.027 / -0.018, E316 -0.037 / -0.022) while the pair level and the delivered graph rise; the pair head trades some map sharpness for the pairing. With E315 (the maps twice as long, -0.004) both halves of the training length are inside the band: the length of training is not the lever for the delivered graph, the wiring is. **Compute:** none at run time (the same model); training 3.2 h instead of 1.6 h per seed for the pair stage. A free add-on candidate that may be stacked with the E320 recipe later (stack card after the E320 result, not before). Not run: the lenient (2 m) topo F1 of `temp/e309/sens.py` and the paired scene bootstrap E316 - E312 (E312's snap cards have no `.counts.pkl` until `day12e_rescore.sh` has written them after E320).
