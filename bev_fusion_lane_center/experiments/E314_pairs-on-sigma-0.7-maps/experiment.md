# E314 - the E312 pair head on the sigma 0.7 maps

E313 (2026-11-12 14:10): the end heat with sigma 0.7 m lifts the keypoint F1 of the maps over the sigma 1.0 m maps of E307 (+0.020 / +0.027) and the crowded entry recall (+0.07 / +0.05), at the price of the isolated entry recall (-0.02 / -0.03). Does it lift the delivered graph? The same recipe as E312, on the E313 maps.

## 0. The card (what the tools run)
```json card
{
 "id": "E314",
 "card": "",
 "title": "pair head `--e2e --scales` on the E313 maps (heat sigma 0.7 m), snapped read-out at p >= 0.1, against the same pair head on the E307 maps (E312)",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.01, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E312 of the same seed (E307 maps): snapped p >= 0.1 topo F1 over the plain graph +0.0566 / +0.0615"},
  "s": {"recipe": "`jends_train.py --e2e --scales --maps-w 1.0 --steps 3000` on `runs/e313_s_s<S>/model.pt`, BFLC_SIG=0.7 for the heat targets of the joint training (run e314_p_s<S>); same read-out"}
 },
 "control": "c",
 "variant": "s",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "min_effect": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.01},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 2.5,
 "kill": "the maps after the joint training are worse than E313 by more than 0.05 keypoint F1 on either seed",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: replay"): `jends_eval.py score --snap --thr 0.1`, differences to the plain graph of the seed, then to E312. KEEP when topo F1 rises by more than +0.01 over E312 on BOTH seeds with the guards held; TIE kept (D82: E312 on the wider heat has the better isolated recall); CUT below -0.01. Reported next to it: entry / exit recall, pair recall / precision, the maneuver coverage (`jmaneuver.py`, p0.1s).

## 1. Result (2026-11-12 15:30; E314 trained 14:12 - 15:20 on the E313 maps with `--e2e --scales`, BFLC_SIG=0.7; `temp/day9/day9.sh`; `runs/e314_p_s<S>/ends_head.json`, `ends_card*.json`; control c = E312, snapped p >= 0.1, differences to the plain graph of the seed)
| seed | arm | entry / exit recall (p >= 0.5) | pair recall / precision | topo F1 (vs plain) | F1@1.0 | lanes found | junction F1 | junction recall | dup_frac | vert | topo P / R |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | c = E312 | 0.745 / 0.771 | 0.300 / 0.674 | 0.6825 (+0.0566) | +0.0385 | +0.1109 | +0.0481 | +0.1248 | -0.0598 | +0.27 | +0.0107 / +0.0971 |
| 0 | **E314 (sigma 0.7 maps)** | 0.730 / 0.727 | 0.353 / 0.674 | 0.6852 (+0.0593) | +0.0419 | +0.1239 | +0.0476 | +0.1335 | -0.0530 | +0.01 | +0.0085 / +0.1048 |
| 1 | c = E312 | 0.752 / 0.767 | 0.333 / 0.665 | 0.6864 (+0.0615) | +0.0525 | +0.1286 | +0.0650 | +0.1730 | -0.0455 | +0.02 | -0.0065 / +0.1175 |
| 1 | **E314 (sigma 0.7 maps)** | 0.734 / 0.753 | 0.355 / 0.683 | 0.6846 (+0.0597) | +0.0555 | +0.1347 | +0.0620 | +0.1762 | -0.0417 | -0.27 | -0.0156 / +0.1230 |
Unsnapped p >= 0.5 (card of record): E314 topo F1 +0.0148 / +0.0200 (E312: +0.0134 / +0.0180).
**Verdict by the rule: TIE** (topo F1 over E312 +0.0027 / -0.0018 against the +0.01; guards held; label "by hand: replay"). Kept and refined (D82): the narrower heat of E313 finds more crowded ends and raises the pair recall (0.35 against 0.30 / 0.33) and the lanes found (+0.013 / +0.006 over E312), but it finds fewer isolated ends (entry / exit recall 0.73 / 0.73 against 0.75 / 0.77), and the delivered graph ends up where E312 stands. **The final system stays E312** (sigma 1.0 m: the better isolated recall at the same topology score); E314 is the option when lanes found matters more than the junction precision. Prediction (+0.01) missed.
Maneuver coverage (`runs/e314_p_s<S>/maneuver_val.json`, p >= 0.1 snapped): 72.1 % / 71.1 % against 70.7 % / 70.1 % of E312 (+1.4 / +1.0 points; plain graph 52.5 % / 49.1 %); pair precision at the recall of E312 (jpair_pr) +0.018 on seed 0 (0.762 against 0.743). The sigma 0.7 maps cover a little more of the maneuvers and the lanes, the topology score does not move.
