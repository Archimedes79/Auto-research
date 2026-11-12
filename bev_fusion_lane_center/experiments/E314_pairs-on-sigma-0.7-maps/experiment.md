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
