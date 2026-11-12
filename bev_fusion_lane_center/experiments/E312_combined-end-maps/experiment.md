# E312 - combined end maps (narrow heat + far-range weight) and the E305 pair head on them, read out snapped at p >= 0.1

E310 / E311 (2026-11-12): on the snapped read-out the pair decision of E305 is at the level of a perfect pair classifier (topo F1 +0.0304 at p >= 0.1 against +0.0314 for the oracle on the same keypoints). What is left is the ENDS: only 52 % of the GT pairs have both ends matched strictly by a keypoint, 69 % loosely. The single-factor maps arms E307 (heat sigma 1.0 m: crowded ends) and E304 (far-range weighted heat loss: ends beyond 35 m, 53 % of the GT ends) attack that from two sides; this card puts the two together and carries the pair head on top. The composition of the arm was fixed before the single-factor results (E307 / E304 run first in the same night chain; if one of them is a CUT, the arm is built without it and the card is amended with the reason before its runs start).

## 0. The card (what the tools run)
```json card
{
 "id": "E312",
 "card": "",
 "title": "junction maps head with heat sigma 1.0 m and far-range weight 2 (arm m2), then the E305 pair head (maps and pairs trained together) on it, against the E305 pair head on the maps of E286 j1; delivered with snapped connector ends at p >= 0.1 (E311)",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.01, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E305 of the same seed (maps of E286 j1, `--e2e`), snapped connectors at p >= 0.1: topo F1 over the plain graph +0.0304 / +0.0285"},
  "m": {"recipe": "maps `jmaps_train.py --ctx dil --sigma 1.0 --far-w 2 --steps 3000` (run e312_m2_s<S>), then `jends_train.py --e2e --maps-w 1.0 --steps 3000` on them (run e312_p_s<S>, BFLC_SIG=1.0 so that the heat targets of the joint training match the maps); same read-out"}
 },
 "control": "c",
 "variant": "m",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "min_effect": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.01},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 4.0,
 "kill": "the maps arm is worse than E286 j1 on the keypoint F1 by more than 0.03 on either seed (the combination does not train)",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: replay" for the topology numbers: `jends_eval.py score --snap --thr 0.1`, differences to the plain graph of the same seed, then the difference to the control): KEEP when topo F1 rises by more than +0.01 over the control on BOTH seeds with the guards held (this means topo F1 over the plain graph above +0.040 / +0.039); TIE kept and refined (D82); CUT below -0.01 or a broken guard or the kill. Reported next to it: the maps (`jmaps_eval.py score` on the m2 maps and on the maps after the joint training: keypoint F1, entry recall by range / crowded, zone IoU), the maneuver coverage (`jmaneuver.py`, p0.1s), pair recall / precision, the share of GT pairs with both ends matched (`jrule_control.py` ceiling).

## 1. Order of work
After E307 / E304 (day chain 4) the chain slot of E306 / E298 / E302 is free (their start scripts are stubs, the originals are `.orig`: the pair decision is at the oracle level, E306 changes only the targets). `temp/day5/day5.sh` runs, per lane (seed), the maps arm, its keypoint score, the joint pair training, the dump, the snapped score and the maneuver coverage.
