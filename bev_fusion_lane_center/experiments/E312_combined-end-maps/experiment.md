# E312 - combined end maps (narrow heat + far-range weight) and the E305 pair head on them, read out snapped at p >= 0.1

E310 / E311 (2026-11-12): on the snapped read-out the pair decision of E305 is at the level of a perfect pair classifier (topo F1 +0.0304 at p >= 0.1 against +0.0314 for the oracle on the same keypoints). What is left is the ENDS: only 52 % of the GT pairs have both ends matched strictly by a keypoint, 69 % loosely. The single-factor maps arms E307 (heat sigma 1.0 m: crowded ends) and E304 (far-range weighted heat loss: ends beyond 35 m, 53 % of the GT ends) attack that from two sides; this card puts the two together and carries the pair head on top. The composition of the arm was fixed before the single-factor results (E307 / E304 run first in the same night chain; if one of them is a CUT, the arm is built without it and the card is amended with the reason before its runs start).

## 0. The card (what the tools run)
```json card
{
 "id": "E312",
 "card": "",
 "title": "junction maps head with heat sigma 1.0 m (E307; the first version of this card added a far-range weight 2, see amendment 2), then the E305 pair head (maps and pairs trained together) with the learned handle scales of E303 on it, against the E305 pair head on the maps of E286 j1; delivered with snapped connector ends at p >= 0.1 (E311)",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.01, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E305 of the same seed (maps of E286 j1, `--e2e`), snapped connectors at p >= 0.1: topo F1 over the plain graph +0.0304 / +0.0285"},
  "m": {"recipe": "maps `jmaps_train.py --ctx dil --sigma 1.0 --far-w 2 --steps 3000` (run e312_m2_s<S>), then `jends_train.py --e2e --scales --maps-w 1.0 --steps 3000` on them (run e312_p_s<S>, BFLC_SIG=1.0 so that the heat targets of the joint training match the maps); the learned handle scales are used by the read-out; same read-out as the control"}
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

**Amendment 2026-11-12 11:25 (before any run of this card):** the pair head gets the handle scales of E303 (`--scales`). On the snapped read-out the scales are worth more than on the unsnapped one: E303 seed 1, same pairs, p >= 0.1: topo F1 +0.0356 with the learned scales against +0.0251 with the chord tangent (F1@1.0 +0.0363 against +0.0298); the seed-0 control was still being scored when this amendment was written. The control c stays E305 (no scales), so the arm changes TWO things (maps m2, scales); the reference points E303 (scales, frozen maps: +0.0320 / +0.0356 over the plain graph) and E305 (e2e, no scales: +0.0304 / +0.0285) are given in the result so that the two effects can be read apart. The scales make the pair training slower (the fit of the GT scales: 1.9 s / step instead of 1.2).

**Amendment 2 2026-11-12 12:00 (before any run of this card):** E307 (heat sigma 1.0 m alone, run 10:57 - 11:48, scored 11:56) is a KEEP by its own rule and far larger than predicted: keypoint F1 0.723 / 0.705 against 0.630 / 0.642 of E286 j1, crowded entry recall 0.789 / 0.789 against 0.542 / 0.576, entry recall beyond 35 m 0.652 / 0.651 against 0.551 / 0.562, zone IoU +0.007 / +0.002 (E307 card). The far-range weight of E304 was stopped at step 200 (a single-factor test on the old sigma is no longer the question; the arm built from both is round 2, card E313 if time allows). Arm m of this card is therefore: the pair head `--e2e --scales` on the E307 maps (sigma 1.0, no far-range weight) instead of the maps m2; the rule, the control c (E305) and the read-out are unchanged. The maps arm m2 (sigma 1.0 + far-w 2) stays the follow-up.

## 1. Order of work
After E307 / E304 (day chain 4) the chain slot of E306 / E298 / E302 is free (their start scripts are stubs, the originals are `.orig`: the pair decision is at the oracle level, E306 changes only the targets). `temp/day5/day5.sh` runs, per lane (seed), the maps arm, its keypoint score, the joint pair training, the dump, the snapped score and the maneuver coverage.
