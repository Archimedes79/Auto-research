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

## 2. Result (2026-11-12 13:25; E312 pair heads trained 11:58 - 13:08 on the E307 maps with `--e2e --scales`, BFLC_SIG=1.0; dumps and scores by `temp/day6/day6.sh`; `runs/e312_p_s<S>/ends_head.json`, `ends_card.json` (the card of record: unsnapped p >= 0.5), `ends_card_snap_thr<t>.json`; differences to the plain delivered graph of the seed, control c = E305 on the snapped read-out `temp/gt2/snap/e305_e_s<S>_thr0.1.json`)
End and pair numbers of the dumps (p >= 0.5, the head's own matching): entry / exit recall **0.745 / 0.771** (seed 0), 0.752 / 0.767 (seed 1) against 0.645 / 0.637 and 0.660 / 0.671 of E305 and 0.652 / 0.675 of E287 p2; pair recall / precision **0.300 / 0.674** and 0.333 / 0.665 against 0.207 / 0.630 and 0.218 / 0.610 (E305).
| seed | read-out | topo F1 (vs plain) | F1@1.0 | lanes found | junction F1 | junction recall | dup_frac | vert | topo P / R |
|---|---|---|---|---|---|---|---|---|---|
| 0 | E305 (c), snapped p >= 0.1 | 0.6563 (+0.0304) | +0.0195 | +0.0702 | +0.0373 | +0.0927 | -0.0518 | +0.25 | +0.0004 / +0.0556 |
| 0 | E303, snapped p >= 0.1 | 0.6579 (+0.0320) | +0.0240 | +0.0751 | +0.0322 | +0.0848 | -0.0438 | +0.33 | +0.0001 / +0.0590 |
| 0 | **E312, snapped p >= 0.1** | **0.6825 (+0.0566)** | +0.0385 | +0.1109 | +0.0481 | +0.1248 | -0.0598 | +0.27 | +0.0107 / +0.0971 |
| 0 | E312, snapped p >= 0.5 | 0.6612 (+0.0353) | +0.0196 | +0.0493 | +0.0193 | +0.0323 | -0.0388 | +0.18 | +0.0201 / +0.0474 |
| 0 | E312, unsnapped p >= 0.5 (card of record) | 0.6392 (+0.0134) | +0.0188 | +0.0512 | -0.0126 | +0.1021 | -0.0470 | +0.27 | -0.0197 / +0.0413 |
| 1 | E305 (c), snapped p >= 0.1 | 0.6534 (+0.0285) | +0.0275 | +0.0724 | +0.0525 | +0.1329 | -0.0384 | -0.05 | -0.0202 / +0.0662 |
| 1 | E303, snapped p >= 0.1 | 0.6605 (+0.0356) | +0.0363 | +0.0910 | +0.0602 | +0.1559 | -0.0346 | +0.11 | -0.0231 / +0.0825 |
| 1 | **E312, snapped p >= 0.1** | **0.6864 (+0.0615)** | +0.0525 | +0.1286 | +0.0650 | +0.1730 | -0.0455 | +0.02 | -0.0065 / +0.1175 |
| 1 | E312, snapped p >= 0.5 | 0.6639 (+0.0390) | +0.0297 | +0.0630 | +0.0358 | +0.0769 | -0.0367 | +0.32 | +0.0078 / +0.0618 |
| 1 | E312, unsnapped p >= 0.5 (card of record) | 0.6429 (+0.0180) | +0.0303 | +0.0661 | +0.0018 | +0.1485 | -0.0466 | +0.40 | -0.0334 / +0.0581 |
The plain graphs: topo F1 0.6259 (seed 0) / 0.6249 (seed 1), lanes found 0.6267 / 0.6137, junction F1 0.2712 / 0.2493.
**Verdict by the rule: KEEP** (topo F1 over the control E305 on the snapped read-out at p >= 0.1: +0.0262 on seed 0, +0.0330 on seed 1, against +0.01; guards held: F1@1.0 +0.019 / +0.025 over c, dup_frac not higher, vertices +0.27 / +0.02 (limit +1.0 over the plain graph), junction F1 +0.048 / +0.065 over the plain graph). Prediction (+0.01 over E305) exceeded 2.6 - 3.3 times. The kill (maps worse than E286 j1 by more than 0.03) is not touched: the E307 maps are 0.06 - 0.09 better. What it is made of (two changes against the control, see the amendments): the E307 maps (sigma 1.0 m) lift the ends (entry / exit recall +0.10 at p >= 0.5, so more pairs have both ends found); the handle scales add about +0.006 .. +0.011 on the snapped read-out (E303); the snapped read-out itself roughly +0.02 .. +0.025 (E311). Read against the first report of the thread (pair head drawn from the keypoints at p >= 0.5, TIE, +0.005 .. +0.007 topo F1 over the plain graph) the delivered graph now gains +0.057 / +0.062 topo F1, +0.11 / +0.13 lanes found and +0.05 / +0.07 junction F1; the maneuver coverage and the pair PR of the same runs are in `runs/e312_p_s<S>/maneuver_val.json` and `runs/e302/pair_pr_e312_p_s<S>_vs_e305_e_s<S>.json` once the chain has written them.
Maps after the joint training (`runs/e312_p_s<S>_maps`, `jmaps_eval.py score`): keypoint F1 0.696 / 0.687 against 0.723 / 0.705 of the E307 maps they started from (-0.027 / -0.018: the pair loss moves the heat a little, as in E305: -0.028 / -0.028), zone IoU 0.755 / 0.755 (0.767 / 0.761), core IoU 0.574 / 0.584 (0.584 / 0.581), best-threshold F1 0.715 / 0.713 (0.737 / 0.727). The maneuver coverage and the pair precision of the runs are in E310 section 6. Thresholds below 0.1 (snapped, E312 seed 0; `temp/gt2/snapopt/`): p >= 0.05 topo F1 +0.0548, lanes found +0.1295, F1@1.0 +0.0415 (p >= 0.1: +0.0566 / +0.1109 / +0.0385): the topo F1 is flat, the lanes found keep rising.
