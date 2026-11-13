# E315 - the junction maps trained twice as long (6000 steps), then the E312 pair head on them

Card of 2026-11-12 21:30, written by the lane chat (Mirko: "gerne weitere Experimente machen"). Origin: the loss curves of the E307 maps and the E312 pair head are still falling at step 3000 (maps: total 1.25 -> 1.17 over the last 500 steps, pair head 1.69 -> 1.59; `temp/e309/curve.py` on `runs/e307_n_s0`, `runs/e312_p_s0`), so the system may be under-trained, not limited by its design. The check of 2026-11-12 evening on the rendered frames (`results/l6_view_E312/`, `temp/e309/where.py`) says the delivered graph is right where it is (0.3 - 3 % of the delivered length is more than 2 m from the GT) and misses at the 1 m - 2 m accuracy of the points (GT covered within 1 m 0.83 - 0.94, within 2 m 0.94 - 0.98): the kind of error that more training can remove.

## 0. The card (what the tools run)
```json card
{
 "id": "E315",
 "card": "",
 "title": "junction maps head trained 6000 steps (heat sigma 1.0 m, dilated context; `jmaps_train.py --steps 6000`), then the E312 pair head recipe on them (`jends_train.py --e2e --scales --maps-w 1.0 --steps 3000`), snapped read-out at p >= 0.1, against E312 (maps 3000 steps, same pair head)",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.01, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E312 of the same seed (maps e307_n_s<S>, 3000 steps; pair head 3000 steps): snapped p >= 0.1 topo F1 over the plain graph +0.0566 / +0.0615 (0.6825 / 0.6864)"},
  "l": {"recipe": "maps `jmaps_train.py --ctx dil --sigma 1.0 --steps 6000` (run e315_m_s<S>), then `jends_train.py --e2e --scales --maps-w 1.0 --steps 3000` on `runs/e315_m_s<S>/model.pt`, BFLC_SIG=1.0 (run e315_p_s<S>); same read-out"}
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
 "kill": "the maps after 6000 steps are worse than E307 (3000 steps) by more than 0.03 keypoint F1 on either seed (then the pair stage is not run)",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: replay"): `jmaps_eval.py score --run e315_m_s<S>` for the maps stage (keypoint F1, crowded entry recall, far-range recall against `e307_n_s<S>`), then `jends_eval.py score --snap --thr 0.1` for the read-out: differences to the plain graph of the seed, then to E312. KEEP when topo F1 over E312 rises by more than +0.01 on BOTH seeds with the guards held; TIE kept (the shorter recipe stays the cheaper one); CUT below -0.01. Reported next to it: entry / exit recall, pair recall / precision, the maneuver coverage (`jmaneuver.py`, p0.1s), per-range coverage (`temp/e309/where.py`, scratch).

## 1. Design and prediction
Two seeds, the same data and loss as E307 / E312; only the steps of the maps stage change (3000 -> 6000; the learning-rate schedule is the trainer's, stretched to the run length). The pair stage is the unchanged E312 recipe, so the joint training adds 3000 more maps steps in both arms (c: 6000 in all, l: 9000 in all). VAL 873 frames, TEST untouched. **Prediction (a number):** topo F1 +0.010 over E312 (a TIE at the edge of the band); keypoint F1 of the maps +0.02; if the maps gain but the delivered graph does not, the limit is the pair stage, not the maps (E316 asks the pair stage).
Cost: maps 6000 steps at about 0.9 s = 1.5 h, pair stage 3000 steps at about 1.9 s = 1.6 h, two seeds in the two GPU slots, then dump and scores (about 15 minutes).

## 2. Result (VAL 873 frames, two seeds; read from the result files on 2026-11-13 02:05 by the lane lead)
Runs: maps `E315.m_s<S>` (6000 steps, 22:00 - 23:56, 0.95 s/step on the loaded CPU), pair stage `E315.p_s<S>` (`--e2e --scales --maps-w 1.0`, 3000 steps, 2.2 - 2.3 s/step under load, 00:04 - 01:32); post scoring by `temp/day11` (snapped, plain, pair PR, maneuver) and the not snapped p >= 0.1 read-out scored afterwards (`temp/e315/unsnap_p01.sh`). Control = E312 of the same seed (E307 maps 3000 steps, same pair stage).

| VAL | E312 s0 | **E315 s0** | E312 s1 | **E315 s1** |
|---|---|---|---|---|
| **snapped p >= 0.1: topo F1** | 0.6825 | **0.6789** | 0.6864 | **0.6825** |
| over the plain graph | +0.0566 | +0.0530 | +0.0615 | +0.0576 |
| **E315 - E312 (the rule)** | | **-0.0036** | | **-0.0039** |
| lanes found | 0.7376 | 0.7450 | 0.7423 | 0.7454 |
| junction F1 (guard -0.03) | 0.3194 | 0.3230 | 0.3143 | 0.3209 |
| junction recall | 0.680 | 0.697 | 0.674 | 0.696 |
| dup_frac (guard +0.01) | 0.1747 | 0.1807 (+0.0060) | 0.1557 | 0.1629 (+0.0072) |
| F1@1.0 (guard -0.01) | 0.8669 | 0.8669 | 0.8685 | 0.8689 |
| vertices / 100 m (guard) | 19.55 | 19.39 | 20.10 | 19.81 |
| not snapped p >= 0.1: topo F1 | 0.6438 | 0.6398 (-0.0040) | 0.6481 | 0.6444 (-0.0037) |
| not snapped, thr 0.5 (card of record): topo F1 | 0.6392 | 0.6407 | 0.6429 | 0.6439 |
| maneuvers realised, p0.1 snapped | 0.7069 | 0.7109 | 0.7008 | 0.7093 |
| entry / exit recall of the keypoints (pair run) | 0.745 / 0.771 | 0.753 / 0.781 | 0.752 / 0.767 | 0.761 / 0.784 |
| pair recall (logit >= 0) / precision | 0.298 / 0.743 | 0.376 / 0.741 | 0.328 / 0.724 | 0.379 / 0.740 |
| precision at E312's recall (`jpair_pr.py`) | | 0.764 (+0.021) | | 0.768 (+0.044) |
| maps stage: keypoint F1 (3 m, thr 0.3) | E307 0.7233 | **0.7481** (+0.025) | E307 0.7051 | **0.7488** (+0.044) |
| maps stage: entry / exit recall | 0.750 / 0.754 | 0.767 / 0.778 | 0.746 / 0.749 | 0.775 / 0.786 |

**Verdict (by hand: replay, the rule of section 0): TIE.** KEEP needed more than +0.01 topo F1 over E312 on both seeds; the result is -0.0036 / -0.0039 (CUT would be below -0.01), all guards held (dup_frac +0.006 / +0.007 stays below 0.01). The shorter recipe (E312: maps 3000 steps) stays, because on a tie the cheaper variant wins (the E315 recipe costs 1.5 h more per seed of training for no gain in the delivered graph). The not snapped read-out says the same (-0.0040 / -0.0037).

**What it shows.** The prediction of the card was right for the maps (keypoint F1 +0.02 predicted, +0.025 / +0.044 measured) and for the pair precision at equal recall (+0.02 predicted, +0.021 / +0.044), and wrong for the delivered graph (+0.010 predicted, -0.004 measured). Twice the maps training lifts the end detection clearly (recall of the keypoints, pair recall +0.08 / +0.05, maneuvers +0.004 / +0.009) but not the topology of the delivered graph: with the snap the cap is the attachment of the connectors to the delivered pieces (69 % of the snap's gain is wiring, E320), not the quality of the ends. Compute: no change at run time (same model); training +1.5 h per seed. E316 (the pair stage trained 6000 steps on the E307 maps) is the other half of the question and runs next.
