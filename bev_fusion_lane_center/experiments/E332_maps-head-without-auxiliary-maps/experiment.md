# E332 - the junction maps head without its auxiliary maps (simplification arm S3)

Card of 2026-11-12 23:55, written by the lane chat (fork D) under D84 item 4 (simplify) and D85 (this arm runs AFTER E320). The junction maps head predicts, besides the entry / exit heat and headings that the keypoints need, a junction zone and a conflict core (1 m) and, on a 0.5 m branch, stop lines and pedestrian crossings. The 0.5 m branch is 1.477 of the head's 3.256 GMACs (45 %), 18.6 % of the whole system's 7.955, and its only reader is the evidence along the curve of the pair head (E331 asks whether that can go). The zone and core are read by the evidence too. Without the evidence the auxiliary maps are pure multi-task supervision; this card asks whether the ends need it.

## 0. The card (what the tools run)
```json card
{
 "id": "E332",
 "card": "",
 "title": "junction maps head without the 0.5 m paint branch (arm a: `jmaps_train.py --no-paint`) and without the zone / core supervision as well (arm b: `--no-paint --no-zone`), then the E312 pair stage without evidence on them, read out unsnapped and snapped at p >= 0.1, against E312 of the same seed",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": -0.004, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E312 of the same seed (full maps head 3.256 GMACs, evidence on): unsnapped p >= 0.1 topo F1 0.6438 / 0.6481, snapped 0.6825 / 0.6864; also reported against E331 (no evidence on the full maps) to separate the two removals"},
  "a": {"recipe": "maps: `jmaps_train.py --ctx dil --sigma 1.0 --steps 3000 --no-paint` (run e332_ma_s<S>, ctx `dil+nopaint`, 1.779 GMACs), then `jends_train.py --no-evidence --e2e --scales --maps-w 1.0 --steps 3000` on it (run e332_pa_s<S>), BFLC_SIG=1.0"},
  "b": {"recipe": "as a with `--no-paint --no-zone` (runs e332_mb_s<S>, e332_pb_s<S>; ctx `dil+nopaint+nozone`): the zone / core outputs stay in the 1 m layout but are no longer supervised; run only when arm a is ADOPTED"}
 },
 "control": "c",
 "variant": "a",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.01},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 10.2,
 "kill": "stage 1 only: the maps keypoint F1 (thr 0.3, 3 m, distance only) more than 0.05 below E307 of the same seed on either seed (then the pair stage is not run); or the loss is not finite",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: replay"): `jmaps_eval.py score --run e332_ma_s<S>` for stage 1 (keypoint F1, crowded and far-range recall against `e307_n_s<S>`; the zone / core / stop / crossing numbers read `n/a` for a head without them), `jends_eval.py dump` and the scores after stage 2, then `evaluation/jrule_removal.py --arm e332_pa_s --control e312_p_s` (arm b: `--arm e332_pb_s`). **Adopted** when the arm loses at most 0.01 topo F1 against E312 of the same seed on BOTH seeds and BOTH read-outs, **unsnapped (the differentiable read-out, D85: counts)** and snapped (the rule step, reported, must hold), with the guards held. Arm a carries the removal of the evidence (E331's question) too: if E331 is adopted the comparison of a against E331 isolates the paint branch; if E331 is NOT adopted the arm cannot be adopted either (it would confound). The chain `temp/e332/chain.sh` runs arm b only when a is adopted. Re-based on E320: `temp/e330/recipe.sh` (control `e320_a_s`, `RULE_READOUTS="attach"`).

## 1. Design and prediction
Code (committed with this card): `JMaps` ctx suffixes `+nopaint` (no `m05` / `head05`, no `j05` output, no paint loss) and `+nozone` (`out['skip'] = ('zone',)`: `jmaps_loss` leaves the zone / core term out; the 1 m output layout is unchanged, so no index moves), `jmaps_train.py --no-paint --no-zone`, `jmaps_eval.py` (n/a for the missing maps); the flags live in the ctx string, so `jends_train.py --maps`, `jends_eval.py` and the loaders of `jmaps_eval` / `junction_view` (`JModel(ctx=args['ctx'])`) build the right model without an edit of `jends.py`; `junction_view.py` still reads `j05` (a viewer: out of scope). Tests `training/grid/test_jmaps_simplify.py`: flags off = the golden numbers of the E307 head, `nopaint` = the head minus 37,058 parameters with bit-identical 1 m outputs for the same weights, the loss drops exactly the paint term (zone term), no dead tensor, the pair head (evidence off) trains the reduced head through the soft-argmax, evidence on fails loudly. CPU smoke of the real trainers: maps `--no-paint --no-zone`, the pair stage on them, the dump of 8 VAL frames: ok.
Compute saved (batch 1, `evaluation/jmodel_cost.py`): maps head 3.256 -> 1.779 GMACs (-45 %), the whole system 7.955 -> 6.477 (-18.6 %), 0.444 -> 0.407 M parameters; the paint sidecars of `data/prep/junction_paint.py` and the 0.5 m target raster are no longer needed by the recipe (arm b: the zone / core rasterisation either; the loader still computes them until `QData` is told to skip: a follow-up outside this card).
**Prediction (a number):** topo F1 -0.004 against E312 on both read-outs for arm a; the maps keypoint F1 -0.005 against E307 (the auxiliary tasks shape the features a little). The honest risk: multi-task supervision from the zone / core and the paint helped the maps in E286 (we never ablated it); a loss of more than 0.01 means the auxiliary maps are doing representation work and the saving is not free.
Cost: arm a: maps stage 3000 steps at about 1.0 s (the branch is gone: less) = 0.85 h, pair stage 1.15 - 1.6 h, post 0.25 h: about 2.55 h wall, 5.1 slot hours; arm b the same (conditional): budget 10.2 GPU h for both. Start (not started): `temp/e332/chain.sh [marker]` (both seeds, kill check `temp/e330/kill_check.py`, the rule gates arm b); single stages `temp/e332/start_maps.sh <seed> <a|b>`, `start_pairs.sh <seed> <a|b>`; `temp/e332/READY` carries the chain.

## 5. Result (arm a, 2026-11-14 06:15, VAL 873 frames, p >= 0.1, two seeds; chain `temp/e332/chain.sh` exit 0 at 05:45; numbers from `runs/e332_pa_s<S>/ends_card_thr0.1.json` (unsnapped) and `ends_card_snap_thr0.1.json` (snapped) against `runs/e312_p_s<S>/` (c) and `runs/e331_e_s<S>/` (E331), rule by `evaluation/jrule_removal.py`, `temp/e332/rule_a.log`)
**Token by the written rule (by hand: replay): NOT ADOPTED. The auxiliary maps stay; arm b (without the zone / core supervision) does not run.** The rule needs a loss of at most 0.01 topo F1 against E312 on BOTH seeds and BOTH read-outs: the unsnapped (differentiable) read-out loses **-0.0189 / -0.0106** topo F1 (F1@1.0 -0.0179 / -0.0096): both beyond 0.01 (seed 1's F1@1.0 just inside); the snapped read-out holds on both seeds. **Stage-1 kill did not fire:** the maps keypoint F1 of the head without the paint branch (`e332_ma_s<S>`) is **0.7409 / 0.7355 against E307 0.7233 / 0.7051** (+0.0175 / +0.0304, kill at -0.05; the kill check reads the keypoint F1 at the threshold 0.3 as the card's kill states; the `jmaps_val.json` best-threshold numbers are 0.7561 / 0.7501 against 0.7366 / 0.7268).

| unsnapped (differentiable), a (E312 c, E331) | seed 0 | seed 1 |
|---|---|---|
| topo F1 | 0.6249 (0.6438, E331 0.6206): **-0.0189** | 0.6376 (0.6481, E331 0.6357): **-0.0106** |
| F1@1.0 | 0.8485 (0.8664): **-0.0179** | 0.8589 (0.8684): -0.0096 |
| junction F1 / dup_frac / vertices | 0.2547 (+0.0036) / 0.1551 (-0.0057) / 19.80 (+0.06) | 0.2494 (+0.0020) / 0.1394 (-0.0012) / 20.08 (-0.19) |
| lanes found / curves | 0.7202 (0.7404) / 11505 (11378) | 0.7392 (0.7486) / 12388 (11849) |

| snapped (rule step), a (E312 c, E331) | seed 0 | seed 1 |
|---|---|---|
| topo F1 | 0.6780 (0.6825, E331 0.6796): -0.0045 | 0.6849 (0.6864, E331 0.6857): -0.0015 |
| F1@1.0 | 0.8634 (0.8669): -0.0034 | 0.8660 (0.8685): -0.0025 |
| junction F1 / dup_frac / vertices | 0.3199 (+0.0006) / 0.1733 (-0.0014) / 19.52 (-0.03) | 0.3213 (+0.0070) / 0.1549 (-0.0008) / 19.87 (-0.22) |
| lanes found | 0.7359 (0.7376) | 0.7458 (0.7423) |

**Reading.** (1) **The loss is the evidence, not the paint branch.** Arm a removes both (the paint branch's only reader is the evidence, so `--no-paint` needs `--no-evidence`): its unsnapped result is the result of E331 (evidence off, full maps: -0.0232 / -0.0124) within noise: a - E331 is **+0.0043 / +0.0019** topo F1 unsnapped and -0.0016 / -0.0008 snapped. Taking the 0.5 m paint branch out costs nothing measurable on top of the evidence removal that E331 already showed does not pay; the evidence removal is what the differentiable read-out pays for. (2) As in E331 and E320 the snapped read-out hides most of the loss (-0.0045 / -0.0015 against -0.0189 / -0.0106). (3) **Pair stage and coverage:** pair precision at the recall of E312 (0.298 / 0.328) is 0.750 / 0.761 against 0.743 / 0.724 (+0.007 / +0.037: the ranking is not worse, as in E331 the loss sits at the fixed operating point); maneuver coverage (tolerance 1.5 m, 40 degrees, `jmaneuver.py`) realised at p >= 0.1 unsnapped 7406 / 7748 of 10991 (0.674 / 0.705; E312 0.713 / 0.706, E331 0.687 / 0.695), snapped 7616 / 7849 (0.693 / 0.714; E312 0.707 / 0.701), junction precision 0.920 / 0.930 unsnapped (E312 0.944 / 0.944). (4) **Maps:** the maps head without the paint branch is not worse as a keypoint detector (kp F1 above E307 on both seeds at stage 1; after the joint pair stage 0.6951 / 0.7144 against E312's 0.7152 / 0.7126; zone / core F1 0.860 / 0.860 and 0.740 / 0.736 against E312's 0.861 / 0.860 and 0.730 / 0.738). **Price of the removal (reported):** the maps head falls from 3.256 to 1.779 GMACs (-1.477, 45 % of the head, 18.6 % of the system's 7.955 GMACs: the unit of the card, taken from it, not re-measured here; ms not measured); it costs 0.011 - 0.019 topo F1 on the differentiable read-out and 0.002 - 0.005 on the snapped one. **D87 / D88:** a removal arm, nothing here is an output selection; the E2E (unsnapped) read-out is the one that counts and it fails the bound.
