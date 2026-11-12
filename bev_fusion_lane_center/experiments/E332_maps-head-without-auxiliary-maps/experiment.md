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
