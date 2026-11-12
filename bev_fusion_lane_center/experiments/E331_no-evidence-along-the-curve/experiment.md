# E331 - the E312 pair stage without the evidence along the curve (simplification arm S1)

Card of 2026-11-12 23:50, written by the lane chat (fork D) under D84 item 4 (simplify) and D85 (this arm runs AFTER E320). The pair head of E287 reads, for every candidate pair, six numbers sampled along its Hermite curve (mean and minimum of B's lane-centre probability, mean zone, mean core, maximum stop-line and crossing probability): the `evidence`. E287 found that it adds nothing to the order features (p2 over p1: TIE) and E312 never ablated it. Without it the head needs no zone / core / paint maps as inputs, which opens E332 (the maps head without its auxiliary maps).

## 0. The card (what the tools run)
```json card
{
 "id": "E331",
 "card": "",
 "title": "E312 pair stage (`jends_train.py --e2e --scales --maps-w 1.0 --steps 3000`) with `--no-evidence` on the E307 maps (BFLC_SIG=1.0), read out unsnapped and snapped at p >= 0.1, against E312 of the same seed",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": -0.003, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E312 of the same seed (maps e307_n_s<S>, evidence on): unsnapped p >= 0.1 topo F1 0.6438 / 0.6481 (`runs/e312_p_s<S>/ends_card_thr0.1.json`), snapped 0.6825 / 0.6864 (`ends_card_snap_thr0.1.json`)"},
  "e": {"recipe": "`jends_train.py --no-evidence --e2e --scales --maps-w 1.0 --steps 3000` on `runs/e307_n_s<S>/model.pt`, BFLC_SIG=1.0 (run e331_e_s<S>); both read-outs"}
 },
 "control": "c",
 "variant": "e",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.01},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 3.3,
 "kill": "the loss is not finite, or the maps after the joint training are worse than E307 by more than 0.05 keypoint F1 on either seed",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: replay"): `evaluation/jrule_removal.py --arm e331_e_s --control e312_p_s` after `jends_eval.py dump --run e331_e_s<S>` and the two scores (`score --thr 0.1` = unsnapped, `score --snap --thr 0.1` = snapped; the chain does it). **The removal is ADOPTED** when the arm loses at most 0.01 topo F1 against E312 of the same seed on BOTH seeds and on BOTH read-outs, **unsnapped (the differentiable read-out, D85: it counts)** and snapped (the rule step of E311, reported, must hold), with the guards held; a gain inside the band is a TIE and the cheaper variant (this one) stays. NOT adopted when one seed or one read-out loses more than 0.01. If E320 is KEEP the control and the read-out are re-based (`temp/e330/recipe.sh`: `CONTROL_PREFIX=e320_a_s`, `RULE_READOUTS="attach"`).

## 1. Design and prediction
One change: the six evidence features are not computed and not given to the pair MLP (`nin` 2 d + 13 + 5 (+ 6 evidence)). The maps head still trains with all its losses (the evidence-free pair head does not read the zone / core / paint maps, they are then pure auxiliary tasks: E332 asks whether they can go). Two seeds, VAL 873 frames, TEST untouched, the flag `--no-evidence` exists since E287 (no code change: the run is the E312 recipe plus one flag). **Prediction (a number):** topo F1 -0.003 against E312 on both read-outs (a TIE); the step time falls by 3 - 6 % (three `grid_sample`s of 12 points for every pair of the K x K grid fewer, from 1.38 s). If the arm loses more than 0.01, the evidence carries information the order features do not, and E332 cannot run (its arms need the evidence off).
Cost: pair stage only, 3000 steps at 1.3 - 1.4 s = 1.1 - 1.2 h per seed with both seeds in the two slots, dump and scores 0.25 h: 1.65 h wall, 3.3 slot hours; no maps stage. Start (not started): `temp/e331/start_pairs.sh <seed>` = run `E331.e_s<S>`; the run name prefix `e331_e_s` is the first line of `temp/e331/READY`.
Inference cost saved: about 0.003 GMACs and three bilinear samplings per pair grid (the read-out of a deployed head would not need the three maps either): the saving that counts is in E332.
