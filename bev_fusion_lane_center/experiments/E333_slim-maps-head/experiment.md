# E333 - a slimmer junction maps head, width 40 instead of 64 (simplification arm S4)

Card of 2026-11-13 00:00, written by the lane chat (fork D) under D84 item 4 (simplify) and D85 (this arm runs AFTER E320). The junction maps head carries 64 channels through four dilated residual blocks and the 1 m branch; with the context at 2 m and an output of 8 + 2 maps that may be more than it needs. Width 40 (a multiple of 8 for the group norms) takes 40 % of the head's MACs away (the 0.5 m branch shrinks little: its cost is the 0.5 m resolution).

## 0. The card (what the tools run)
```json card
{
 "id": "E333",
 "card": "",
 "title": "junction maps head of width 40 (`jmaps_train.py --d 40`, 1.967 GMACs instead of 3.256), then the E312 pair stage on it, read out unsnapped and snapped at p >= 0.1, against E312 of the same seed",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": -0.004, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E312 of the same seed (width 64, 3.256 GMACs): unsnapped p >= 0.1 topo F1 0.6438 / 0.6481, snapped 0.6825 / 0.6864"},
  "w": {"recipe": "maps: `jmaps_train.py --ctx dil --sigma 1.0 --steps 3000 --d 40` (run e333_m_s<S>), then `jends_train.py --e2e --scales --maps-w 1.0 --steps 3000` on it (run e333_p_s<S>, evidence on as E312), BFLC_SIG=1.0"}
 },
 "control": "c",
 "variant": "w",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.01},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 4.8,
 "kill": "stage 1 only: the maps keypoint F1 more than 0.05 below E307 of the same seed on either seed (then the pair stage is not run); or the loss is not finite",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: replay"): `jmaps_eval.py score --run e333_m_s<S>` (stage 1), the dump and the two scores after stage 2, then `evaluation/jrule_removal.py --arm e333_p_s --control e312_p_s`. **Adopted** when the arm loses at most 0.01 topo F1 against E312 of the same seed on BOTH seeds and BOTH read-outs, **unsnapped (the differentiable read-out, D85: counts)** and snapped (the rule step, reported, must hold), with the guards held; a TIE inside the band keeps the cheaper variant. If it passes, width 32 (1.629 GMACs, -50 %) is the follow-up card; if it fails, the maps head is as wide as it must be. Re-based on E320: `temp/e330/recipe.sh`.

## 1. Design and prediction
No code change: `--d` exists (the width of the head, stored in the maps args, read by `JEndModel` as `maps_args['d']` for the head and for the pair head's input). One change against E312: the width, with the same maps recipe (3000 steps, sigma 1.0 m) and the same joint pair stage. Compute (batch 1): maps head 3.256 -> 1.967 GMACs (-40 %), 0.444 -> 0.198 M parameters; the whole system 7.955 -> 6.665 (-16 %); width 48 is -28 % (2.351), width 32 -50 % (1.629), width 40 + no paint branch (E332 a) 0.766 GMACs (-76 %). CPU smoke of the real trainers with `--d 40` and the pair stage on it: ok.
**Prediction (a number):** topo F1 -0.004 against E312 on both read-outs, maps keypoint F1 -0.01 against E307. Honest risk: the context blocks are what lifted the keypoint F1 from 0.58 (no context, E286 j0) to 0.63 (E286 j1) and the narrow heat (E307) found its gain in separating crowded ends; a narrower head may lose that separation.
Cost: maps stage (a little faster than E307's 1.0 s per step) 0.85 h, pair stage 1.15 - 1.6 h, post 0.25 h: about 2.4 h wall, 4.8 slot hours. Start (not started): `temp/e333/chain.sh [marker]`; single stages `temp/e333/start_maps.sh <seed>`, `start_pairs.sh <seed>`; `temp/e333/READY` carries the chain.
