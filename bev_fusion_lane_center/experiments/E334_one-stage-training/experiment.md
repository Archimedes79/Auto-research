# E334 - one joint training stage instead of two (simplification arm S5)

Card of 2026-11-13 00:05, written by the lane chat (fork D) under D84 item 4 (simplify) and D85 (this arm runs AFTER E320). The E312 system is trained in two stages: the junction maps head alone (3000 steps, E307), then the pair head together with the maps head (`--e2e`, 3000 steps, E305 / E312). The second stage already trains the maps head with the pair loss and the maps loss (`--maps-w 1.0`). Is the first stage needed? One stage from the untrained head would remove a training run, a checkpoint hand-over and a pipeline step.

## 0. The card (what the tools run)
```json card
{
 "id": "E334",
 "card": "",
 "title": "one joint stage: `jends_train.py --e2e --scales --maps-w 1.0` from an UNTRAINED junction head (pyramid of the baseline, `temp/e334/make_init.py`), 3000 steps (arm o3) and, conditional, 6000 steps (arm o6), read out unsnapped and snapped at p >= 0.1, against E312 of the same seed (maps 3000 + joint 3000 steps)",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": -0.008, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E312 of the same seed (6000 maps steps in all, 3000 of them with the pair head): unsnapped p >= 0.1 topo F1 0.6438 / 0.6481, snapped 0.6825 / 0.6864"},
  "o3": {"recipe": "`jends_train.py --maps runs/e334_init_s<S>/model.pt --e2e --scales --maps-w 1.0 --steps 3000` (run e334_o_s<S>), BFLC_SIG=1.0: the maps head sees 3000 steps in all, half of E312's"},
  "o6": {"recipe": "the same with `--steps 6000` (run e334_o6_s<S>): the same number of maps steps as E312; run only when o3 loses between 0.01 and 0.03"}
 },
 "control": "c",
 "variant": "o3",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.01},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 9.3,
 "kill": "the loss is not finite, or the maps keypoint F1 of the joint maps (`jmaps_eval.py score --run <name>_maps`) is more than 0.10 below E307 of the same seed on either seed",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: replay"): the dump and the two scores of the pair run, `jmaps_eval.py score --run e334_o_s<S>_maps` for the joint maps, then `evaluation/jrule_removal.py --arm e334_o_s --control e312_p_s` (o6: `--arm e334_o6_s`). **Adopted** when the arm loses at most 0.01 topo F1 against E312 of the same seed on BOTH seeds and BOTH read-outs, **unsnapped (the differentiable read-out, D85: counts)** and snapped (the rule step, reported, must hold), with the guards held. If o3 passes, the pipeline has one training stage; if only o6 passes, the second maps stage is not needed but its steps are (a joint run of 6000 steps is still one run); if neither passes, the separate maps stage stays. Re-based on E320: `temp/e330/recipe.sh`.

## 1. Design and prediction
No change of `jends_train.py`: `--maps` takes a checkpoint of a maps run; `temp/e334/make_init.py <seed>` writes one with the pyramid of the baseline (`runs/e273_p_s<S>`) and the junction head as `jmaps_train.py` initialises it (the same seeded start), with `args` (ctx dil, d 64, sigma 1.0, base, seed), so every loader (`jends_eval.py`, `jmaps_eval.py`) reads the joint result as before. The pair head starts from nothing too: in training the GT end cells are injected as keypoint candidates (`JEnds.forward`, `gt` given), so the pair loss has ends to work on from step 0 while the heat is still blank; the maps loss (`--maps-w 1.0`: heat BCE + dice with the narrow sigma, headings, zone, core, paint) teaches the heat in the same steps. CPU smoke of the real trainer from the init checkpoint: ok.
**Prediction (a number):** o3 loses 0.008 topo F1 on both read-outs (half the maps training: E315 asks the other direction, twice the maps training, and its answer sets the expectation; the maps stage of E307 reached keypoint F1 0.72 / 0.71 in 3000 steps from the narrow heat target); o6 loses 0.002. Honest risk: the pair head trains on bad keypoints for the first thousand steps and the heat head is pulled by two losses at once; a collapse would show as a keypoint F1 far below E307 (the kill).
Cost: pair stage only, 3000 steps at 1.4 s (E312 recipe, joint, scales) = 1.15 - 1.6 h per seed with both seeds in the two slots, post 0.25 h: about 1.65 h wall for o3, 3.3 slot hours; o6 2.3 - 3.2 h per seed, 3.0 h wall incl. post, 6.0 slot hours (conditional). Total budget 9.3 GPU h. Start (not started): `temp/e334/start_pairs.sh <seed> [o3|o6]` (runs `E334.o_s<S>` / `E334.o6_s<S>`; it builds the init checkpoint if missing); `temp/e334/READY` carries the prefix `e334_o_s` for the generic pair-arm chain.
