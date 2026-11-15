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

## 2. Result (2026-11-15, by hand: replay of the stored cards; runs E334.o_s0 22:01 - 23:43 and E334.o_s1 23:20 - 01:04, VAL 873 frames, 22 scenes; the arm o6 is not run: it was conditional on o3 losing between 0.01 and 0.03 topo F1)
**Token by the rule written above: NOT ADOPTED, by ONE guard missed by 0.0004** (`jrule_removal.py --arm e334_o_s --control e312_p_s`, exit 1): the duplicate guard of the unsnapped read-out on seed 0, dup_frac +0.0104 against the bound +0.01. **Topo F1 is inside the 0.01 band on both seeds and both read-outs** (unsnapped -0.0011 / -0.0034, snapped -0.0024 / -0.0049; the prediction was -0.008), and every other guard holds. The token stays as written (the rule was fixed before the run; no rewriting, no re-run on the strength of the near miss). The numbers are from `runs/e334_o_s<S>/ends_card*_thr0.1.json` and `runs/e312_p_s<S>/...`; the intervals from `jdiag.py boot` run by hand on 2026-11-15 (paired scene bootstrap, 22 scenes, 1000 resamples; `temp/e334/boot_vs_e312_*.log`). **D87 does not apply:** E334 removes a TRAINING stage (the separate maps stage), it does not replace a rule step of the delivered graph by an end-to-end piece; so "TIE, kept and refined" is not the reading, the arm is simply not adopted. (It is the more end-to-end recipe: one joint stage in which the pair loss reaches the maps head from step 0.)

| VAL, control E312 of the same seed | seed 0 | seed 1 |
|---|---|---|
| unsnapped topo F1: E334 / E312 / difference | 0.6427 / 0.6438 / **-0.0011** | 0.6447 / 0.6481 / **-0.0034** |
| unsnapped difference, paired 95 % interval | [-0.0079, +0.0055] | [-0.0081, +0.0017] |
| snapped topo F1: E334 / E312 / difference | 0.6801 / 0.6825 / **-0.0024** | 0.6815 / 0.6864 / **-0.0049** |
| snapped difference, paired 95 % interval | [-0.0071, +0.0027] | [-0.0098, -0.0001] |
| dup_frac, unsnapped / snapped (bound +0.01) | **+0.0104 FAIL** / +0.0090 | +0.0086 / +0.0071 |
| dup_frac difference, paired interval (unsnapped / snapped) | [-0.0014, +0.0236] / [-0.0010, +0.0196] | [-0.0000, +0.0163] / [-0.0003, +0.0142] |
| F1@1.0 (unsnapped / snapped, bound -0.01) | -0.0011 / -0.0006 | -0.0033 / -0.0020 |
| junction F1 (unsnapped / snapped, bound -0.03) | -0.0090 / -0.0075 | -0.0064 / -0.0114 |
| junction F1 interval (unsnapped / snapped) | [-0.0127, -0.0057] / [-0.0139, -0.0014] | [-0.0137, -0.0001] / [-0.0195, -0.0028] |
| vertices per 100 m (unsnapped / snapped, bound +1.0) | -0.02 / -0.04 | +0.05 / +0.01 |
| lanes found (unsnapped / snapped) | -0.0072 / -0.0034 | -0.0159 / -0.0095 |
| pair precision at E312's recall (not in the rule) | 0.686 against 0.743 (-0.057) | 0.646 against 0.724 (-0.077) |
| maneuvers realised p >= 0.1 / snapped, of 10 991 (not in the rule) | 7633 / 7594 (E312 7833 / 7770) | 7486 / 7498 (E312 7760 / 7703) |
| maps keypoint F1 against E307 (kill below -0.10) | 0.6608 against 0.7233 = -0.0625 | 0.6481 against 0.7051 = -0.0570 |

**Reading, plainly.** (1) **The near miss is a near miss:** the failing guard, dup_frac +0.0104 on the unsnapped read-out of seed 0, is 0.0004 over the bound, its paired interval [-0.0014, +0.0236] contains zero and the bound, and the three other dup_frac values stay at +0.0071 to +0.0090. The topo F1 of one joint stage is within the noise of the two-stage recipe on both seeds (the intervals contain zero except the snapped seed 1, [-0.0098, -0.0001]). (2) **But the pair head is worse in a way the rule does not see:** pair precision at E312's recall is lower by 0.057 / 0.077, the maneuver coverage by 200 / 274 maneuvers of 10 991, lanes found by 0.007 - 0.016, junction F1 by 0.006 - 0.011 (the intervals exclude zero), and the joint maps head reaches a keypoint F1 0.057 - 0.063 below E307 (not a kill, which was -0.10). The ends are worse, and the graph hides it behind the snap and the close-node step. (3) **The prediction** (-0.008 topo F1 on both read-outs) was too pessimistic on topo F1 (observed -0.001 to -0.005); it said nothing about the pair precision, which fell. (4) Two seeds give no noise band: topo F1, F1@1.0, junction F1, dup_frac and lanes found agree in sign between the seeds, vertices per 100 m do not (-0.02 against +0.05, inside the bound); the pair-stage seed noise was not measured in this card.
**What the simplification would save if a re-run ever decided it.** The first stage (E307, the maps head alone) cost 3 052 / 3 073 s per seed on the then quiet box (0.85 h) in the ledger, the pair stage of E312 4 168 / 4 192 s; one joint stage is one recipe, one run, no checkpoint hand-over (`temp/e334/make_init.py` would go). The E334 runs themselves took 6 082 / 6 213 s (1.7 h) on the loaded box of the night (two trainings, loader-bound, `experiments/E342_pair-head-host-time/experiment.md` section 4): that is not a measurement of the saving, which is at most the 0.85 h of the first stage per seed. **No re-run is started on the strength of the near miss:** the rule was written before the run, a re-run would be a new card with a new bound and a critic pass, only if the saving (about 0.85 h of GPU per seed and one pipeline step) is worth the loss of pair precision it came with; the data say it is not free.
