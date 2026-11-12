# E319 - heading-guided forward / backward exchange along the lane in the junction maps head (`dirx`), then the E312 pair head on it

Card of 2026-11-12 22:45, written by the lane chat (fork B). Origin: Mirko's question of the evening, "why does the right shape not emerge by itself?", with his own hypotheses (his judgement, not a colleague's): smoothness is not a constraint, the position uncertainty is not large enough, and a forward / backward information exchange has to take place. This card asks the third one with the cheapest block that is fully differentiable; the other two are asked on the pair head by E317 / E318 (their own cards). Single factor: the exchange block, everything else is the E307 maps recipe and the E312 pair recipe.

## 0. The card (what the tools run)
```json card
{
 "id": "E319",
 "card": "",
 "title": "junction maps head with the heading-guided bidirectional exchange `DirX` (ctx dil+dirx, `jmaps_train.py --dirx`, heat sigma 1.0 m, 3000 steps), then the E312 pair head recipe on it (`jends_train.py --e2e --scales --maps-w 1.0 --steps 3000`), snapped read-out at p >= 0.1, against E312 (same recipe without the block)",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.005, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E312 of the same seed (maps runs/e307_n_s<S>, 3000 steps; pair head runs/e312_p_s<S>, 3000 steps): snapped p >= 0.1 topo F1 over the plain graph +0.0566 / +0.0615 (0.6825 / 0.6864)"},
  "d": {"recipe": "maps `jmaps_train.py --ctx dil --sigma 1.0 --steps 3000 --workers 3 --dirx` (run e319_m_s<S>), then `jends_train.py --e2e --scales --maps-w 1.0 --steps 3000` on `runs/e319_m_s<S>/model.pt`, BFLC_SIG=1.0 (run e319_p_s<S>); same read-out"}
 },
 "control": "c",
 "variant": "d",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "min_effect": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.01},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 5.0,
 "kill": "stage 1: the maps after 3000 steps with the block are worse than E307 of the same seed by more than 0.03 keypoint F1 on either seed (then the pair stage is not run), or the loss is not finite",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: replay"): `jmaps_eval.py score --run e319_m_s<S>` for the maps stage (keypoint F1, crowded entry recall, entry recall beyond 35 m against `e307_n_s<S>`), then `jends_eval.py score --snap --thr 0.1` for the read-out: differences to the plain graph of the seed, then to E312. KEEP when topo F1 over E312 rises by more than +0.01 on BOTH seeds with the guards held; TIE kept (the head without the block stays: it is cheaper); CUT below -0.01. Reported next to it: entry / exit recall, pair recall / precision, the maneuver coverage (`jmaneuver.py`, p0.1s), per-range coverage, and the cost: training seconds per step against E307 / E312 (log), milliseconds per frame of the maps head with and without the block on an IDLE GPU (not measured while the chain runs; CPU numbers below are indicative only).

## 1. Question and what exists
The junction maps head reads B's frozen pyramid at 2 m through four dilated residual blocks (dilation 1, 2, 4, 8: a context radius of about 38 m), then two convolutions at 1 m. That context is isotropic: a cell looks everywhere with the same learned kernel; nothing makes it look ALONG the lane, ahead and behind, where the evidence for an end lies (the continuation of the lane beyond the junction, the lane that ends in it). What the earlier cards measured and what it points at:
- E310 / E312: on the snapped read-out the pair decision is at the level of a perfect pair classifier (topo F1 +0.0304 against +0.0314 with the oracle pairs on the same keypoints); what is left is the ENDS: only 52 % of the GT pairs have both ends matched strictly, 69 % loosely.
- E307: entry recall beyond 35 m is 0.65 (0.55 before) and the far range stays the weakest region (E300: pair recall beyond 35 m 0.23 against 0.38 near); E285: 61 % of the entries carry no paint, so the head has to infer them from the context of the lane.
- E308: node tokens of B at the keypoints (more context in the pair head) improved the pair ranking (AUC +0.01) and nothing else: the pair head is not short of context; the maps (the keypoints) are where the context is missing.

## 2. Design (all differentiable)
`jmaps.DirX` (training/grid/jmaps.py), between the context blocks and the output convolutions of the 1 m stage (the features `f1` that the pair head reads are the exchanged ones):
- heading field t(x): the unit vector of the pyramid's own centre-lane heading (`P["centre1"]` channels 1, 2; detached, the exchange does not move the pyramid) in the (rows = x, columns = y) convention of the 1 m maps; the centre-lane probability (channel 0) enters as one extra input channel (where the field means something);
- per round: every cell reads the 64-channel features at x + s t(x) (AHEAD, forward) and x - s t(x) (BEHIND, backward) for s = 2, 4, 8 m (bilinear `grid_sample`: differentiable in the features AND in the sampling positions, i.e. in the heading) and a residual 1 x 1 block mixes them with its own features: f <- f + mix(cat[f, ahead_s, behind_s, p]) (conv 7 x 64 + 1 -> 32, GroupNorm, ReLU, conv 32 -> 64);
- two rounds: information reaches 16 m each way along the curve (the second round follows the curve the first round saw); between the rounds the heading is refined, t <- unit(t + turn(f)), a 1 x 1 correction of the features (the Kalman-smoother reading: a second pass with a better direction estimate);
- the last conv of each mix and the heading correction start at ZERO, so the model starts as the E307 model (bit-identical outputs, tested) and the block opens through training; the flag lives in the ctx string ("dil+dirx"), so `jends_train.py --maps`, `jmaps_eval.py` and `junction_view.py` build the same model without any edit.
Differentiability: no hard step is added; the named hard steps of the system are unchanged (the peak index choice and the soft-argmax read-out of the keypoints, the delivered thresholds, the snap of E311).

Cost (batch 1, 100 x 100, counted by `torch.utils.flop_counter` and measured by `grid/test_jmaps_dirx.py`): **33,218 parameters** (+7.5 % of the head's 444,490), **0.330 GMAC** in the 1 x 1 convolutions (+10 % of the head's 3.256 GMAC) plus 12 bilinear shifts of 64 channels (6 per round). CPU, 2 threads, batch 1: head 56 ms without, 107 ms with the block (the machine is loaded by the night chain and `grid_sample` is slow on CPU: indicative only; the GPU number is a to-do on an idle GPU). If the topo F1 ties, the head without the block stays.

Checks done before any run (CPU, 2026-11-12 22:3x): flag off is bit-identical to the code before E319 (weights, outputs, loss, gradients; also on 16 real VAL frames with the trained E307 weights); flag on at the start is bit-identical to flag off; after two steps no tensor of the head is dead; `shift_along` is exact on a linear field, rows = x, negative step = behind; the heading channels of the pyramid are usable as the guide: on 16 real VAL frames the centre-lane probability read 3 m along the heading is 0.79 (mean of ahead and behind) on ridge cells (p > 0.6), read 3 m across 0.40 (with the two channels swapped: 0.46 / 0.60); the CPU smoke of both trainers (`--dirx --smoke`, then the pair stage on the written maps) runs.

## 3. Prediction (a number) and what would count
Two seeds, VAL 873 frames, TEST untouched. **Prediction: topo F1 +0.005 over E312** (below the band: the expected outcome is a TIE); maps keypoint F1 +0.015 against E307 (the attention layer of E286 j2 gave +0.01 over the dilated context j1; E307's narrower heat gave +0.09, a pure context change is smaller); entry recall beyond 35 m +0.03. Reason for the modest number: the dilated blocks already see a radius of 38 m; the block changes the DIRECTION of the context, not its size. What would support Mirko's hypothesis even on a TIE of the topo F1: the far-range entry recall and the crowded-entry recall rise by 0.03 or more with the cost above, and the lanes found rise with them; a clean KEEP needs +0.01 topo F1 on both seeds.
Stage 1 costs about 0.9 h per seed (E307: 1.0 s per step on a shared GPU, 3000 steps), stage 2 about 1.2 - 1.6 h (E312: 70 minutes), the two seeds run in the two slots: about 5 GPU hours, about 2.5 h wall.

## 4. Files
Code: `training/grid/jmaps.py` (`DirX`, `shift_along`, ctx "+dirx"), `training/train/jmaps_train.py` (`--dirx`), test `training/grid/test_jmaps_dirx.py` (and the line in `tools/maintenance/run_all_tests.sh`). Start scripts (not started): `temp/e319/start_maps.sh <seed>`, `temp/e319/start_pairs.sh <seed>`, `temp/e319/kill_check.py <seed>`, `temp/e319/chain.sh` (both lanes, the scores of `temp/day11/day11.sh`, waits for an optional marker file); scratch checks `temp/e319/dev/`, `check_real.py`, `smoke_cpu.py`.

## 5. Result
Not run yet.
