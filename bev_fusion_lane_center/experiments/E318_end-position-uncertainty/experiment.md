# E318 - a position uncertainty per keypoint (heteroscedastic end position), on top of the E317 geometry loss

Card of 2026-11-12 22:50, written by the lane chat. Second of Mirko's hypotheses of the evening: the position uncertainty the head is allowed to express is not large enough. Today every keypoint is a point with the same implicit sigma: the heat target has a fixed Gaussian (1.0 m since E307) and the pair head sees only the position, so a label offset of 1 - 2 m at a junction end (the largest single loss of the E283 junction analysis) is punished like a real error and nothing tells the pair head or the planner which end is soft. A Laplace NLL with a learned scale per keypoint (heteroscedastic regression, Kendall and Gal; the "uncertainty" head of the end heads of MapTR-like and CenterNet-like detectors) lets the net declare an end uncertain at the price of log sigma, and gives the planner an uncertainty per end for free.

## 0. The card (what the tools run)
```json card
{
 "id": "E318",
 "card": "",
 "title": "E317 + a position scale per keypoint (`jends_train.py --geom-w 0.5 --unc`: log sigma from the keypoint token, the end term becomes the Laplace NLL |d| exp(-s) + s, s is a detached feature of the pair logit and the output `kp_logsig`), snapped read-out at p >= 0.1, against the E317 arm of the same seed",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.005, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E317 arm g of the same seed (`runs/e317_g_s<S>`: `--geom-w 0.5`, no scale); the second reference is E312 of the same seed (snapped p >= 0.1 topo F1 0.6825 / 0.6864). If E317 is not run first the control is a run of the same recipe started in the same window (`--geom-w 0.5`, run id E318.c_s<S>, `temp/e318/start_pairs.sh <seed> c`)"},
  "u": {"recipe": "`jends_train.py --e2e --scales --maps-w 1.0 --steps 3000 --geom-w 0.5 --unc` on `runs/e307_n_s<S>/model.pt`, BFLC_SIG=1.0 (run e318_u_s<S>); same read-outs"}
 },
 "control": "c",
 "variant": "u",
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
 "kill": "the loss is not finite, log sigma sits at the clamp (-3 or +3) for more than 90 % of the keypoints after 1000 steps (the scale head collapsed), or the maps after the joint training are worse than E307 by more than 0.05 keypoint F1 on either seed",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: replay"): `jends_eval.py score --snap --thr 0.1` and `score --thr 0.1`, differences to the plain graph, then to the control and to E312. KEEP when snapped topo F1 over the control rises by more than +0.01 on BOTH seeds with the guards held; TIE kept when the numbers stay inside the band AND the uncertainty is informative (below); CUT below -0.01. **The uncertainty has to be worth something even when topo F1 does not move** (written before the run): on the VAL dump (`logsig` per keypoint, `ends_val.pkl`), for the keypoints matched to a GT end, the rank correlation (Spearman) between sigma = exp(log sigma) and the true end error in metres is at least 0.3 on both seeds, and the delivered pairs with the 20 % smallest sum of the two sigmas have a pair precision higher by 0.03 than the delivered pairs with the 20 % largest; then the scale is a usable planner output (an uncertainty per lane end) and the arm is kept as the head that delivers it, TIE otherwise CUT. Reported next to it: the same numbers as E317 (unsnapped read-out, gap closed, end error, kinks), the mean sigma by range (near / beyond 35 m) and by painted / unpainted ends (E285 flags), maps keypoint F1 after the joint training.

## 1. Design and prediction
Single factor against E317 (the control): the position scale. The head `JEnds(unc=True)` adds `logsig = Linear(d, 1)` on the self-attended keypoint token (zero-initialised, so sigma = 1 m at the start; clamped to [-3, 3], i.e. 0.05 - 20 m) and two inputs to the pair logit (the detached log sigma of the entry and of the exit keypoint, so the pair loss cannot shape the scale; checked in the self-test). The end term of the geometry loss of E317 (every valid keypoint within 4 m / 30 degrees of a GT end, nearest end) is the Laplace NLL |d| exp(-s) + s instead of the smooth L1; the shape term is unchanged. Self-tests (`python jends.py`): a 3 m error pushes log sigma up and a 0.1 m error pushes it down, a keypoint that is no end gets no gradient, the pair loss does not reach `logsig`, the flag-off model is bit-identical to the E312 code. Two seeds, VAL 873 frames, TEST untouched.

Honest risk. A Laplace NLL on the position error of a keypoint whose position is a soft-argmax inside a 3 x 3 window can only reduce its error by shifting heat; the scale itself does not move the peak. So the expected benefit is not a better topo F1 (**prediction a number: +0.005 over E317, a TIE**) but the information: the Spearman correlation of sigma with the true end error of at least 0.3 (a head that has learned nothing gives 0), and a pair precision gain of 0.03 between the confident and the unsure fifth. If sigma ends in the clamp or at one value for all keypoints the kill applies.

## 2. Cost
- Inference: +257 parameters (the scale layer 65, the two extra inputs of the first layer of the pair head 128 and of the scale head 64) and +0.2 MMACs on a head of 90 467 parameters and 21.65 MMACs (`temp/e317/t_real.py flops`, batch 1, CPU flop counter); no new kernel at read-out: well below 0.1 ms.
- Training: as E317 (1.6 h per seed, two seeds in the two slots, about 15 minutes of dump and scores). Run order: after E317 in the same window (the E317 arm is the control), or both in one window if the two GPU slots are free for 3.3 h (4 runs: E317.g_s0, E317.g_s1, E318.u_s0, E318.u_s1, two at a time).
- Files changed for it: the same as E317 plus `JEnds(unc)` in `training/grid/jends.py` and `--unc` in `training/train/jends_train.py`; `evaluation/jends_eval.py` loads the run with its scales and writes `logsig` into the dump.
- Start (not started): `temp/e318/start_pairs.sh <seed> [c]`.
