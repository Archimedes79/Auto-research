# E299 - time for the junction head: ONE ego-warped state of the previous keyframe through a zero-start gate (recursive state first, cross-attention as the comparison), on the E312 system (D82, D87 improvement item)

Card of 2026-11-11 (plan card, brief: "Nutze Cross Attention ueber die Zeit"), updated 2026-11-13 21:30 by the lane chat to the E312 system on the lane lead's order (D87 item 4: the cheapest variant first). **Status: card only, no code, no GPU.** Literature (`literature/scan_2026-11-11_pruning_uncertainty_time.md`, idea 1): CenterTrack feeds the previous detections as one extra heat-map channel (prior-off ablation -1.8 AMOTA on nuScenes), RVLD warps the previous output (flicker 0.053 -> 0.014); both at 10 - 30 Hz, the nuScenes keyframes are 0.5 s apart. E290 measured on the E284 head: 20 % of the true pairs flip between two keyframes; averaging with the warped previous logits cut the flips by 44 % but delivered fewer true pairs (36 % -> 31 %), and 52 - 54 % of the true pairs are delivered in neither frame: recall is the first problem. The ends of E312 are delivered with pair recall 0.30 / 0.33 (logit >= 0) and snapped topo F1 0.6825 / 0.6864; a witness from the previous keyframe is the one cheap source of information that the single-frame head does not have. This is an improvement card, not an end-to-end replacement of a rule step: D87's 'kept and refined' does not apply, a CUT is a CUT.

## 0. The card (what the tools run)
```json card
{
 "id": "E299",
 "card": "",
 "title": "the E312 pair head gets ONE ego-warped state of the previous keyframe (its own keypoint tokens and heat, a recursive state: nothing is computed twice at inference) through a zero-initialised gate, trained with the CenterTrack noise on the previous state (jitter, 10 % false, 40 % missing), against E312 of the same seed; the cross-attention variant (the previous keyframe re-run through the maps head) as the comparison arm",
 "kind": "block",
 "deterministic": false,
 "metric": "pair_recall",
 "direction": "higher",
 "band": 0.03,
 "prediction": {"value": 0.03, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E312 of the same seed (E307 maps + `--e2e --scales --maps-w 1.0`, 3000 steps): pair recall (logit >= 0) 0.298 / 0.328, precision 0.743 / 0.724; snapped topo F1 0.6825 / 0.6864"},
  "r": {"recipe": "RECURSIVE STATE (cheapest, first): `jends.py` + a gated cross-attention of the current keypoint tokens to the ego-warped keypoint tokens and heat that the SAME network produced for the previous keyframe (at inference the previous output, kept; in training the previous keyframe's tokens from the frozen E312 forward on its cached features, no backpropagation through time); the gate starts at zero (flag off and gate 0 = E312 bit for bit); the first frame of a scene has no state: the layer is skipped; run ids E299.r_s<S>", "run_ids": ["E299.r_s0", "E299.r_s1"]},
  "x": {"recipe": "CROSS-ATTENTION (the original E299): the same layer, but the previous keyframe's tokens come from the maps head run again on the previous cached features (a second forward of the maps head in training AND at inference: +3.26 GMACs, the comparison for what the recursion costs in accuracy); run ids E299.x_s<S>, run only after r", "run_ids": ["E299.x_s0", "E299.x_s1"]}
 },
 "control": "c",
 "variant": "r",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "pair_recall", "vs": "c", "min_effect": 0.03},
  {"metric": "pair_precision", "vs": "c", "max_loss": 0.02},
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.01},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 4.0,
 "kill": "step 0 (CPU, below) does not show a flip reduction of at least 25 % relative with topo F1 held nor a recall gain: then the previous frame is a weak witness at 2 Hz and the card is not run; or the loss is not finite; or the gate stays at zero after training (the layer does not learn: the gradient report shows no gradient into the gate)",
 "evaluate": {"split": "val"}
}
```
**Verdict (by hand: pair scorer + pair flip rate, the training runs through run.py):** KEEP when pair recall rises by more than 0.03 over c on BOTH seeds with the guards held (pair precision, snapped topo F1, dup_frac, junction F1); TIE (inside the band) is kept and refined (the refinement is a longer memory: two buffered frames, or the world memory of the BEV system's idea); CUT below c - 0.03 or a guard broken on both seeds. Reported: the flip rate (E290 definition), the arm x against r (what the recursion costs), ms and MACs, the pair recall of the first frames of scenes (no state) against the rest.

## 1. Card text
- **Statement (falsifiable):** a single ego-warped state of the previous keyframe, fed through a zero-start gate, raises the pair recall of the E312 head by more than 0.03 without losing precision, because 20 % of the true pairs flip between keyframes (E290) and the previous frame sees the same junction from 0.5 s earlier.
- **Step 0, the cheapest test, CPU, no training (the kill):** the E290 replay on the dump of E312 (`runs/e312_p_s<S>/ends_val.pkl`): the pair logit averaged with the ego-warped logit of the matched pair of the previous keyframe of the same scene (weight 0.5, then a sweep of the weight); read out with the snapped E312 read-out. Reported: flips (the share of true pairs delivered in one of two consecutive frames only), pair recall / precision, snapped topo F1. The scratch script `temp/e290/pair_flip.py` (E284 head) is adapted and PROMOTED into `evaluation/` with a test when it is used for this card (the git rule: it is the way a number is made); the matching of the pairs across frames uses the GT ends of `jwiring.py gt-ends` only for the flip statistic, never for the averaged logit.
- **Why the recursive form first:** at inference the previous frame's keypoint tokens are already computed (they are the previous output): the state costs a buffer, a warp (2-D rigid transform from the ego poses) and a small attention over at most 32 keypoint tokens: below 0.1 MMACs and no second forward pass of the maps head. The cross-attention form of the original card needs the maps head again on the previous frame (3.26 GMACs of the 7.96) and is the comparison arm.
- **Training the state:** the previous keyframe is the cached row - 1 of the same scene (row order of the cached features, `qgraph_view._infos()` poses, TRAIN scenes only); the state is E312's own frozen output on that row (tokens and heat), noised as CenterTrack does (position jitter, 10 % false tokens, 40 % missing) so that the head does not trust a perfect past; the gate is initialised at zero so the layer can only help; no backpropagation through time (the state is a constant input). Causal: only the past is used. The first frame of a scene has no state.
- **Hard steps (D55, named):** the choice of the previous row (the scene order), the matching of states by the warp (soft attention, no hard association), the p >= 0.1 threshold and everything after it as in E312. Nothing is added to the delivered path that is a rule.
- **Controls and confounds:** c (E312 of the same seed); the arm x for the recursion; a zero-gate run (the layer with the gate fixed at 0) proves the flag-off identity; one training per arm and seed (the scene bootstrap is the interval); the gain on the first frames of scenes (no state) must be zero and is reported as the null. A confound: the head trains 3000 steps like E312, the state adds parameters (about 15 k for the attention, estimated): the E316 control (the same head at 6000 steps gained +0.007 topo F1 and +0.04 pair precision) says that more training alone moves the numbers; the card therefore reads r against c at the SAME 3000 steps and reports E316's value as the cost of training length.
- **Prediction (numbers, written before the run; a hit is within +-0.02 for recall):** pair recall +0.03 (the E290 replay recovered about a third of the flips but cost true pairs; a learned gate should keep the gain without the loss), precision -0.005, snapped topo F1 +0.005; the arm x +0.04 recall.
- **Cost / price (reported next to it):** parameters about 15 k, MACs below 0.1 MMACs per frame for r, +3.26 GMACs for x; ms and host time measured on the idle GPU in the window; training about 1.9 s per step (the state is a constant input).

## 2. What has to be built (after the critic pass; flag off = E312 bit for bit)
Step 0 script and its test (promotion), the state layer in `training/grid/jends.py` behind a flag (`--state r|x`, default off), the loader hook for the previous row and the ego warp (`jtest_eval`-style frames of a split are not enough: the training rows), the CenterTrack noise, tests (the shapes, the zero gate = E312 bit for bit, the warp by a known pose, the first frame of a scene skips the layer, no gradient into the state, mutations: the state taken from the NEXT row (acausal), the warp in the wrong direction, the gate not zero-initialised, noise off), the READY file after a fresh critic pass on the code. No GPU before step 0 passes and the card is critic-accepted; then the windows.

## 3. Referee
The fresh `ar-critic` pass (cards mode) is written below when it has run.
