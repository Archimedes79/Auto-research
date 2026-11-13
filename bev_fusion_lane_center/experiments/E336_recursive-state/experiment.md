# E336 - temporal fusion by recursion: the head carries its own feature state from the previous keyframe, ego-warped, instead of a second input (plan card)

Card of 2026-11-13, written by the lane chat (BEV Fusion Spurmitte side session) on Mirko's order in chat ("Plane zeitliche Fusion per Rekursion"; his earlier question: can the fusion use the feature maps computed before instead of 2 inputs?). **Status: planned, NOT started** (no code, no run; GO belongs to the lane lead after the E320 follow-up, GPU only in the windows of D84 / D86; the holds of D85 on E317 / E318 / E319 are untouched). Relation to E299 (D87 point 3: "the cheapest variant first"): E299 gives the junction head a cross-attention to the KEYPOINT tokens of the previous frame (a second input of the head, tokens re-computed from the previous cached features); E336 gives the head a STATE, so the network runs once per frame and the past arrives as one tensor. If E336's gate stays closed (kill below) E299 is the next variant, not both at once.

**Why recursion.** (1) Cost: the frozen GaussianCaR cache is read once per frame; the previous frame costs one warp and one 3x3 convolution on the head grid, not a second forward pass (the pair input of E299 runs the maps head twice per sample). (2) The weak spot is recall, not ranking: E290 measured 20 % of the true pairs flipping between two keyframes and 52 - 54 % of the true pairs delivered in neither of two frames; a second look through a state can add what one frame misses, an average of logits (E290 replay) cannot. (3) E2E (D55): the state is a tensor on the forward path, a loss on the output reaches the previous frame's contribution through the warped state; no rule step is added (the warp is a fixed bilinear resampling by the known ego pose, differentiable in the features, the pose is data). (4) Foundation: GaussianCaR stays frozen (D50), the recursion lives in the lane head only; no change of shared code.

**Design.** State `s_{t-1}` = the output feature map of the shared trunk of the maps head (C channels on the 2 m head grid, after the pyramid merge), kept with the frame index. Per frame: `f_t` = trunk feature of the current cache; `w = warp(s_{t-1}, T_{t<-t-1})` by the relative ego pose of the two lane frames (bilinear `grid_sample`, zero outside the old view); `valid` = warped coverage mask; `h_t = f_t + g * conv3x3([f_t, w, valid])` with `g` a zero-initialised per-channel gate (the start is exactly the E312 network); `s_t = h_t`. Heat maps, links and the pair head read `h_t` as they read `f_t` now. First frame of a scene: `w = 0`, `valid = 0` (cold start). Training sample = K = 2 consecutive keyframes of one scene (TRAIN scenes only; causal: only the past): the previous frame's state is computed without gradient (truncated at depth 1, cheap), the loss is on the current frame; noise on the state as in CenterTrack (pose jitter 0.3 m / 1 deg, 20 % state dropout to the cold start) so that the head does not trust a perfect past. Inference online: carry `s_t` frame to frame inside a scene, reset at the scene start. Depth 2 (gradient through two steps) is a refinement arm only after the first result.

## 0. The card (what the tools run)
```json card
{
 "id": "E336",
 "card": "",
 "title": "E312 maps + pair head with a recursive ego-warped feature state of the previous keyframe (zero-initialised gate, truncated at depth 1, state noise), against the same E312 recipe without it",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.01, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E312 of the same seed (maps e312_m2_s<S>, pair head e312_p_s<S>), snapped connectors at p >= 0.1: topo F1 0.6825 / 0.6864"},
  "r": {"recipe": "the same maps and pair-head recipe and steps (3000, `--e2e --scales`, BFLC_SIG=1.0) with the recursive state in the shared trunk (code: a `--state` switch of `jmaps.py` / `jends_train.py`, selftest with a zero gate = E312 output bit for bit), sequences of 2 keyframes, state noise as above; 2 seeds"},
  "z": {"recipe": "same architecture and training as r, but the state is always the cold start (w = 0, valid = 0): the control for the extra convolution and the changed sampling; 2 seeds"}
 },
 "control": "c",
 "variant": "r",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "min_effect": 0.01},
  {"metric": "topo_F1", "vs": "z", "min_effect": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.01},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 9.0,
 "kill": "the gate does not open: the mean absolute gate over the last 500 steps stays below 0.02 on both seeds (the head ignores the past), or the loss is not finite; then E299 (keypoint tokens) is the next variant and E336 is closed by data",
 "evaluate": {"split": "val"}
}
```
By hand, reported next to the rule (label "by hand: temporal diagnostics", never a verdict): (1) first frames of a scene (cold start) and later frames separately: the cold-start frames must not lose more than 0.01 topo F1 against c; (2) the E290 pair-flip rate between consecutive frames for c and r (the temporal stability that the state is meant to buy); (3) pair recall in frames where the true pair was missed in the previous frame of c; (4) a shuffled-state read-out of r (the state of another scene): if it scores like r the head uses the parameters, not the past; (5) ms per frame (idle GPU, batch 1, state carried) and parameters against c; (6) the truncated-depth question: depth 2 only if r is KEEP or TIE with the gate open. The verdict is the written rule; a TIE is "keep and refine" (D82), refinement = depth 2, or the state at the pair head only (smaller), or E299's tokens.

**Prediction (a number, written before the run).** topo F1 snapped +0.01 over c on both seeds (the same size as the pair training length, E316 +0.007, and below the ceiling of the oracle ends, +0.10). Reason for caution: 0.5 s between keyframes, 20 % of the pairs flip, the camera changes; the gain is expected from recall in frames that the neighbour sees well, not from smoothing.

**Cost / what it shows the user.** Training about 1.2 x the E312 step (one extra no-gradient trunk pass on the previous frame; not measured yet: measure in the smoke run), inference +1 warp +1 conv on the head grid, no second forward pass. Budget 9 GPU h = 3 arms x 2 seeds x about 1.5 h, two slots, a start only inside the windows.

**Order of work.** (1) code: `--state` in the trunk with the zero-gate selftest (output equals E312 bit for bit), the sequence sampler (previous row of the same scene, relative pose from the CRN infos as in `temp/e290/pair_flip.py` / `qgraph_view._infos()`; promote to the lane folder, not scratch, with its test and README row); (2) critic pass on the card and the code (`ar-critic` cards, then code); (3) smoke 40 steps, ms and memory; (4) run r and z with seeds 0 / 1 through `run.py` (`run_ids` listed in this card before the launch); (5) `decide.py`, then the diagnostics above.

**Not in this card.** Recursion inside GaussianCaR (frozen foundation, D50); a long memory (world map of the BEV system); an unstructured transformer over time (deferred by D84 point 6); TEST (E321 is the one TEST use of E312, D84).
