# E290 One buffered frame for the junction pairs: how often do the pairs flip between keyframes, and does an ego-warped prior heat help (D82, thread: "Cross Attention ueber die Zeit")

Card of 2026-11-11 (a plan card: the CPU measurement first, the training only on its word), written by the lane chat on Mirko's thread (D82) and the open item of the HANDOVER (3: "Spurmitte has NO temporal fusion ... first on the end keypoints, rule = pair flip rate + topo F1"). Literature (`scan_2026-11-11_pruning_uncertainty_time.md`, idea 1): CenterTrack feeds the previous detections as one heat-map channel (prior-off ablation -1.8 AMOTA on nuScenes); RVLD warps the previous output (flicker rate 0.053 to 0.014, F1 +0.04); caveat: both use 10 - 30 Hz video, nuScenes keyframes are 0.5 s apart (from memory, to check in the deep read). The scan's cheapest first test is this card's arm w.

## 0. The card (what the tools run)
```json card
{
 "id": "E290",
 "card": "",
 "title": "pair flip rate of the junction heads between consecutive VAL keyframes, and the effect of warping the previous frame's pair logits by the ego motion and averaging them with the current ones (no training)",
 "kind": "variant",
 "deterministic": true,
 "metric": "pair_flip_rate",
 "direction": "lower",
 "band": 0.02,
 "prediction": {"value": 0.30, "vs": "c"},
 "arms": {
  "c": {"recipe": "the pair decisions of the head of record (E284 h2 now, E287 when it is scored) at p >= 0.5 on every VAL keyframe; pair flip rate = over the pairs that exist in the GT of BOTH of two consecutive keyframes (the same entry and exit, ego-warped within 3 m and 30 degrees; the GT of the two frames identifies them), the share that is delivered in exactly one of the two frames"},
  "w": {"recipe": "the logit of every pair of the current frame averaged with the ego-warped logit of the matching pair of the previous frame (weight 0.5), the pair then read out at 0.5; the frames without a previous frame (the first of a scene) unchanged"}
 },
 "control": "c",
 "variant": "w",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "pair_flip_rate", "vs": "c", "min_effect": 0.25},
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.005}
 ],
 "budget_gpu_h": 0.0,
 "kill": "the ego warp does not map the GT ends of frame t onto the GT ends of frame t+1 for at least 95 % of the true pairs present in both (the pose chain is wrong)",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: replay on stored head outputs"; `min_effect` for the flip rate is RELATIVE: 0.25 = a quarter of c's rate): GO for the training arm (a learned prior channel / cross-attention to the previous keypoints, zero-initialised so that it can only help: a refinement, D82) when the flip rate falls by at least 25 % relative with topo F1 not lower by more than 0.005 on both seeds; NO-GO if the flip rate is already below 0.10 (nothing to fix) or the warp helps neither number (then the 0.5 s gap makes the previous frame a weak witness and the card says so). Reported: the flip rate by the number of lanes of the arm, by turn class, by ego speed (a stopped car has the same view twice), the share of pairs that are delivered in neither frame.

## 1. What the measurement needs
(1) Two consecutive keyframes of one scene and the relative pose between the lane frames of the two (the lane frame is the ego frame at the CAM_FRONT time of the keyframe, x right, y forward; the poses are in the CRN infos that `qgraph_view._infos()` already reads); (2) the GT ends of both frames (`jgt.py`) to say which pair is which (not the predicted ones: the flip rate is about the decision, not about the keypoint extraction); (3) the head outputs `runs/<run>/ends_val.pkl` (keypoints, logits, `allowed`) of every VAL frame. The matching of a pair in frame t+1 to its predecessor in frame t is by the two ends (entry and exit each within 3 m and 30 degrees after the warp).

## 2. Design
CPU only, replay: the E284 head first (it exists), the E287 head when its dump exists. Two seeds. Controls: c; a shuffled-frame control (the previous frame of another scene as the "previous" one): the flip rate must not fall (it would mean the average only smooths).
**Prediction (a number):** pair flip rate of c about 0.30 for the E284 head (its pair recall is 0.31: most true pairs are delivered in none of the frames, so the rate is over the minority that are delivered in one); arm w -25 % relative or more is the GO line.

## 3. Command
`temp/e290/pair_flip.py` (scratch, to be promoted into `evaluation/` when used a second time). Result `runs/e290/pair_flip_<run>.json`.

## 4. First measurement (2026-11-11, CPU, `temp/e290/pair_flip.py`, the E284 h2 head, VAL, 804 consecutive keyframe pairs of 22 scenes, 8887 true pairs that exist in both frames of a pair; weight 0.5; the previous frame only, causal)
| | seed 0 | seed 1 |
|---|---|---|
| pair flip rate of the head of record (delivered in exactly one of the two frames) | 0.200 | 0.195 |
| delivered in both / in neither frame | 26 % / 54 % | 29 % / 52 % |
| flip rate with the ego-warped previous logit averaged in (arm w) | 0.112 (-44 % relative) | 0.111 (-43 %) |
| delivered in neither frame with arm w | 63 % | 61 % |
| true pairs delivered per frame-pair with c / with w (share of the 2 x 8887 pair-frames) | 36.4 % / 31.5 % | about 38 % / 33 % |
Reading: (a) one in five true pairs flips between two keyframes (the GO line of the card was -25 % relative with topo F1 held); (b) the averaging cuts the flips by 43 - 44 % relative, but it does so mostly by delivering FEWER true pairs (delivered in neither: 54 -> 63 %, true pairs delivered 36 -> 31 %): it smooths by shrinking, the precision side (false pairs that flip) and topo F1 are not measured here; so the GO line is NOT declared yet, the measurement needs the false pairs and the delivered-graph score (the replay with the connectors, `qends_eval.score` with the previous frame's pairs); (c) the dominant fact is recall, not flips: 52 - 54 % of the true pairs are delivered in neither of two consecutive frames, so a better pair decision (E287) is worth more than a temporal average of the E284 one; the temporal arm is re-measured on the E287 dump. The kill condition (the warp maps the GT ends of frame t onto frame t+1 for at least 95 % of the true pairs present in both) is not read directly: `pairs_both` is the count of pairs with both ends matched, 8887 of the true pairs of the 804 frame pairs (the share of all true pairs in consecutive frames is to be printed next).
