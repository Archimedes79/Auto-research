# E299 One buffered frame for the junction head: ego-warped prior heat / cross-attention to the keypoints of the previous keyframe (brief: "Nutze Cross Attention ueber die Zeit") (D82)

Card of 2026-11-11 (plan card, behind E287 and E290). Literature (`scan_2026-11-11_pruning_uncertainty_time.md`, idea 1): CenterTrack feeds the previous detections as one extra heat-map channel (prior-off ablation -1.8 AMOTA on nuScenes), RVLD warps the previous output (flicker 0.053 -> 0.014); both at 10 - 30 Hz, the nuScenes keyframes are 0.5 s apart. E290 measured on the E284 head: 20 % of the true pairs flip between two keyframes; averaging with the warped previous logits cut the flips by 44 % but delivered fewer true pairs (36 % -> 31 %), and 52 - 54 % of the true pairs are delivered in neither frame: recall is the first problem.

## 0. The card (what the tools run)
```json card
{
 "id": "E299",
 "card": "",
 "title": "the keypoint tokens of E287 p2 get a cross-attention to the ego-warped keypoint tokens (and heat) of the previous keyframe through a zero-initialised gate, trained with the CenterTrack noise on the previous frame (jitter, 10 % false, 40 % missing), against p2 without it",
 "kind": "block",
 "deterministic": false,
 "metric": "pair_recall",
 "direction": "higher",
 "band": 0.03,
 "prediction": {"value": 0.04, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E287 p2"},
  "t": {"recipe": "`jends.py` + a gated cross-attention layer over the previous frame's keypoints (positions warped by the relative ego pose of the two lane frames, tokens from the same frozen maps head run on the previous cached features; the first frame of a scene has no previous frame: the layer is skipped); zero-initialised gate, so it can only help; 2 seeds"}
 },
 "control": "c",
 "variant": "t",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "pair_recall", "vs": "c", "min_effect": 0.03},
  {"metric": "pair_precision", "vs": "c", "max_loss": 0.02},
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.01}
 ],
 "budget_gpu_h": 2.0,
 "kill": "the E290 replay on the E287 dump (the zero-cost average with the warped previous logit) shows neither a flip reduction of at least 25 % relative with topo F1 held nor a recall gain: then the previous frame is a weak witness at 2 Hz and the card is not run",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: pair scorer + pair flip rate"): KEEP when the pair recall rises by more than 0.03 on both seeds with the guards held; the flip rate (E290 definition) is reported, TIE kept and refined (D82; the refinement is a longer memory: two buffered frames, or the world memory of the BEV system's idea); CUT below. **Data:** the previous keyframe of the same scene (row - 1 in the cached feature order, same scene), the relative pose from the CRN infos (`qgraph_view._infos()`, the code of `temp/e290/pair_flip.py`), TRAIN scenes only for training. **Causal:** only the past frame is used.
