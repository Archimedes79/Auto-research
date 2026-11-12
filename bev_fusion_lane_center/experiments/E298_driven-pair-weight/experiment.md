# E298 Weight the connections the ego drives (brief: "Wichte Verbindungen wo man faehrt hoeher") (D82)

Card of 2026-11-11 (plan card, behind E287; code after the night chains because it touches the shared loader and the pair loss). The lane GT l4 labels the path the ego really drives (tier 1, non-causal: labels only, never an input); in `jgt` every true pair carries `driven` (the connector run has tier 1 for most of its points): 14 % of the pairs in VAL. A training weight makes the pairs that matter for the planner count more; at inference the route (the SD route, E291) plays the same role as a prior.

## 0. The card (what the tools run)
```json card
{
 "id": "E298",
 "card": "",
 "title": "pair head p2 of E287 with the positive terms of the pair BCE weighted by 1 + w x driven (w = 2; the dual softmax stays unweighted), against p2 with equal weights",
 "kind": "variant",
 "deterministic": false,
 "metric": "driven_pair_recall",
 "direction": "higher",
 "band": 0.03,
 "prediction": {"value": 0.06, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E287 p2 (same maps, same seeds)"},
  "w": {"recipe": "`jends_train.py --driven-w 2` (batch key `pdrv` from `jgt` pairs: 1 for a pair the ego drives; the weight multiplies the positive terms only); everything else as p2"}
 },
 "control": "c",
 "variant": "w",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "driven_pair_recall", "vs": "c", "min_effect": 0.03},
  {"metric": "pair_recall", "vs": "c", "max_loss": 0.02},
  {"metric": "pair_precision", "vs": "c", "max_loss": 0.02},
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.01}
 ],
 "budget_gpu_h": 1.0,
 "kill": "the driven flag does not line up with the pairs of the batch (the count of driven pairs differs from `jgt`'s)",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: pair scorer, `jsplits.py` for the driven split"): KEEP when the recall of the driven pairs rises by more than 0.03 on both seeds with the guards held; TIE kept and refined (D82; the follow-up raises w or adds the route prior); CUT when the overall pair recall / precision or topo F1 breaks a guard on both seeds. Reported: the driven split of E300 for both arms. **Not an input at inference:** `driven` is non-causal; the inference-time analogue is the route of the SD map (E291), which the head may take as a feature; tested separately.

## 2. Amendment 2026-11-12 14:05 (before any run of this card): the arms move to the system of E312
The card was written on E287 p2 (frozen maps of E286). The system of the thread changed on 2026-11-12 (E307 maps, E305 joint training, E303 scales, E311 snapped read-out: E312, KEEP), so the question is asked on it: control c = E312 of the same seed (pair head `--e2e --scales` on the E307 maps, `runs/e312_p_s<S>`), arm w = the same with `--driven-w 2` (run `e298_w_s<S>`, BFLC_SIG=1.0). Metric and rule unchanged: the recall of the driven pairs (`jsplits.py score`, logit >= 0, p >= 0.5, the head's own operating point) rises by more than 0.03 on both seeds with the overall pair recall / precision within 0.02; topo F1 on the snapped read-out (p >= 0.1) is the added guard (not lower than c by more than 0.01). Cost: one pair training per seed (about 70 minutes).
