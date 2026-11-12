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

## 3. Result (2026-11-12 16:50; E298 on the E312 system, trained 15:20 - 16:32 (`--e2e --scales --driven-w 2`, BFLC_SIG=1.0), `temp/day10/day10.sh`; control c = E312 of the same seed; the driven split by `jsplits.py score --run <run>`, 1553 driven GT pairs of 11083; `runs/e298_w_s<S>/ends_head.json`, `ends_card_snap_thr0.1.json`)
| seed | arm | recall of the driven pairs | recall of the other pairs | overall pair recall / precision (the head's own numbers) | snapped p >= 0.1: topo F1 (vs plain) | lanes found | junction F1 |
|---|---|---|---|---|---|---|---|
| 0 | c = E312 | 0.373 | 0.286 | 0.300 / 0.674 | 0.6825 (+0.0566) | +0.1109 | +0.0481 |
| 0 | **E298 (driven weight 2)** | **0.495** (+0.122) | 0.301 | 0.331 / 0.670 | 0.6766 (+0.0507) | +0.1114 | +0.0430 |
| 1 | c = E312 | 0.398 | 0.317 | 0.333 / 0.665 | 0.6864 (+0.0615) | +0.1286 | +0.0650 |
| 1 | **E298 (driven weight 2)** | **0.511** (+0.113) | 0.317 | 0.346 / 0.647 | 0.6865 (+0.0616) | +0.1326 | +0.0659 |
**Verdict by the rule: KEEP** (driven pair recall +0.122 / +0.113 against +0.03 on both seeds; guards: overall pair recall +0.031 / +0.013, precision -0.004 / -0.018 (inside 0.02; with the strict matching of `jpair_pr.py` -0.016 / -0.021), snapped topo F1 -0.0059 / +0.0001 over E312, inside the 0.01). The prediction (+0.06) is exceeded about twofold. The head learns the pairs the ego drives (the planner's pairs) far better and loses nothing measurable elsewhere: precision at the recall of E312 +0.000 / -0.013, the pairs that are not driven keep their recall (0.301 / 0.317 against 0.286 / 0.317). **Use:** the planner-facing variant of the final system (E312 + `--driven-w 2`); the delivered-graph score is the same as E312's within the noise (the topo F1 does not distinguish driven from other pairs). Not an input at inference (the weight acts on the loss only).
