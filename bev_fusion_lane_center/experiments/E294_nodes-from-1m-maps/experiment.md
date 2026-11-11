# E294 Is the 2 m node grid needed? Nodes read from the 1 m maps only (brief: "Wenn das klappt brauche ich kein 2x2m Grid ... ggf. eine Pyramide fuer Kontext") (D82)

Card of 2026-11-11 (plan card, behind E286). If the 1 m maps (lane centre of E271, the junction maps of E286 with entry / exit heat and heading, the 0.5 m lines) are good, the node set can be read from them (differentiable sparsification of the 1 m centre map, E271: coverage 0.72 / 0.73, precision 0.87 at 10.3 vertices per 100 m) and the 2 m grid becomes a context level (the pyramid) only. The 2 m node head of B is the control: node F1 0.851 / 0.850 at 2 m, recall 0.83, 32 nodes per 100 m, median position error 0.28 m, but locked to the grid (E289).

## 0. The card (what the tools run)
```json card
{
 "id": "E294",
 "card": "",
 "title": "nodes from the 1 m maps (E271 sparsification of the centre map, junction maps as type channels) against the 2 m node head of B, scored as nodes by the lane suite's node counts at 2 m",
 "kind": "block",
 "deterministic": false,
 "metric": "node_F1",
 "direction": "higher",
 "band": 0.03,
 "prediction": {"value": 0.0, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = B's delivered nodes (close:1.5@0.5), node F1 at 2 m 0.851 / 0.850, recall 0.83, precision 0.87 (the card runs/e273)"},
  "m": {"recipe": "the nodes of the E271 head (`soft_lines.sparsify` on the 1 m centre map, budget 12 per 100 m) with the type read from the E286 entry / exit / split / merge heat at the node (types need E301 for split / merge); no 2 m grid, no node head; scored with the same node counts (the node is a match within 2 m of a GT node)"}
 },
 "control": "c",
 "variant": "m",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "node_F1", "vs": "c", "min_effect": 0.03},
  {"metric": "node_R", "vs": "c", "max_loss": 0.05},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0}
 ],
 "budget_gpu_h": 1.0,
 "kill": "the 1 m nodes cannot be scored by the node counts of `qgraph_eval` (no `exists / flags / offset` contract): then the card is a vertex-level card scored by `vertex_eval.py` (cov5 / prec1) and the rule is rewritten before the run",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: node counts"): KEEP when node F1 rises by more than 0.03 on both seeds with the recall and vertex guards held; TIE (the likely outcome) is kept and refined (D82): the 1 m route stays as the alternative representation and the 2 m grid as context; CUT below -0.03 or a broken guard. Reported: the grid-phase jitter of the 1 m nodes (the E289 probe), ms.

## 1. Why it is not the first thing
E286 first (do the 1 m maps see the ends?), E289 first (is the grid phase a problem for the line?). If E286's ends are good and the E289 line probe says the lines are stable, the 2 m grid costs nothing and E294 stays a TIE; if the grid phase hurts the delivered curve, E294 is the fix. The 1 m centre head of E271 exists (runs `e271_a_s0 / s1`).
