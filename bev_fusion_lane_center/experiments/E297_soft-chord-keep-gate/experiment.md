# E297 Prune the points on the line by their removal cost: a soft chord-distance keep gate on the node chain (brief: "Punkte auf Linie nahe an Node prunen ... Spezialtypen nicht prunen") (D82)

Card of 2026-11-11 (plan card). The delivered graph has 19 - 20 vertices per 100 m after the output stage (Douglas-Peucker 0.3 m, budget 14.3). A vertex that lies within eps of the chord of its two neighbours on the line adds nothing; Douglas-Peucker removes it but is not differentiable. The scan found no published differentiable Douglas-Peucker; the pieces exist: the removal cost of a vertex = its distance to the chord of the neighbours (one matrix operation over the neighbour index lists of the links), a keep gate sigmoid((cost - eps) / tau) with a budget loss (the E271 sparsification, `soft_lines.sparsify`), special types (stop / exit / split / merge nodes) get a gate of 1.

## 0. The card (what the tools run)
```json card
{
 "id": "E297",
 "card": "",
 "title": "a learned keep gate per node (removal cost = distance to the chord of the neighbours along the links, special types protected) with a vertex budget loss of 14.3 per 100 m, trained on the frozen B chain, against B with the hard output stage (DP 0.3 m)",
 "kind": "block",
 "deterministic": false,
 "metric": "Q",
 "direction": "higher",
 "band": 0.010,
 "prediction": {"value": 0.005, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = B + the hard step of E275 + the output stage `simplify_dp` (the delivered graph of record, vertices per 100 m 19 - 20 with the node ids, 14.3 budget on the DP basis)"},
  "k": {"recipe": "`qkeep.py` (to write): gate g_i = sigmoid((cost_i - 0.3) / 0.1) for lane nodes, 1 for special types; the delivered vertex set = the nodes with g >= 0.5 (the hard read-out is the named step); loss = the E271 budget loss on sum g per 100 m of delivered line + a reconstruction term (the Hermite polyline through the kept nodes stays within 0.3 m of the all-nodes polyline); phase 3b on frozen B, 2 seeds"}
 },
 "control": "c",
 "variant": "k",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "Q", "vs": "c", "min_effect": 0.01},
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 0.0}
 ],
 "budget_gpu_h": 1.5,
 "kill": "the chord cost of the neighbours cannot be computed for the links of a frame without a loop over pairs (> 5 ms per frame): then the operation is vectorised or the card is cut",
 "evaluate": {"split": "val"}
}
```
By hand (the lane suite numbers; label "by hand: lane suite"): the headline Q (S without the sparsity term) at the same or a lower vertex count. KEEP > +0.01 on both seeds with the guards held (the vertex guard is "not more than the control"); TIE kept and refined (D82): the gate is the differentiable replacement of the DP output stage, which is the point (E2E: no rule step at the end); CUT below -0.01 or a broken guard. The gain, if any, is speed and the E2E claim, not Q: the DP stage already meets the budget (scan, idea 5).
