# can22-minimalcycles: port card

Facts about the paper and code: see method.md and code.md (checked). Everything below about OUR code, cost and effort is a proposal / estimate, not measured.

## Block A: pair head (take it)
Interface (proposal):
- `PairHead(d_node, d_emb=64, d_hid=128)`: `forward(node_feat [B,N,d_node], idx [B,P,2]) -> logits [B,P]` with `idx` = candidate (entry, exit) pairs at a junction (our "all pairs" set), embedding MLP per node, ordered concat `[f_i ; f_j]`, MLP to one logit. Ordered, so a pair and its reverse differ.
- `pair_loss(logits, label [B,P], end_i [B,P,2], start_j [B,P,2], pos_w=3.0, margin, w_end=1.0)`: weighted BCE with logits (weight `pos_w*lab + (1-lab)`) plus the endpoint coupling (positives: 2 x L1 distance between end of i and start of j; negatives: 3 x max(margin - d, 0)). Multi-label rows are allowed (one entry, several exits).
- Differentiable: logits and the endpoint term wrt node features and predicted node positions. No matching needed if the node ids are the GT ids of the junction (our oracle setting); with predicted nodes, labels follow the node-to-GT assignment of our head.
Spec tests (ours):
1. Toy BCE: 3 entries x 3 exits, one positive per row, all logits 0, pos_w 3, mean reduction: loss = (3*3*ln 2 + 6*ln 2) / 9 = 15/9 * ln 2 = 1.1552. Pure numpy reference.
2. Toy endpoint term: positive pair with end_i = start_j gives 0; offset 0.1 gives 0.2; negative pair at distance 0.02 with margin 0.05 gives 3 * 0.03 = 0.09 (formula of detr.py as extracted, to be re-read in the file before the port).
3. Paper number (only with repairs and the author's label files): `validator_tr_nusc.py` with `TR_MC_Nusc.pth` should give C-F about 55.3 on nuScenes val (paper Table 1); MC-F 62.5 and order 0.776 are not reproducible from the shipped metric code (blocks commented out).
Cost: parameters and FLOPs negligible (P pairs through two small MLPs); one day with tests. Risk: low. Open: our pair scorer (0.71 junction F1) may already contain these ingredients; the three details to compare are the positive weight, the endpoint coupling and "matched nodes only" (compare against our current loss before porting).

## Block B: minimal-cycle cover as an auxiliary target (watch, not now)
Interface (proposal): `CycleHead(d_node, n_cycle_queries=32)` outputs membership logits `[B,M,N_slots+n_boundary]`, existence `[B,M]`, centre `[B,M,2]`; target from GT lane graph: planar arrangement of GT centrelines inside the BEV box (numpy or shapely `polygonize` of the union of centrelines and box boundary), each face = one cycle, membership = ids of centrelines on its border, centre = pole of inaccessibility. Needs Hungarian matching of cycles (query slots) and a re-labelling of GT cycles to predicted curves (the paper's `min_max` step).
Spec test: two straight directed lines crossing once inside a box: 4 minimal cycles (regions = 1 + n + m for n lines and m crossings, here 1+2+1), each with both lines and two boundary segments as members; a numpy sweep or shapely gives the same count. Paper-number test: none cheaper than retraining (H-GT-F 60.9 on nuScenes, Table 4, one read).
Cost (estimate): GT cycle extraction from our lane graph 1 to 2 days; head and matching 2 to 3 days; training cost small. Risk: high that it does nothing for junction F1: the paper's own C-F gain for the transformer on nuScenes is 0.1; the target is defined over query slots while our head works on dense nodes; cycle targets of estimated curves depend on a trained curve head.

## Idea derived from the paper (ours, untested, not in the paper)
Crossing-pattern consistency: which entry-to-exit paths of a junction cross each other is fixed by the GT pairs; a loss on soft pair probabilities times the geometric crossing indicator of the candidate paths would transmit the cycle idea without query slots. Sketch only; write a card before any work.

## Take / leave
Take: pair-head recipe (ordered concat, weighted BCE, endpoint coupling, matched-only supervision), the shared-start / shared-end fork-merge matrices as a possible extra target (weight 0 in the shipped script, so no evidence for them), the cover idea as a reading aid for a junction target.
Leave: DETR queries and Hungarian matching of curves, Polygon-RNN, the perspective backbone, the rule-based end-point merge (a named gap for us), the repository metric code, their cycle extraction pipeline.
What it does not settle: whether a pairwise MLP on better node features closes the gap between our learned pairs (0.71) and the oracle (0.88); the paper reports no junction-only pair metric.
