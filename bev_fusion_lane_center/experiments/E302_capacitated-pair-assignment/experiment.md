# E302 Entry-to-exit assignment with a learned capacity (one-to-many): Sinkhorn where an entry may feed several exits (D82)

Card of 2026-11-11 (plan card, behind E287). The scan (`scan_2026-11-11_junction_topology.md`): the field uses independent pair logits or a geometry prior; no lane paper puts a Sinkhorn / dustbin layer or an order constraint on entry x exit. E285 measured the degrees of the true pairs: 27 % of the entries have more than one exit (turn and straight from one lane), 33 % of the exits more than one entry (merges): a one-to-one transport with unit masses is wrong for about a third of the ends; a CAPACITATED transport (row mass = the predicted out-degree of the entry, column mass = the in-degree of the exit, a dustbin for the rest) is right. With the SD lane count (E291) the capacities have a prior.

Written after the first code review of 2026-11-12 (what the review changed: the plan entry is a mass and is divided by the smaller capacity before it is read as a probability; the card first said "NLL under P, read-out at 0.5 x the row mass"). The control p2 and this arm are both trained after commit 15f30cbc (the rank features come from real peaks in training now: an older p2 is not a valid control).

## 0. The card (what the tools run)
```json card
{
 "id": "E302",
 "card": "",
 "title": "pair head with a degree head (predicted out-degree of every entry, in-degree of every exit in {1, 2, 3+}) and a log-domain Sinkhorn with those masses and a dustbin, against the dual softmax of E287 p2",
 "kind": "block",
 "deterministic": false,
 "metric": "pair_precision_at_recall",
 "direction": "higher",
 "band": 0.03,
 "prediction": {"value": 0.05, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E287 p2 (BCE + dual softmax with a 'none' entry)"},
  "k": {"recipe": "`jends.py` + a degree head (2 linear layers on the keypoint tokens, cross-entropy against the true degrees from `jgt`), the pair probability = the transport plan P of a Sinkhorn (10 iterations, row masses = the expected out-degree, column masses = the expected in-degree, dustbin row / column), pair probability q = P / max(1, min(row mass, column mass)) (the share of the smaller capacity the pair uses, in [0, 1]), loss = BCE (pos_weight 3) on q + 0.5 x the degree CE (classes 0, 1, 2, 3+; 0 = a keypoint that is no GT end) + the crossing penalty of E287, no dual softmax; read-out q >= 0.5 (= logit >= 0 of the dump, so the E287 read-out and counts are unchanged)"}
 },
 "control": "c",
 "variant": "k",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "pair_precision_at_recall", "vs": "c", "min_effect": 0.03},
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.01},
  {"metric": "pair_recall", "vs": "c", "max_loss": 0.02}
 ],
 "budget_gpu_h": 1.5,
 "kill": "the Sinkhorn plan is not finite or the degree head predicts degree 1 for every end (the capacity carries no information)",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: pair scorer"): the headline is the pair precision at the pair recall of c (the point on the precision-recall curve of the head that has the same recall as the control; the curves come from sweeping the read-out threshold). KEEP above +0.03 on both seeds with the guards held; TIE kept and refined (D82): the degree head with the SD counts (E291) is the refinement; CUT below. Differentiable: Sinkhorn is a soft step; the read-out threshold is the named hard step.
