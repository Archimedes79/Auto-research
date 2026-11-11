# E302 Entry-to-exit assignment with a learned capacity (one-to-many): Sinkhorn where an entry may feed several exits (D82)

Card of 2026-11-11 (plan card, behind E287). The scan (`scan_2026-11-11_junction_topology.md`): the field uses independent pair logits or a geometry prior; no lane paper puts a Sinkhorn / dustbin layer or an order constraint on entry x exit. E285 measured the degrees of the true pairs: 27 % of the entries have more than one exit (turn and straight from one lane), 33 % of the exits more than one entry (merges): a one-to-one transport with unit masses is wrong for about a third of the ends; a CAPACITATED transport (row mass = the predicted out-degree of the entry, column mass = the in-degree of the exit, a dustbin for the rest) is right. With the SD lane count (E291) the capacities have a prior.

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
  "k": {"recipe": "`jends.py` + a degree head (2 linear layers on the keypoint tokens, cross-entropy against the true degrees from `jgt`), the pair probability = the transport plan P of a Sinkhorn (10 iterations, row masses = the expected out-degree, column masses = the expected in-degree, dustbin row / column), loss = NLL of the true pairs under P + the degree CE + the crossing penalty of E287; read-out at P >= 0.5 x the row mass"}
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
