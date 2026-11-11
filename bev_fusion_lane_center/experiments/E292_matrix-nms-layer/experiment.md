# E292 Matrix-NMS as a layer inside the training graph (D82: "bleibe differenzierbar"; follows E288)

Card of 2026-11-11 (a plan card, not coded yet), written by the lane chat. E288 showed that the one-pass Matrix-NMS decay with R 1.75 m and sigma 0.1 reproduces the hard close-node step of the delivered graph (topo F1 0.6270 / 0.6250 against 0.6259 / 0.6249, vertices -0.05 / -0.10 per 100 m, both seeds) as a smooth operator on the existence scores and the node positions. The hard step is the named non-differentiable read-out; a layer after the node head would let the loss on the delivered graph reach the pruning (E277's three-pass soft step was a CUT, E279's 1.25 m version a KEEP that is open for Mirko).

## 0. The card (what the tools run)
```json card
{
 "id": "E292",
 "card": "",
 "title": "the lane head B with the Matrix-NMS layer (R 1.75, sigma 0.1) after the node head, the stages after it (links, sections, assignment) fine-tuned with the damped existence, against B with the hard step of record",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.02,
 "prediction": {"value": 0.003, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = B + the hard step `close:1.5@0.5` (runs e273_p_s0 / e273_p_s1): topo F1 0.6259 / 0.6249"},
  "m": {"recipe": "`QGraph(soft_nms=('mnms', 1.75, 0.1))`: `no['e']` after the node head is the Matrix-NMS-damped logit (the head's own stays in `no['e_raw']`, as E277); phases 4 - 7 fine-tuned from B's checkpoint (links, sections, assignment, joint), 2 seeds; read out WITHOUT the hard step (`plain@0.5`) and with it"}
 },
 "control": "c",
 "variant": "m",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "min_effect": 0.02},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.02},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 6.0,
 "kill": "the damped existence has no gradient to the node head (the gradient report of `qgraph.py` shows zero for `nodes`); or the layer at inference does not reproduce `qgraph_readout.matrix_nms` on the same scores (checked on 20 frames)",
 "evaluate": {"split": "val"}
}
```
By hand (the lane suite numbers, label "by hand: lane suite"): KEEP when topo F1 rises by more than +0.02 on both seeds with the guards held; a TIE (the likely outcome: the layer is the hard step in smooth form) is **kept and refined (D82)**: it then counts as the differentiable replacement of the hard step (the E2E claim: no rule step between the node head and the delivered graph), provided no guard breaks; CUT below -0.02 or a guard broken on both seeds. Reported: lanes found, Q, duplicates, ms of the layer (it is an (n x n) pass on at most 512 nodes).

## 1. The layer
`qnodes.matrix_nms_logit(e_logit, pos, R, sigma)`: the torch version of `qgraph_readout.matrix_nms` on the existence probabilities of the (up to 2500) cells restricted to the top-512 by e (an index choice with the gradient through the values, as `sections` / `qlinks`): overlap o = relu(1 - d / R), decay = min over the higher nodes of exp(-(o^2 - c^2) / sigma), c = the compensation, damped logit = logit(sigmoid(e) x decay). Differentiable in e and pos; the sort is by e (detached). The node positions of the loss are B's.

## 2. Design
Phases 4 - 7 of `qgraph_train.py` from B's checkpoint with `--soft-nms mnms 1.75 0.1` (the E277 switch extended), the E274 / E276 switches at their defaults; the gates of phases 4 - 7 as in E273. Two seeds. VAL 873, TEST never. Cost: about 3 GPU hours per seed.
**Prediction (a number):** topo F1 +0.003 over c (a TIE).

## 3. Not done here
The code (the torch layer and its unit test against the numpy function), the switch in `qgraph_train.py`, the run. Behind E287 / E289 in the queue; the night of 2026-11-11 has the pair head and the junction maps first.
