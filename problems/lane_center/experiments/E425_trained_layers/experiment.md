# E425 the head trained with the soft output layers in its forward

The card is the json block below: `research check <problem> E425` checks it, `research start` freezes it in the snapshot. What the runs do,
how the verdict is made and what counts is the card as the snapshot holds it; a changed experiment is a new experiment.

```json card
{
 "id": "E425",
 "title": "the head trained with the soft output layers in its forward",
 "question": "does the champion's stage 2, trained with the four soft output layers inside the head's forward (the loss reads the graph after them), deliver a better VAL graph than the champion's weights with the same layers switched on afterwards?",
 "hypothesis": "the loss reads the wiring after the layers, so the head and B''s node head can raise links the layers damp wrongly and lower the fork probability of false forks; trained this way the topo F1 is higher by 0.005 or more than with the layers switched on afterwards, at junction F1 not lower",
 "prediction": 0.005,
 "kind": "compare",
 "control": "fixed",
 "seeds": [0, 1],
 "steps": ["stage2", "predict", "evaluate"],
 "arms": {
  "fixed": {"from": "E422.layers", "note": "the champion's weights, the layers switched on after the training"},
  "plain": {"set": {"stage2": {"head_run": "e373_cb_s{seed}"}}, "note": "the champion's stage 2 from the champion's stage-1 head, trained on this commit without the layers: the check that the training repeats (it must give the champion's weights)"},
  "trained": {"set": {"stage2": {"head_run": "e373_cb_s{seed}", "output_layers": "soft_output"}}, "note": "the same stage 2 with the layers in the forward (3000 steps)"}
 },
 "rule": {},
 "kill": [],
 "note": "ordered by the owner on 2026-11-28 (D165: end-to-end differentiable techniques); written before any result of E422; plain is expected far below fixed (it is the champion without layers) and is judged only as a check; no adoption without the owner"
}
```

## Why
E422's layers are switched on after the training. Trained end to end they are part of the network: the loss (the likelihood of the true
successors and the type cross-entropy) reads the wiring after them, and the gradient runs through their sigmoids and maxima into the head and
B''s node head. The position layer moves only the delivered positions, which no loss reads: it trains nothing here.

## Plan
- Code: `jlive_train --output-layers soft_output` (a key of the config block `stage2`, default off; the settings go into the model's args).
  For a finite loss behind the layers: `skip_suppress` keeps 0.001 of a damped link, `close_gaps` never takes more than half of a source's
  old mass (so its none slot stays), tested (`test_layers_train`, `test_jlive_train`: two steps with the layers).
- Checks before the start: `test_jtoken`, `test_lane_train`, `test_jlive_train`.
- Runs: stage 2 of both seeds, with and without the layers (about 25 minutes each, two at a time), predict, evaluate. TEST is not touched.

## Risks and what is read with the verdict
- The loss sums over a source's true successors, so it can put all mass on one branch of a fork; the fork gate then sees one successor and
  nothing pushes the other one back up: junction recall can fall.
- A weak true link that the gap layer lifts above its threshold gets a bounded loss without being raised itself.
- Read from the records: junction precision and recall, lanes found, vertices; from the training logs: the link likelihood (nll_s, nll_l) of
  `trained` against `plain` at the same steps; from the dumps: delivered links per frame.

## Prediction and what would refute it
VAL, `trained` against `fixed`: topo F1 +0.005 (inside the band of 0.010: the standard verdict TIE is the likely one). Refuted: a gain below 0
on either seed, or junction F1 lower by more than 0.01.

## How the verdict is read
First the check: `plain` stage-2 weights (head and B') equal to `e372_s4_s{0,1}` bit for bit (compared by script, reported). If they differ,
`trained` against `fixed` is not like with like and says nothing until `plain` with the layers switched on afterwards is measured as a further
experiment. Then the standard rule on topo F1 with the problem's guards; junction numbers, heading error, time and memory are reported.

## Design delta
R1: the soft output layers become part of the training: a loss on the output reaches every module through them; their settings stay fixed.
R6: stage 2 of the champion's recipe would read the loss after the layers. R7: the read-out's thresholds are inside the forward and so inside
the loss.

## Conclusion
