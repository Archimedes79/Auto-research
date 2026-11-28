# E423 what the soft fork gate does for the junctions

The card is the json block below: `research check <problem> E423` checks it, `research start` freezes it in the snapshot. What the runs do,
how the verdict is made and what counts is the card as the snapshot holds it; a changed experiment is a new experiment.

```json card
{
 "id": "E423",
 "title": "what the soft fork gate does for the junctions",
 "question": "on VAL, with the champion's weights and the skip and gap-closing layers: does the soft fork gate raise junction F1 without losing more than the guard in lanes found?",
 "hypothesis": "behind the first two layers 70 % of the delivered junctions are false on TRAIN, many of them a second successor at a node the head's own type softmax does not call a fork; keeping a second successor only as far as p_fork is above 0.05 raises junction F1 by about 0.03 on VAL at an unchanged topo F1",
 "prediction": 0.0,
 "kind": "compare",
 "control": "skipgap",
 "seeds": [0, 1],
 "arms": {
  "skipgap": {"set": {"import_weights": "runs/e420_skipgap_s{seed}"}, "note": "the champion's weights with the skip and gap-closing layers (E420's networks), predicted and scored now"},
  "fork": {"set": {"import_weights": "runs/e423_fork_s{seed}"}, "note": "the same and the fork gate"}
 },
 "rule": {},
 "kill": [],
 "note": "ordered by the owner on 2026-11-28 (D165: fix the junction errors, differentiable steps only); an attribution experiment: the candidate for an adoption is E422.layers; no adoption from this card"
}
```

## Why
TRAIN (291 frames, seeds 0 / 1; the network's own outputs), behind the skip and gap-closing layers: junction precision 0.30 / 0.29, recall
0.60 / 0.57. With the fork gate: junction F1 0.398 / 0.385 -> 0.435 / 0.424 (+0.037 / +0.039; precision 0.35 / 0.35, recall 0.57 / 0.55),
topo F1 +0.000 / +0.001, lanes found -0.008 / -0.008.

Null forms on the same outputs (does the type softmax matter, or only fewer links?):
| variant | junction F1 | topo F1 | lanes found |
|---|---|---|---|
| fork gate | 0.435 / 0.424 | 0.756 / 0.756 | 0.803 / 0.795 |
| every further successor cut (p_fork = 0) | 0.285 / 0.283 | 0.729 / 0.728 | 0.767 / 0.761 |
| p_fork shuffled among the kept nodes of a frame | 0.297 / 0.292 | 0.732 / 0.729 | 0.770 / 0.762 |

Left open, from hard-rule probes on TRAIN (`probe.py`, no record): a gate at merges as well raises junction F1 by about 0.24 (precision
0.82) but costs 0.026 topo F1, also with the truth in the type's place, so it is not part of this round. An earlier attempt at junction
precision by another route (E376, tag `v1`: a max-positive likelihood) did not reach its +0.03.

E399's skip layer in junctions was checked first: of the links it removes on TRAIN a third follow the truth, but keeping exactly those (an
oracle) lowers topo F1 and junction F1: they are second routes between the same nodes.

## Prediction and bar
VAL, `fork` against `skipgap`: topo F1 0.000 (band 0.010: the standard verdict TIE is expected); junction F1 +0.03; lanes found -0.008.
The gate works when, on each seed, junction F1 gains at least 0.015 and lanes found lose at most 0.010; otherwise it does not.

## How the verdict is read
The standard rule on topo F1 with the problem's guards; then the bar above, per seed, with junction precision and recall from the runs'
records.

## Design delta
R1: one fixed soft layer more (`jtoken.fork_gate`); the second successor of a node depends differentiably on the node's type softmax. R7: the
read-out's link and keep thresholds also in the forward.

## Imported networks
`<runs root>/lane_center/e423_fork_s{0,1}`: the champion's `e372_s4_s{0,1}/model.pt` with `skip_suppress`, `close_gaps` and `fork_gate`
added to its args (weights equal, checked; `import.json`).

## Critic (cards mode, 2026-11-28): revise
Taken in: numbers of the tested soft layer only, a bar per seed, the two null forms, E376 named, the unsupported explanation of the merge cost
dropped.

## Conclusion
**Verdict** (`research judge`, 2026-11-28, `verdict.json`): `fork` **TIE** against `skipgap`: topo F1 0.7135 -> 0.7108, -0.0027
[-0.0046, -0.0006] (as predicted, inside the band). VAL 873 frames in 22 scenes; nothing adopted.

**The bar of this card fails** (junction F1 gain >= 0.015 and lanes found down by at most 0.010, each seed):
| seed | junction F1 | precision / recall | lanes found | time per frame (ms) |
|---|---|---|---|---|
| 0 | 0.2973 -> 0.3273 (+0.030) | 0.208 / 0.522 -> 0.249 / 0.477 | 0.7266 -> 0.7114 (-0.0152) | 20.5 -> 21.3 |
| 1 | 0.2856 -> 0.3143 (+0.029) | 0.200 / 0.500 -> 0.237 / 0.467 | 0.7253 -> 0.7136 (-0.0117) | 21.1 -> 22.1 |

- Junction F1 gains as hoped; lanes found fall by more than the guard, on VAL nearly twice as far as on TRAIN (-0.008).
- One frame shows how (scene 39, seed 0, frame 787): the gate cuts a true turn at a node the type softmax does not call a fork, and the
  covered true lane length falls from 0.955 to 0.801; seed 1 keeps it.
- Follows: a milder gate (p_fork threshold 0.005, chosen on TRAIN after this result; not run yet, no number).
