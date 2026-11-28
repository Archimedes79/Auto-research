# E422 the champion measured again with the new evaluation, and the four soft output layers against it

The card is the json block below: `research check <problem> E422` checks it, `research start` freezes it in the snapshot. What the runs do,
how the verdict is made and what counts is the card as the snapshot holds it; a changed experiment is a new experiment.

```json card
{
 "id": "E422",
 "title": "the champion measured again with the new evaluation, and the four soft output layers against it",
 "question": "on VAL with the new evaluation: what does the champion deliver in quality, time and memory, and does the champion's network with the four soft output layers (skip link, gap closing, fork gate, positions) deliver a better graph without breaking a guard?",
 "hypothesis": "the layers act only on the wiring and on the node positions: damping covered jumps and closing gaps raises topo F1, the fork gate removes false forks (junction F1 up), the position layer lowers the heading error (zig-zag) and raises F1 at 0.5 m; on TRAIN all of this holds with 0.007 fewer lanes found",
 "prediction": 0.04,
 "kind": "compare",
 "control": "champ",
 "seeds": [0, 1],
 "arms": {
  "champ": {"set": {"import_weights": "runs/e372_s4_s{seed}"}, "note": "the champion's weights (E372 S4), predicted, timed and scored now"},
  "layers": {"set": {"import_weights": "runs/e424_layers_s{seed}"}, "note": "the champion's weights; the head's args turn on skip_suppress, close_gaps, fork_gate and refine_positions (jtoken.OUTPUT_LAYERS soft_output)"}
 },
 "rule": {},
 "kill": [],
 "note": "ordered by the owner on 2026-11-28 (D165): measure again instead of keeping old results compatible, fix junction and zig-zag errors with differentiable steps only; the candidate is layers; the runs are started one after the other so that the times are measured on an otherwise idle GPU; no adoption without the owner"
}
```

## Why
The evaluation changed (heading error, time, memory, a short list of numbers; the evaluator hash is new), so no result of before is
comparable and the champion is measured again. The layers answer errors found on TRAIN: jumps over nodes and unfilled gaps (E399, E420),
false forks (junction precision 0.30 behind the first two layers) and node positions that zig-zag across the lane (the node head shrinks its
offset toward the cell centre, and neighbouring nodes miss the lane to different sides).

## The layers (`net/model/jtoken.py`; no parameter, in the head's forward, off by default)
1. `skip_suppress` (E399): a link that another path of the same graph covers keeps 0.001 + (1 - cover) of itself.
2. `close_gaps` (E420): a weak link that is the only way on from an open end is delivered; since E420 changed so that a source's none slot
   keeps at least half of its mass.
3. `fork_gate` (E423): a source keeps a second successor only as far as its own type softmax calls it a fork (p_fork above 0.05).
4. `refine_positions` (E424): the offset from the 2 m cell centre times 1.2 inside the cell; then half of one step that moves the two ends of
   every short delivered link onto a common circular arc of their headings.
No layer reads ground truth. All are built from sigmoids, maxima and clamps (differentiable almost everywhere); the skip layer searches the
8 strongest links of a node (a hard cut inside the forward), and the sigmoids with width 0.01 are close to steps. The read-out keeps its four
named thresholds. The settings were chosen on TRAIN (scratch scripts, no record).

## TRAIN (291 frames, seeds 0 / 1; the network's own outputs on the GPU, this scorer)
| variant | topo F1 | junction F1 | F1@0.5 | F1@1.0 | lanes found | heading error (deg) | vertices / 100 m |
|---|---|---|---|---|---|---|---|
| champion | 0.7192 / 0.7225 | 0.340 / 0.340 | 0.742 / 0.738 | 0.905 / 0.904 | 0.810 / 0.802 | 11.26 / 10.99 | 25.3 / 25.3 |
| skip + gap | 0.7562 / 0.7556 | 0.398 / 0.385 | 0.737 / 0.733 | 0.905 / 0.902 | 0.811 / 0.803 | 11.66 / 11.54 | 23.2 / 23.3 |
| + fork gate | 0.7563 / 0.7561 | 0.435 / 0.424 | 0.737 / 0.733 | 0.904 / 0.902 | 0.803 / 0.795 | 11.39 / 11.32 | 22.4 / 22.4 |
| + positions (layers) | 0.7608 / 0.7601 | 0.435 / 0.425 | 0.753 / 0.749 | 0.906 / 0.904 | 0.802 / 0.795 | 9.75 / 9.73 | 20.2 / 20.3 |

The first two layers make the zig-zag slightly worse than the champion's (long links replaced by short ones); the position layer more than
repays it.

## Prediction
VAL, `layers` against `champ`: topo F1 +0.040; junction F1 +0.07 (from about 0.25); heading error about 1.2 degrees lower; F1@0.5 +0.01;
lanes found within 0.007 of the champion (guard 0.010: a pass or fail within 0.003 of it is an edge result).

## How the verdict is read
- Validity first: the quality numbers of `champ` must repeat the champion's own (E396.champ: topo F1 0.668172 / 0.681281), because its
  outputs are unchanged and the scorer's topo F1 is untouched; if they differ the experiment says nothing.
- The standard rule on topo F1 with the problem's guards. A KEEP supports the sum of the layers only: most of its topo F1 is E420's gain
  measured again (+0.039 there), whose skip layer was shaped after a VAL guard failure of E399 (a selection on VAL). It says nothing about the
  fork gate or the position layer alone: those are E423 and E424.
- Time and memory from the runs' records; a time counts only when both runs have `cost.quiet` = 1, otherwise it is reported as not measured.
  An adoption is not recommended when the layers add more than 3 ms per frame (quiet) or more than 300 MB of peak memory.
- KEEP: a candidate; the owner decides an adoption.

## Design delta
R1: four fixed soft layers in the forward, not learned (whether that satisfies R1 or the steps must be learned is the owner's call at an
adoption; E425 trains the head with them in its forward); the skip layer's top-8 search is a hard selection inside the forward. R3: time and
memory are now measured by every run. R7: the read-out's thresholds are also inside the forward (skip, gap, fork and position layers).
DESIGN.md changes only at an adoption.

## Imported networks
`<runs root>/lane_center/e424_layers_s{0,1}`: the champion's `e372_s4_s{0,1}/model.pt` with the four settings added to its args, weights and
B' tensors equal (checked); `import.json` in each folder records both sha256.

## Critic (cards mode, 2026-11-28): revise
Taken in: the attribution of a KEEP, the heading prediction against the champion (measured now), the validity check, the time and memory
limits and quiet rule, the top-8 truncation, TRAIN tables from the network's own outputs.

## Conclusion
**Verdict** (`research judge`, 2026-11-28, `verdict.json`): `layers` **TIE**: topo F1 0.6748 -> 0.7135, +0.0388 [+0.0287, +0.0491] (seeds
+0.0380 / +0.0395; prediction +0.040), but the guard on lanes found broke (-0.0125 > 0.010). VAL 873 frames in 22 scenes; TEST not touched;
nothing adopted.

Validity: `champ` repeats E396.champ exactly (topo F1 0.668172 / 0.681281). Both arms of the table are timed on an otherwise idle GPU
(`cost.quiet` = 1), one run after the other.

| number (mean of seeds 0 / 1) | champion | layers |
|---|---|---|
| topo F1 | 0.6748 | 0.7135 |
| junction F1 (precision / recall) | 0.251 (0.156 / 0.643) | 0.321 (0.243 / 0.472) |
| heading error (deg) | 11.60 | 10.59 |
| F1@0.5 / F1@1.0 | 0.647 / 0.8467 | 0.651 / 0.8443 |
| lanes found | 0.7223 | 0.7098 |
| duplicates / vertices per 100 m | 0.064 / 23.96 | 0.013 / 19.32 |
| time per frame, median / p90 (ms) | 20.0 / 41 | 28.1 / 55 |
| peak memory (MB) / parameters (M) | 89 / 2.323 | 89 / 2.323 |

- Predictions: topo and junction met; the heading error fell by 1.0 degrees (predicted 1.2); lanes found fell twice as far as predicted.
- Not adoptable as it stands: the lanes-found guard and the time limit of this card (+8.1 ms against 3 ms) both fail.
- Junctions, seed 0, frames with at least four true lanes (825, scratch count): right 3027 -> 2235, false 15834 -> 6552, missed 1654 -> 2446:
  the layers remove most false forks and merges and a quarter of the right ones.
- Follows: the lanes lost belong to the fork gate (E423), the time to the position layer (E424); a second round with a milder gate and a
  cheaper position step (not run yet, no number).
