# E426 the champion measured again after the package step

The card is the json block below: `research check <problem> E426` checks it, `research start` freezes it in the snapshot. What the runs do,
how the verdict is made and what counts is the card as the snapshot holds it; a changed experiment is a new experiment.

```json card
{
 "id": "E426",
 "title": "the champion measured again after the package step",
 "question": "on VAL with the evaluator of the package step (bc75a1d1, 7201d2ebe42a): what does the champion deliver in quality, time and memory, and are its graph numbers those of E422.champ (evaluator 0a7eaf74e9e4)?",
 "hypothesis": "the package step moved code and changed no computation: per seed, every graph number and every per-scene score equals E422.champ to the last stored digit (topo F1 0.668172 / 0.681281), on the same checkpoint and ground truth, with 2.323143 M parameters and 88.9091796875 MB peak memory",
 "kind": "measure",
 "seeds": [0, 1],
 "arms": {
  "champ": {"set": {"import_weights": "runs/e372_s4_s{seed}"}, "note": "the champion's weights (E372 S4), predicted, timed and scored on the package code"}
 },
 "note": "D165 (the owner, 2026-11-28): after a change of the evaluator, measure again instead of keeping old results compatible. The runs are started one after the other so that the times are measured on an otherwise idle GPU. No adoption, no registration without the owner"
}
```

## Why
The package step (`bc75a1d1`: every import `lane_center....`, steps started as modules) changed hashed evaluator files. The judge refuses a
control reused from a run scored by another evaluator, and the ledger marks the earlier runs as not comparable as soon as a newer run is
recorded. Until the champion is measured on this code, a new experiment has no champion run to compare with.

## Plan
- No code.
- Before the start: `research anchor lane_center E422.champ_s0` and `..._s1` (the stored predictions of E422 scored by today's evaluator).
  Done 2026-11-28: both agree at tolerance 0.0.
- Runs: predict and evaluate. One seed after the other, so that no two runs share the GPU: `research start lane_center E426 --arms champ
  --seeds 0`, after its end the same with `--seeds 1`. TEST is not touched.

## What would refute the hypothesis
Per seed against E422.champ: any graph number or per-scene score differing at all (the anchor tolerance of `problem.json` is 0), another
`identity.checkpoint_hash` (e9c749d6b3879794 / 56c5c8d951bbdef4) or `gt_signature` (5a92e34113f3), another parameter count or peak memory.
Then the package step changed the prediction or the scorer, and the difference has to be found before anything is compared.

The time is measured and reported, it refutes nothing: E422's two seeds are the same computation and differ by 1.8 ms in the median. A time
with `cost.quiet` = 0 is not compared. A later experiment with a time limit measures its control again next to its candidate.

## Design delta
None.

## Conclusion
After `research judge`: the numbers with split and denominator, the comparison with E422.champ, what follows.
