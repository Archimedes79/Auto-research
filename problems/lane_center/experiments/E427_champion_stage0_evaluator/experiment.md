# E427 the champion measured again after the stage-0 step

The card is the json block below: `research check <problem> E427` checks it, `research start` freezes it in the snapshot. What the runs do,
how the verdict is made and what counts is the card as the snapshot holds it; a changed experiment is a new experiment.

```json card
{
 "id": "E427",
 "title": "the champion measured again after the stage-0 step",
 "question": "on VAL with the evaluator of the stage-0 step (6caf3b1d, d2dfebb49c91): what does the champion deliver in quality, time and memory, and are its graph numbers those of E426.champ (evaluator 7201d2ebe42a)?",
 "hypothesis": "the stage-0 step changed how the evaluation finds the token cache of a run with its own B' and nothing for the champion's ident e292_a: per seed, every graph number and every per-scene score equals E426.champ to the last stored digit (topo F1 0.668172 / 0.681281), on the same checkpoint and ground truth",
 "kind": "measure",
 "seeds": [0, 1],
 "arms": {
  "champ": {"set": {"import_weights": "runs/e372_s4_s{seed}"}, "note": "the champion's weights (E372 S4), predicted, timed and scored on the code of the stage-0 step"}
 },
 "note": "D165 (the owner, 2026-11-28): after a change of the evaluator, measure again instead of keeping old results compatible. The runs are started one after the other so that the times are measured on an otherwise idle GPU. No adoption, no registration without the owner"
}
```

## Why
The stage-0 step (`6caf3b1d`: the full stage 0 of v1 back, a run's own token cache) changed three hashed evaluator files
(`net/prep/lane_ptok_data.py`, `evaluation/lane_dump.py`, `net/prep/lane_tokcache.py`). The judge refuses a control scored by another
evaluator; until the champion is measured on this code, a new experiment has no champion run to compare with.

## Plan
- No code.
- Before the start: `research anchor lane_center E426.champ_s0` and `..._s1` (the stored predictions of E426 scored by today's evaluator).
  Done 2026-11-28 on the working tree of the step: both agree at tolerance 0.0.
- Runs: predict and evaluate, one seed after the other so that no two runs share the GPU: `research start lane_center E427 --arms champ
  --seeds 0`, after its end the same with `--seeds 1`. TEST is not touched.

## What would refute the hypothesis
Per seed against E426.champ: any graph number or per-scene score differing at all (the anchor tolerance of `problem.json` is 0), another
checkpoint hash or ground-truth signature, another parameter count or peak memory. Then the step changed the prediction or the scorer.
The time is measured and reported; it refutes nothing (a time with `cost.quiet` = 0 is not compared).

## Conclusion
MEASURED (`dc3c014b`), VAL, 873 frames in 22 scenes, evaluator d2dfebb49c91. The hypothesis holds: per seed all 18 graph numbers and all
22 per-scene scores equal E426.champ exactly (topo F1 0.668172 / 0.681281), on the same checkpoints (e9c749d6b3879794 / 56c5c8d951bbdef4)
and ground truth (5a92e34113f3), with 2.323143 M parameters and 88.91 MB peak memory. Time (reported, not compared): median 21.22 / 18.60 ms
against E426's 18.97 / 18.64 ms, both quiet; seed 0's 2.3 ms more is inside the spread E426's card names for two runs of one computation.
E427.champ is the champion run a new lane experiment compares with on this evaluator.
