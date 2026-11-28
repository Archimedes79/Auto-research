# E421 what the gap-closing layer adds to the skip-link layer

The card is the json block below: `research check <problem> E421` checks it, `research start` freezes it in the snapshot. What the runs do,
how the verdict is made and what counts is the card as the snapshot holds it; a changed experiment is a new experiment.

```json card
{
 "id": "E421",
 "title": "what the gap-closing layer adds to the soft skip-link layer",
 "question": "on VAL, with the champion's weights: does the gap-closing layer on top of the skip layer (E420.skipgap) gain topo F1 against the skip layer alone (E399.skip) without breaking a guard?",
 "hypothesis": "the gap layer closes gaps between open ends, so it adds topo F1 and lanes found to the skip layer at an unchanged junction F1",
 "prediction": 0.017,
 "kind": "compare",
 "control": "skip",
 "seeds": [0, 1],
 "arms": {
  "skip": {"from": "E399.skip", "note": "E399's runs with the skip layer alone"},
  "skipgap": {"from": "E420.skipgap", "note": "E420's runs with the skip layer and the gap-closing layer"}
 },
 "rule": {},
 "kill": [],
 "note": "companion of E420 (the critic's third point): no run of its own; written before E420's results were read; no adoption from this card"
}
```

## Why
E420 judges `skipgap` against the champion only, and the skip layer alone already gains topo F1, so E420 cannot show what the third layer adds.
TRAIN (291 frames, seeds 0 / 1): +0.020 / +0.015 topo F1 and +0.017 / +0.015 lanes found against the skip layer alone.

## Conclusion
**Verdict** (`research judge`, 2026-11-28, `verdict.json`): `skipgap` KEEP against the skip layer alone. VAL (873 frames, 22 scenes),
pooled over seeds 0 / 1: topo F1 0.6932 -> 0.7139, +0.0207 [+0.0179, +0.0239] (prediction 0.017, met); lanes found 0.7058 -> 0.7275,
F1@1 0.8396 -> 0.8462, junction F1 0.2882 -> 0.2881. No run of its own; the card was committed before E420's results were read. Nothing is
adopted from this card.
