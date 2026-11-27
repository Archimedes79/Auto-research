# E396 the champion measured in the research system

```json card
{
 "id": "E396",
 "title": "the champion measured in the research system",
 "question": "does the champion E372 S4 give its numbers through the research system (the code moved, the data on D:, the dump as a step, the scorer behind the Evaluation adapter)?",
 "hypothesis": "the code moved without a change and the dump runs at the batch of its GPU parity, so the numbers are those of its v1 cards (topo F1 0.6682 / 0.6813)",
 "kind": "measure",
 "seeds": [0, 1],
 "arms": {"champ": {"set": {"import_weights": "runs/e372_s4_s{seed}"}, "note": "the weights of E372 S4, the decode of D115 (the base config)"}}
}
```

## Conclusion
MEASURED (`verdict.json`, research judge): VAL 873 frames, topo F1 0.668172 (seed 0) and 0.681281 (seed 1), junction F1 0.2538 / 0.2481,
Q 0.6796 / 0.6808, vert100_dp 23.81 / 24.12: the numbers of the v1 cards to the last digit. The research system reproduces the champion
(code moved, data on D:, dump as the step `predict`, scorer behind the Evaluation adapter). Registered as the champion under D120
(`champions/lane_center/`). The `cost` in `result.json` is the read-out of the stored dump, not the network's time per frame (16.21 ms, v1).
