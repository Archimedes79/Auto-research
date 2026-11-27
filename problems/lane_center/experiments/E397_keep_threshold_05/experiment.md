# E397 the champion's decode at the keep threshold 0.5

```json card
{
 "id": "E397",
 "title": "the champion's decode at the keep threshold 0.5",
 "question": "does the champion lose topology when its decode keeps a node only at p_none < 0.5 instead of 0.6?",
 "hypothesis": "the keep threshold 0.6 was chosen on TRAIN (E366 / E371) because the nodes with p_none in [0.5, 0.6) are mostly real lane points the head is unsure of; dropping them cuts links and lowers topo F1",
 "prediction": -0.008,
 "kind": "compare",
 "control": "champ",
 "seeds": [0, 1],
 "arms": {
  "champ": {"from": "E396.champ", "note": "the champion's runs of E396 (its decode, keep threshold 0.6)"},
  "kt05": {"set": {"import_weights": "runs/e372_s4_s{seed}", "decode": {"keep_thr": 0.5}}, "note": "the same weights, keep threshold 0.5"}
 },
 "note": "the first comparison judged by the research system; it also tests 'from' (the control reuses stored runs) and the paired scene bootstrap of the pooled seeds"
}
```

## Conclusion
(after research judge)
