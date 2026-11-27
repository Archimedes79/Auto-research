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
TIE (`verdict.json`, research judge, the standard rule of the card, band 0.01): keep threshold 0.5 against the champion's 0.6, VAL 873 frames,
22 scenes, two seeds pooled: topo F1 0.6721 against 0.6748, gain -0.0026, paired scene bootstrap [-0.0039, -0.0017] (seed 0 -0.0034,
seed 1 -0.0019). The interval excludes 0 but lies inside the band: no change worth taking. The guard lanes_found breaks (-0.0123 > 0.01);
junction F1 rises (0.2509 -> 0.2649) and vert100_dp falls (23.96 -> 22.88). Prediction -0.008: sign right, size too large. The champion's
decode stays. First comparison judged by the research system: the control reused E396's runs (`from`), the judge checked comparability,
the seeds, the guards and the 22-scene warning.
