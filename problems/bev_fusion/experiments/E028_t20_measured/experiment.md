# t20_base measured in the research system

```json card
{
 "id": "E28", "title": "t20_base measured in the research system",
 "question": "what does the D48 reference t20_base (one seed) give through the research system, with the same evaluation as E27?",
 "hypothesis": "t20_base lies below t21_ctl by less than the band 0.02, as in the cold v8 reads of v1 (0.4465 against 0.4549)",
 "kind": "measure", "seeds": [0], "steps": ["import", "evaluate"],
 "arms": {"t20": {"set": {"_import": "runs/t20_base_s0"}, "note": "the newest checkpoint of the run t20_base_s0"}}
}
```

## Conclusion
MEASURED (`verdict.json`, research judge), the evaluation of E27: board7 0.4569 (one seed), drivable area IoU 0.845, object mAP 0.345.
t21_ctl lies 0.006 above it, inside the band 0.02 of the score, as in v1 (cold v8: 0.4465 against 0.4549). Not registered: the
choice between t21_ctl and t20_base is the owner's (`DECISIONS.md`, Open).
