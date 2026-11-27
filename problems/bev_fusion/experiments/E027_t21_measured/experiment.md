# t21_ctl measured in the research system

```json card
{
 "id": "E27", "title": "t21_ctl measured in the research system",
 "question": "what does the champion candidate t21_ctl (the t20 network continued 1500 steps) give through the research system: its weights imported, the warm evaluation on ground truth v13, 1000 VAL frames spread over 82 scenes?",
 "hypothesis": "the code moved without a change; the probe of 2026-11-27 gave board7 0.4631 for seed 0 with the same evaluation",
 "kind": "measure", "seeds": [0, 1], "steps": ["import", "evaluate"],
 "arms": {"t21": {"set": {"_import": "runs/t21_ctl_s{seed}"}, "note": "the newest checkpoint of the run t21_ctl_s<seed> (ckpt_00001500.pt)"}}
}
```

## Conclusion
MEASURED (`verdict.json`, research judge): the warm evaluation on ground truth v13, 1000 VAL frames spread over the 82 scenes, gives
board7 0.4631 (seed 0) and 0.4635 (seed 1); drivable area IoU 0.850 / 0.846, object mAP 0.348 / 0.347. Seed 0 is the probe of the same
day to the fourth digit. The cold v8 reads of v1 (0.4549 / 0.4548) are another protocol and not comparable. Not registered: the
champion is the owner's choice (`DECISIONS.md`, Open).
