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
(after research judge)
