# E000 <title>

The card is the json block below: `research check <problem> E000` checks it, `research start` freezes it in the snapshot. What the runs do,
how the verdict is made and what counts is the card as the snapshot holds it; a changed experiment is a new experiment.

```json card
{
 "id": "E000",
 "title": "<title>",
 "question": "the one question this experiment answers",
 "hypothesis": "what you expect and why (the mechanism)",
 "prediction": 0.0,
 "control": "c",
 "arms": {
  "c": {"set": {}, "note": "the control: today's champion settings"},
  "a": {"set": {}, "note": "the change, as settings of the base config (a new option enters the code default-off first)"}
 },
 "seeds": [0, 1],
 "rule": {},
 "kill": []
}
```
Keys: `prediction` = the expected gain of the score over the control; `rule` may set `min_effect` (default: the problem's band),
`max_seed_spread` (the control's seeds further apart make the verdict INVALID), `boot` (bootstrap draws) or `script` (a rule script
in this folder, judge.py, when the standard rule cannot say it); `kill` = `[{"metric": "total", "after_step": 400, "above": 0.7}]`
on the training's log.jsonl; an arm can reuse stored runs (`"from": "E12.a"`) or name the run it builds on (`"needs": {"<setting>": "E12.a"}`);
`"kind": "measure"` measures arms without a control and without a verdict; `steps` and `split` override the problem's defaults.

## Design delta
The rules of DESIGN.md this experiment attacks or changes, with the changed text (the coder gets DESIGN.md plus this).

## Plan
What the coder changes (files, switches, tests), what is checked before the start, the budget.

## Conclusion
After `research judge`: the verdict, the numbers with split and denominator, what was learned, what follows.
