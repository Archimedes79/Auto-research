# E219 recall weight 3 on the fast j1 (j1rc) against its equal-budget control j1eq (re-encoded from the loop-18 record for the P1 check)

The card (read by `experiment.py` and `decide.py`; schema `autoresearch/schemas/experiment.schema.json`):

```json card
{
 "id": "E219",
 "card": "",
 "title": "recall weight 3 on the fast j1 (j1rc) against its equal-budget control j1eq (re-encoded from the loop-18 record for the P1 check)",
 "kind": "variant",
 "metric": "Q",
 "direction": "higher",
 "band": 0.01,
 "prediction": {
  "value": 0.01,
  "vs": "c"
 },
 "arms": {
  "a": {
   "cards": [
    "runs/e18/rc_eval/j1rc.json",
    "runs/e18/rc_eval/j1rcs1.json"
   ],
   "run_ids": [
    "E219.j1rc",
    "E219.j1rcs1"
   ],
   "note": "weight 3, seeds 0 / 1, pruning 0.4"
  },
  "c": {
   "cards": [
    "runs/e18/rc_eval/j1eq.json",
    "runs/e18/rc_eval/j1eqs1.json"
   ],
   "run_ids": [
    "E219.j1eq",
    "E219.j1eqs1"
   ],
   "note": "weight 1, same budget, seeds 0 / 1"
  }
 },
 "control": "c",
 "pairing": "position",
 "rule": [
  {
   "metric": "Q",
   "min_effect": 0.005
  },
  {
   "metric": "lanes_found",
   "min_effect": 0.04
  }
 ],
 "budget_gpu_h": 2,
 "kill": "",
 "evaluate": {
  "split": "val",
  "compare": "c"
 },
 "recorded_decision": "the weight buys lanes found (+0.062) and J, not Q (Q +0.004 / +0.012, inside the band); j1rc adopted on VAL against j1, j1eq named the safer fast variant"
}
```


(experiment.md made from experiment.json by `experiment.py migrate --create-md`; the card follows, the story of the experiment is in the ledger and the summaries)
