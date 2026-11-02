# E215 fork pruning by junction-head heat closes the J gap of the champion (re-encoded from the loop-18 record for the P1 check)

The card (read by `experiment.py` and `decide.py`; schema `autoresearch/schemas/experiment.schema.json`):

```json card
{
 "id": "E215",
 "card": "",
 "title": "fork pruning by junction-head heat closes the J gap of the champion (re-encoded from the loop-18 record for the P1 check)",
 "kind": "variant",
 "deterministic": true,
 "metric": "J",
 "direction": "higher",
 "band": 0.02,
 "prediction": {
  "value": 0.14,
  "vs": "c"
 },
 "arms": {
  "a": {
   "cards": [
    "runs/eval/champion/val_e215.json"
   ],
   "note": "champion with fork_prune 0.3 (evaluate.py champion --compare champion_noprune)"
  },
  "c": {
   "names": [
    "champion_noprune"
   ],
   "note": "the same checkpoint and assembly without pruning"
  }
 },
 "control": "c",
 "rule": [
  {
   "metric": "J",
   "min_effect": 0.1,
   "note": "J up to the control's level"
  },
  {
   "metric": "F1@1.0",
   "max_loss": 0.005,
   "note": "at the same F1@1"
  },
  {
   "metric": "lanes_found",
   "max_loss": 0.005
  }
 ],
 "budget_gpu_h": 0,
 "kill": "",
 "evaluate": {
  "split": "val",
  "compare": "c"
 },
 "recorded_decision": "KEEP (loop 18: J 0.265 -> 0.409, F1@1 and lanes found unchanged, Q 0.701 -> 0.735)"
}
```


(experiment.md made from experiment.json by `experiment.py migrate --create-md`; the card follows, the story of the experiment is in the ledger and the summaries)
