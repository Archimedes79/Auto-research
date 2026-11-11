# E300 Where does the pair head fail? Splits of the end and pair recall by driven / city / lanes / turn / painted / visible / range (D82: "what might be overlooked")

Card of 2026-11-11, CPU analysis on the dump of a head (`runs/<run>/ends_val.pkl`), `evaluation/jsplits.py`. One number per head hides where it fails; the splits show what the next experiment must attack and where the guards of the experiments should look.

## 0. The card (what the tools run)
```json card
{
 "id": "E300",
 "card": "",
 "title": "recall of the true pairs and of the GT ends of a pair head by the attributes of the ground truth: pair driven by the ego or not, city (Boston / Singapore), lanes of the entry arm, turn class, end painted or not, visible or hidden, distance from the ego",
 "kind": "variant",
 "deterministic": true,
 "metric": "pair_recall",
 "direction": "higher",
 "band": 0.03,
 "prediction": {"value": 0.0, "vs": "c"},
 "arms": {
  "c": {"recipe": "the head of record on the E284 dump (h2, seeds 0 and 1), all pairs"},
  "s": {"recipe": "the same recall inside each split; the largest differences between the splits are the findings (no verdict)"}
 },
 "control": "c",
 "variant": "s",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [{"metric": "pair_recall", "vs": "c", "min_effect": 0.0}],
 "budget_gpu_h": 0.0,
 "kill": "the splits do not add up to the total (the count of GT pairs differs from `jgt`'s 11083)",
 "evaluate": {"split": "val"}
}
```
By hand (an analysis, no KEEP / CUT; label "by hand: analysis"): the table of the result section is the finding; it is repeated for E287 when its dump exists and for the better head of the later experiments. Command: `jsplits.py score --run e284_h2_s0 e284_h2_s1`.
