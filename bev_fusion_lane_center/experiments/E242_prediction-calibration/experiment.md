# E242 Prediction calibration: tolerance, reference class, optimism

The card (read by `experiment.py` and `decide.py`; schema `autoresearch/schemas/experiment.schema.json`):

```json card
{
 "id": "E242",
 "card": "H19.S3",
 "title": "prediction calibration: a prediction that names value, tolerance and reference class is hit at least 70 % of the time (loop 18 prose predictions: 7 of 12)",
 "kind": "variant",
 "metric": "pred_mean_abs_err",
 "direction": "lower",
 "band": 0.005,
 "prediction": {
  "value": -0.006,
  "vs": "c"
 },
 "arms": {
  "a": {
   "recipe": "predictions of cycle 19 and the first loop after it, written as {value, tol, class} where tol = max(2 x measured seed SD of E240, 0.005) and value is the median of the reference class (below) shrunk by half towards zero unless the card names a structural mechanism; scored by experiments/E242_prediction-calibration/rescore.py (CPU) from decision.json files; needs at least 8 numeric predictions",
   "note": "scored forward; the retrospective part (arm c, 5 numeric effect predictions with decision.json + 7 prose hits of loop 18) is computed now"
  },
  "c": {
   "recipe": "loop 18 as recorded: 5 numeric effect predictions in decision.json files (E215 +0.140, E219 +0.010 / +0.003 / +0.008, E232 +0.030), mean absolute error 0.0145 in metrics.py (it is the mean over exactly these five; the 7 of 12 hit count is a separate prose count)",
   "note": "reference value 0.0145 from `metrics.py`; the signed errors are +0.0044, -0.0023, -0.0071, -0.0043, -0.0543 (decision.json files)"
  }
 },
 "control": "c",
 "variant": "a",
 "rule": [
  {
   "metric": "pred_mean_abs_err",
   "min_effect": 0.005,
   "note": "cycle-19+ mean absolute error at least 0.005 below 0.0145, i.e. <= 0.0095"
  },
  {
   "metric": "pred_hit_rate",
   "min_effect": 0.1,
   "note": "hit rate inside the written tolerance at least 0.10 above the prose rate 7 / 12 = 0.58, i.e. >= 0.68; the metric is counted by rescore.py, not by the words 'met' / 'missed'"
  }
 ],
 "budget_gpu_h": 0,
 "kill": "if fewer than 8 numeric predictions exist after two loops, report the retrospective table only and stop; if the written tolerances are all copied from the band (tol = band) the experiment is void",
 "evaluate": {
  "split": "ledger",
  "compare": "c"
 }
}
```


## 0. What the tools run (the fields of `experiment.json`)
| field | value |
|---|---|
| id / card / title | E242 / H19.S3 / a prediction with value, tolerance and reference class is hit at least 70 % of the time (loop 18: 7 of 12 by prose, mean error 0.0145 over five numeric) |
| kind | variant (a rule of the card, no code beyond a 40-line CPU scoring script rescore.py (to be written in this folder) in this folder) |
| deterministic | not applicable (CPU, ledger and decision files) |
| metric, direction, band | pred_mean_abs_err, lower, 0.005 |
| prediction | -0.006 against arm c (cycle 19+ mean absolute error 0.0085 against the loop-18 value 0.0145) |
| arms | a: predictions written with {value, tol, class} from cycle 19 on; c: loop 18 as recorded |
| control | c |
| rule | pred_mean_abs_err min_effect 0.005 (<= 0.0095); pred_hit_rate min_effect 0.10 (>= 0.68) |
| budget_gpu_h, kill | 0; void with fewer than 8 numeric predictions after two loops |
| evaluate | ledger (decision.json files, LEDGER.csv) |

## 1. Card
| | |
|---|---|
| lens | system review (time per decision, prediction track record) |
| group | A9 evaluation, data, metrics |
| sources | `metrics.py -v` (loop 18: 7 / 12 hits, mean error 0.0145), decision.json of E215 / E219 / E232, LEDGER.csv, E240 |
| status | proposed |

- **Statement.** A written prediction that carries a tolerance (at least twice the measured seed SD of E240, never below 0.005) and starts from the median of its reference class instead of the hoped-for effect is hit in at least 70 % of the first 8+ numeric predictions of cycle 19 and the next loop, and its mean absolute error is at most 0.0095, against 7 of 12 hits and 0.0145 in loop 18.
- **Mechanism (what the records say).** (i) The hit count is read from free text: `metrics.py` takes "met" / "missed" from the result prose. Two of the five "missed" rows are mixed results (E230.rc: "Q above the control and J above 0.38 met; lanes found >= 0.74 missed"; E219.j1eq: "lanes found <= 0.69 met, Q <= j1 + 0.005 missed by 0.0015"): a prediction with two numbers is scored as a miss when either misses, and a miss by 0.0015 weighs like one by 0.05. (ii) Only 12 of 35 loop-18 runs carry a prediction at all; the numeric ones (decision.json) are five, with signed errors +0.0044 (E215), -0.0023, -0.0071, -0.0043 (E219 x3) and -0.0543 (E232): four of five are optimistic (measured below predicted), mean signed error -0.0127; the mean absolute error 0.0145 of `metrics.py` is exactly the mean over these five and is dominated by one row (E232, the third joint end-to-end run, predicted +0.03 although the first two had been negative). (iii) The fine-tune variants of E219 / E240 all lie in a narrow class: |Q effect| of 0.0077 (j1rc - j1eq), 0.0037 (rc1 - f1eq), 0.0041 (j1eq8 - j1eq), 0.006 (j1eq - j1), 0.002 (j1eqs1 - j1): median about 0.004, none above 0.008. A Q prediction of +0.010 for such an experiment (E219.j1rc) is outside the class without a stated reason. (iv) A tolerance of the order of the seed SD (E240: expected 0.003) is the honest scale of what can be called a hit.
- **Prediction.** Retrospective (CPU, now, arm c, from the five decision.json files): scored with tol = 0.005 the numeric predictions give 3 hits of 5 (E215 +0.0044, E219 -0.0023, -0.0043; E219 -0.0071 and E232 miss); with tol = 0.010 they give 4 of 5. Prose hits and numeric hits therefore differ in at least 2 of the 12 predictions (E230.rc and E219.j1eq are the two to check). Forward (arm a): over the first 8 numeric predictions of cycle 19 and the loop after it, hit rate 0.70 (0.55 - 0.85) and mean absolute error 0.0085 (0.006 - 0.012). If the first loop only produces 4 predictions, report and wait one more loop.
- **Decision rule.** KEEP (write into `../../../autoresearch/templates/experiment.md` and `card.schema.json` that `prediction` is {value, tol, class} and that `decide.py` prints hit / miss against the tolerance, a DECISIONS entry because the template and tools are protected) if both clauses hold on at least 8 predictions. CUT if mean absolute error > 0.0145 (the rule makes it worse) or the hit rate < 0.58. TIE in between: keep writing tolerances (costs nothing) but do not change the schema. The result is not a significance test: n = 8 predictions give a hit-rate interval of about +-0.3, so the rule is a bookkeeping check, and the expected benefit is the reference-class discipline (a card that predicts above the class maximum must name its mechanism), not the number.
- **Test design.** Arm c: loop 18 as recorded (frozen, table from decision.json and LEDGER.csv; no re-run). Arm a: every numeric prediction on a card of cycle 19 and later is written with its class and tolerance before the run (the card is committed before the run: git history proves the order). Null: predict the class median (a constant 0.004 on every Q effect); an arm a that does not beat the constant has no skill and the rule is then "predict the median". Matched reference: the same experiments scored with the old prose rule. Seed twin: not applicable. Equal budget: no GPU.
- **Cost.** 0 GPU h, 0.2 coding days (rescore.py; the loop table of `metrics.py` gets two columns, hit by tolerance and signed error, a protected-tools change via DECISIONS), 0.1 day to write the reference-class table. Risk: tolerances written wide to guarantee a hit (kill criterion below).
- **Kill criterion.** See the table; also void if the tolerance of a prediction is edited after the run (git history of the card).
- **Inputs.** Ready: decision.json of E215 / E219 / E232, `metrics.py -v`, LEDGER.csv. To build: rescore.py; the E240 seed SD (the tolerance rule uses it; until then tol = 0.005 for Q, 0.01 for lanes found).
- **Referee** (2026-10-25, ar-referee): REJECT as an experiment; MERGE as a process rule, cost nothing, start tonight. Reasons: (1) it is not a test of anything in the product, and arm a versus arm c compares different experiments, so luck, regression to the mean and the choice of tolerance cannot be separated; n = 8 gives a hit-rate interval of +-0.3, as the card itself says; (2) the hit rate is set by the tolerance (tol = max(2 x SD, 0.005)), so "hit rate >= 0.70" measures how wide the tolerance is, not skill; the card's own null (predict the class median) is in the test design but not in the rule; (3) the mean absolute error mixes scales (E215 is on J with 0.14, E219 on Q, E232 on route success), and the single -0.0543 row dominates the 0.0145; (4) the retrospective 3 / 5 and 4 / 5 hit counts and the mean 0.01448 recompute correctly from the five signed errors, which is a 30 minute CPU job and can go into the E240 summary. Do instead: from now on every card writes prediction as {value, tol, class}, with value near the class median (Q effects of a fine-tune: median 0.004, max 0.008 in E219) and an explicit mechanism if it is above; E244 and E245 predict +0.012, which is outside that class. Revisit a template or tool change (protected, a DECISIONS entry) only after 8 scored predictions exist.
