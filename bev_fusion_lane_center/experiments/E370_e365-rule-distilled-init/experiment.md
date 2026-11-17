# E370 - the E365 token head with a RULE-DISTILLED initialisation: stage A learns the one-line rule's links as teacher labels, stage B fine-tunes on the GT only; the delivered system has no rule

Card of 2026-11-17 (version 1, 02:15 system clock; written by the Spurmitte chat that Mirko addresses as "Fable" on the lane lead's order of 02:05: "the intent of E369 (start at the rule's floor, learning only adds) has a clean version inside the solution set: rule-distilled initialisation, as E325 imitated the nearest-piece rule; the rule never runs at inference"). **Status: draft, no critic pass yet, not built, nothing run.** E369 (the rule as a fixed prior input) is a DIAGNOSTIC only: an exact link list in the forward path is a wiring rule (D88 / D113, the lane lead's reading, put to Mirko). E370 is the champion-eligible version of the same idea. Design, cache (B' = e292_a, K = 512), architecture (the S1 head: d 64, 3 layers), the GT loss, the decode of record and the controls are those of E365 (experiments/E365_e2e-token-champion-s1, sections 0 - 1) unless written here.

**Why:** E365 S1 is worse than the rule `lo_near` on the same tokens (0.5486 / 0.5743 against 0.6265 / 0.6281): trained from a random start on the GT, the head settles at a conservative point (precision 0.86, recall 0.40). **The question:** does a head that is first taught to BE the rule (stage A) and then trained on the GT (stage B) end above the rule and over the okayish gate, with no rule anywhere in the delivered system?

**The design (two training stages: inside 'at most two' of G-simple):**
- **Stage A (teacher, `--teacher rule`, 3000 steps, from a random start):** the targets are the rule's, computed on the cached tokens exactly as `jtoken_eval.py ctl` computes c-rule (the B' tokens with e >= 0.5; `lo` and `lo_near`): wiring: for each class c and source i, G_teacher(i) = the rule's link of i in class c (a single positive) or empty (target: the none slot), the same multi-positive likelihood as E365; existence / type: a token is 'kept' by the teacher when B' e >= 0 (the rule's nodes), its type target = the GT type of its assigned GT node if assigned, else 1 (lane point); a token with e < 0 has type target 0 (none); the type CE as E365 (weight 0.5, both readings); no scale loss (the rule has no handle scales: the scale head trains in stage B). The rule only produces LABELS during stage A; it is not an input and never runs at inference.
- **Stage B (GT, the E365 S1 loss unchanged, 12000 steps, `--init <stage-A model.pt>`):** a fresh optimiser and schedule (5 % warm-up, cosine), the teacher loss OFF in the arm of record.
- **The stage-A FIXTURE (written now):** at the end of stage A the head's delivered graph on the TRAIN sample (every third of the 873 evenly spaced TRAIN frames: 291 frames) is within 0.01 topo F1 of c-rule's graph on the SAME frames (c-rule on TRAIN is computed by `ctl --split train --every 3`, a task of the build), both seeds; if not, stage A continues ONCE to 6000 steps (named in the result); if it still misses, stage B starts anyway and the result says "stage A did not reach the rule" (a diagnosis, not a kill).
- **Delivered system:** the trained head of stage B alone (no rule input, no prior, no teacher): G-simple holds as for E365 (one network, two training stages, the hard top-K the named budget).

## 0. The card (what the tools run)
```json card
{
 "id": "E370",
 "card": "",
 "title": "E365 token head, rule-distilled initialisation: stage A 3000 steps on the c-rule links as teacher labels (the rule never runs at inference), stage B 12000 steps on the GT loss of E365 from the stage-A weights, two seeds, against c-rule, the E365 S1 heads, the best E367 arm and plain B, the okayish gate of E365",
 "kind": "variant",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.08, "vs": "ref"},
 "arms": {
  "ref": {"recipe": "E365 S1 heads at the decode of record: VAL topo F1 0.5486 / 0.5743 (runs/e365_s1_s<S>/token_card_rec.json); no run_ids for decide.py: the verdict is by hand"},
  "rd": {"recipe": "stage A `jtoken_train.py --name e370_a_s<S> --seed S --ident e292_a --teacher rule --steps 3000 --bs 8 --lr 1e-3 --workers 4`, stage B `jtoken_train.py --name e370_b_s<S> --seed S --ident e292_a --init runs/e370_a_s<S>/model.pt --steps 12000 --bs 8 --lr 1e-3 --workers 4` (the teacher and init options are added by a coder)", "run_ids": ["E370.a_s0", "E370.a_s1", "E370.b_s0", "E370.b_s1"]}
 },
 "control": "ref",
 "variant": "rd",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "ref", "min_effect": 0.01}
 ],
 "budget_gpu_h": 2.0,
 "kill": "a loss is not finite; or no learning in a stage (the mean loss over steps 1801 - 2000 not below 0.8 x the mean over steps 101 - 200; for stage B read against its own start); the stage-A fixture is a diagnosis, never a kill; a rule-step number is never a kill reason (D113 point 2). A run cut by run.py or the GPU is resumed from its 500-step checkpoint by the supervisor (resume steps reported); `--timeout` from 0.65 s per step at the start, then from the speed measured over 1500 steps",
 "evaluate": {"split": "val"}
}
```
**The verdict (by hand, written before any run; `decide.py` is not used):** the **okayish gate of D113 / E365, unchanged**, on the stage-B head at the decode of record: G-topo >= 0.6359 / 0.6349; G-found: F1@1.0 >= 0.8182 / 0.8060, lanes found >= 0.6066 / 0.5937; G-cost (the head = the S1 head: 0.127 M; the whole path as E365, measured on a quiet host); G-simple (two training stages, no rule in the delivered system); PASS / MISS / QUALITY-ONLY / INCONCLUSIVE as E365; a PASS is then confirmed on two fresh head seeds (E367's rule) before "a champion exists" is reported. **Reported next to it (labels, not the verdict):** ABOVE-RULE = stage-B topo F1 >= c-rule + 0.01 on both seeds (0.6365 / 0.6381; c-rule is a reference here, never a bar or a kill: D113 point 2); B-ADDS = stage B >= stage A + 0.01 on both seeds (does the GT fine-tune keep the rule's recall and add); the paired scene-bootstrap interval against plain B and against S1 per seed; the decode selected per head on its TRAIN sample (E366 / E367 rule) as a diagnosis; c-hi (0.819 / 0.820), E324 s.
**Pre-registered secondary numbers:** the stage-A fixture (TRAIN-sample topo F1 against c-rule on the same frames); the TRAIN-sample topo F1 at stage-B steps 0, 3000, 6000, 12000 (does the GT loss drag the head back to the conservative point of S1?); precision and recall of the delivered graph (S1: 0.86 / 0.40); the long-class delivered share (S1: 51 %); the kept share of the assigned tokens (S1: 0.868).
**Prediction (guesses, with windows):** stage A reaches the rule on TRAIN within 0.01 (70 %); stage-B VAL topo F1 0.635 (0.58 - 0.69) mean over seeds, F1@1.0 0.80 (0.76 - 0.84), lanes found 0.62 (0.56 - 0.68); ABOVE-RULE about 45 %; B-ADDS about 50 %; the okayish gate: PASS about 25 % (F1@1.0 is the harder half: c-rule 0.773, floors 0.818 / 0.806), MISS 45 %, INCONCLUSIVE / QUALITY-ONLY 30 %. **The risk, said now:** stage B uses the same GT loss that made S1 conservative; a warm start at the rule may only delay the drift (the step 3000 / 6000 / 12000 TRAIN numbers show it); if it drifts, the next card keeps a decaying teacher term in stage B (a second arm, not this card).
**Confound, said now:** stage A adds 3000 steps (15000 in all against S1's 12000); E367 c_long (24000 steps, one seed) is the schedule control.

## 1. Card text
- **Statement:** a rule-distilled start lets the GT training improve on the rule instead of falling below it. **Mechanism:** the rule's operating point (every node one link, chains whole) is the start; the GT loss then corrects false links and adds what the rule misses. **Not claimed:** anything about S2 / S3 / S4 / TEST; that 3000 teacher steps are the right budget (the fixture says whether the rule was reached). **Cost:** stage A about 10 min, stage B about 40 min per pair (0.2 s per step); the delivered head as S1 (0.127 M parameters).
- **Referee:** a short fresh `ar-critic` (cards) before the build; the code critic after it.

## 2. Detail plan
| step | what exactly | where | done when |
|---|---|---|---|
| critic | fresh ar-critic on this card, v2 | Fable | committed |
| build | `--teacher rule` (stage-A targets from the c-rule code of `jtoken_eval.ctl`, the existence / type targets above, no scale loss) and `--init <model.pt>` (weights only, fresh optimiser and schedule, recipe records the init path and its sha256) in jtoken.py / jtoken_train.py; `ctl --split train --every 3` for c-rule on the TRAIN sample; CPU tests with mutants (teacher targets equal the rule's links on hand frames and 3 cache frames; no GT key read in stage A except the type labels named above; init loads the weights exactly) | coder (after the E369 build: the same files) | tests green, code critic |
| stage A | a_s0, a_s1 (one pair), then the fixture on TRAIN | `run.py --id E370.a_s<S> --gpu --require-card` | fixture numbers here |
| stage B | b_s0, b_s1 from the stage-A weights, the supervisor of E367 | `run.py --id E370.b_s<S> --gpu --require-card` | done flags |
| read-out | dump, score (record + the TRAIN-selected decode), TRAIN sample at 3000 / 6000 / 12000, degraded | `run.py --id E370.read_s<S>` | cards |
| decide | the verdict above | by hand | result section, ledger notes, commit, one line to the lead |
GPU order (the lane lead, 02:05): b25 pair, E370 (A + B, both seeds), E367 a, E369, c_long; the lane chat's S2 runs (E368) before 05:00 in a free gap.

## 3. Summary
(empty until the runs exist)
