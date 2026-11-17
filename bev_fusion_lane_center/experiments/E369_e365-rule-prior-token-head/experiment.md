# E369 - the E365 token head that STARTS at the one-line rule: the rule's links and the B' existence as a fixed prior on its logits, the learned part zero-initialised

Card of 2026-11-17, **version 2 at 02:15 (system clock) after the first critic pass** (fresh ar-critic, cards: REVISE, 11 findings; a probe on 60 VAL frames showed that the v1 prior (soft distance terms with small penalties) reproduces only 86 % of the rule's links and 0 of its long links: the design is changed, not tuned: (1) the rule's links enter as a CONSTANT input (the c-rule link matrix of the cached tokens) with a fixed weight, so the step-0 head equals c-rule by construction and the fixture is a plumbing test; (2) the free sets are those of the rule on the kept tokens; (3) the kill that used a rule-step number is removed (D113 point 2: a rule-step system is never a kill reason); (4) the control is the candidate's own step 0; (5) **G-simple: the prior is a wiring rule in the forward path, pre-registered as such** (see below); (6) "the learning can only add" is withdrawn: the learning STARTS at the rule; the prior is FIXED in the arm of record; (7) a prior temperature so that the rule links start at p of about 0.8, not one-hot; (8) the zero-init list corrected to the code; (9) the fixture two-sided; (10) the predictions per gate condition; (11) JSON notes). Written by the Spurmitte chat that Mirko addresses as "Fable"; the idea was put to Mirko at about 01:35 in answer to his question "how can the model in the solution set fail?", the lane lead ordered the card at 01:45. **Status: v2, not built, nothing run.** Design, cache (B' = e292_a, K = 512), loss, decode of record and controls are those of E365 (experiments/E365_e2e-token-champion-s1, sections 0 - 1) unless written here.

**Why:** E365 S1 (section 3) is worse than the rule `lo_near` on the same tokens (topo F1 0.5486 / 0.5743 against 0.6265 / 0.6281): the head learned a conservative operating point (precision 0.86, recall 0.40; its loss scores every link alone, the topology metric whole chains, and the rule never refuses a link) and did not fit its training frames. **The question of E369:** a head that starts AT the rule and is trained from there: does the learning improve on the rule (remove its false links, add what it misses, the junction links in particular), or does it drift back to the conservative point of S1?

**The design:**
- **The rule input (constant, computed from the cached tokens like the allowed gate, no gradient, named):** r^short_ij = 1 where c-rule's `lo` links i -> j (nearest kept token ahead within 45 degrees of the heading, at most 8 m, heading difference at most 60 degrees), r^long_ij = 1 where `lo_near`'s extra long link goes (from a kept token without a `lo` successor to the nearest kept token without a `lo` predecessor inside the allowed gate, chord above 8 m, at most 40 m), both computed EXACTLY as `jtoken_eval.py ctl` computes c-rule (`rule_frame`: on the tokens with B' e >= 0.5, the free sets on these kept tokens only); has^c_i = 1 when token i has a rule link in class c.
- **The prior on the logits (FIXED in the arm of record, no learnable scale):** wiring class c: z^c_ij += lambda x r^c_ij; the none slot of class c: z^c_none,i += lambda x (1 - has^c_i) + mu; lambda = 12, mu chosen with lambda so that a rule link starts at p of about 0.8 among its candidates (the temperature: not one-hot, so that a wrong rule link does not cost tens of nats at the first step) and the none slot wins exactly where the rule has no link; the type head: the none logit += -e_j + log 6 INSIDE `read_types` (both readings, layer 1 and the last), so that at step 0 p_none(j) = sigmoid(-e_j) and the kept nodes are those of c-rule (e >= 0), and the key weights after SWITCH stay those of the prior. The values of lambda and mu are fixed on the TRAIN sample by the fixture below (never on VAL) and written here before the first training step.
- **The learned part = the S1 head unchanged, its outputs starting at zero:** zero-initialised at step 0: the type Linear (weight and bias), the wiring key projection `wk` (NOT `wq`, so that the q . k term has a gradient), the last layer of the wiring pair MLP `wbias`, the destination evidence b; the structural key bias log(w_j + 1e-3) stays (it is part of the S1 design; at step 0 w_j = sigmoid(e_j)) and is checked by the fixture. Nothing in the delivered graph is a STEP: the output is still the argmax / threshold of a softmax (the decode of record of E365); but **the rule input IS a wiring rule in the forward path** (a hand-written nearest-in-cone and free-set link list, a hard boolean feature of the inputs).
- **G-simple, pre-registered:** D113 point 4 lists "wiring rule" among what may not be in the delivered system. The prior counts as a wiring rule for G-simple UNLESS Mirko's recorded reading (a decision file, asked through the lane lead tonight) says otherwise; without that reading the best outcome of this card is QUALITY-ONLY, not a champion. The card is still worth its slots: it measures whether learning ADDS to the rule, which decides the next design (a rule-free head that is pushed toward recall, or a learned prior).
- **The FIXTURE (before training, CPU, both seeds, two-sided, written now):** the step-0 head's delivered graph equals c-rule's: at least 99.9 % of c-rule's links present AND extra links at most 0.1 % of c-rule's AND the kept node sets identical, on the TRAIN sample (291 frames, where lambda / mu are fixed) and then once on VAL; topo F1 within 0.001 of c-rule on VAL (0.6265 / 0.6281). The training does not start until it passes.

## 0. The card (what the tools run)
```json card
{
 "id": "E369",
 "card": "",
 "title": "E365 token head + the c-rule links and the B' existence as a fixed prior on its wiring and type logits (the step-0 head = c-rule), the learned part zero-initialised, one training stage of 12000 steps, two seeds, against its own step 0 (= c-rule), the E365 S1 heads and plain B",
 "kind": "variant",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.02, "vs": "ref"},
 "arms": {
  "ref": {"recipe": "the head's own step 0 = c-rule by the fixture (runs `E369.fix_s0 / fix_s1`; c-rule's stored cards runs/eval/e365_ctl/rule_e292_a_s<S>*.json: topo F1 0.6265 / 0.6281, F1@1.0 0.773 / 0.773, lanes found 0.586 / 0.586); no run_ids for decide.py: the verdict is by hand", "run_ids": ["E369.fix_s0", "E369.fix_s1"]},
  "pr": {"recipe": "`jtoken_train.py --name e369_pr_s<S> --seed S --ident e292_a --prior rule --steps 12000 --bs 8 --lr 1e-3 --workers 4` (the prior option is added by a coder; everything else as E365 S1)", "run_ids": ["E369.pr_s0", "E369.pr_s1"]}
 },
 "control": "ref",
 "variant": "pr",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "ref", "min_effect": 0.01}
 ],
 "budget_gpu_h": 2.0,
 "kill": "a loss is not finite; or no learning (the mean loss over steps 1801 - 2000 not below 0.8 x the mean over steps 101 - 200, read against the pre-registered step-0 loss); the fixture fails (no training starts). Diagnostic, never a kill: the TRAIN-sample topo F1 at step 3000 below the head's OWN step-0 TRAIN-sample topo F1 by more than 0.05 on both seeds is logged and the run continues. A run cut by run.py or the GPU is resumed from its 500-step checkpoint by the supervisor (resume steps reported); `--timeout` from 0.65 s per step at the start, then from the speed measured over 1500 steps",
 "evaluate": {"split": "val"}
}
```
**The verdict (by hand, written before any run; `decide.py` is not used):** (1) **FIXTURE** as above; (2) **ADDS** = the trained head's topo F1 >= its own step 0 + 0.01 on BOTH seeds (0.6365 / 0.6381) at the decode of record; **NO-ADD** = below on both seeds; else MIXED; (3) the **okayish gate of D113 / E365, unchanged** (G-topo >= 0.6359 / 0.6349; G-found: F1@1.0 >= 0.8182 / 0.8060, lanes found >= 0.6066 / 0.5937; G-cost; G-simple as pre-registered above): PASS / MISS / QUALITY-ONLY / INCONCLUSIVE as E365; a PASS needs the recorded reading on G-simple and then the confirmation on two fresh head seeds (E367). Reported next to it: the E365 S1 heads (learned only), the best E367 arm when it exists, c-hi (0.819 / 0.820), plain B, E324 s; the paired scene-bootstrap interval against the step 0 and against plain B per seed; the decode selected per head on its TRAIN sample (E366 / E367 rule) as a diagnosis.
**Pre-registered secondary numbers:** the step-0 training loss (against S1's final 0.42); the TRAIN-sample topo F1 at steps 0, 3000, 6000, 12000 (the drift); the long-class delivered share (S1: 51 %; c-rule's own share is read at the fixture); the share of delivered links that are not rule links (what the learning changed); the kept-node share against B' e >= 0.5.
**Prediction (guesses, with windows):** the fixture passes (95 %: it is plumbing now); the step-0 loss about 0.8 (0.5 - 1.5); trained topo F1 0.645 (0.60 - 0.69), F1@1.0 0.79 (0.75 - 0.83), lanes found 0.61 (0.56 - 0.66), mean over seeds; ADDS about 50 %, NO-ADD 35 % (the drift back toward the conservative point of S1, said now), MIXED 15 %; the quality half of the gate (G-topo and G-found on both seeds) about 20 % (F1@1.0 needs +0.045 / +0.033 over c-rule, lanes found +0.021 / +0.008, and S1 learned F1@1.0 DOWN); a full PASS needs the recorded G-simple reading on top: about 10 % tonight.

## 1. Card text
- **Statement:** a token head that starts at the rule improves on it by learning. **Mechanism:** the rule never refuses a link (its chains stay whole); the learning can remove its false links, add the junction links it misses and re-weight the existence. **Not claimed:** that this is rule-free (it is not: the rule is an input), that the prior's lambda / mu are optimal (fixed by the fixture on TRAIN), anything about S2 / S3 / S4 / TEST. **Cost:** as E365 S1 plus the rule computation on the cached tokens (CPU, about 1 ms per frame, reported) and two scalars: 0.127 M parameters; training about 40 min per pair.
- **Referee:** fresh ar-critic (cards) 2026-11-17 on v1: REVISE, 11 findings, taken in as listed in the header; the code critic after the build.

## 2. Detail plan
| step | what exactly | where | done when |
|---|---|---|---|
| build | `--prior rule` in `training/grid/jtoken.py` (the rule input from the cached tokens with the exact c-rule code, the fixed prior terms, the type prior inside `read_types`, the zero-init list above), trainer + `load_model` (recipe), CPU tests with mutants (step 0 = c-rule on hand frames; zero-init; gradient reaches the learned part, none reaches the prior), a `fixture` command in `jtoken_eval.py` | coder (after its current step) | tests green, code critic |
| fixture | lambda / mu on the TRAIN sample, then the step-0 head vs c-rule on VAL, both seeds (CPU) | `run.py --id E369.fix_s<S>` | PASS written here |
| train | pr, seeds 0 and 1, the supervisor pattern of E367, after the running E367 pairs, before c_long and the d 96 timing | `run.py --id E369.pr_s<S> --gpu --require-card` | done flags |
| read-out | dump, score (record + the TRAIN-selected decode), TRAIN sample at 3000 / 6000 / 12000, degraded | `run.py --id E369.read_s<S>` | cards |
| decide | the verdict above | by hand | result section, ledger notes, commit, one line to the lead |

## 3. Summary
(empty until the runs exist)
