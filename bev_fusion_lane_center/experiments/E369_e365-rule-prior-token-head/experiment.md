# E369 - the E365 token head with the rule as a differentiable PRIOR on its logits: the step-0 head IS the one-line rule, the learning can only add

Card of 2026-11-17 (version 1, 01:55 system clock; written by the Spurmitte chat that Mirko addresses as "Fable"; the idea was put to Mirko at about 01:35 in answer to his question "how can the model in the solution set fail?", and the lane lead ordered the card at 01:45: "the most promising single idea of the night: it fixes the structural problem, the head is WORSE than the one-line rule on the same tokens; the prior makes the FLOOR the rule and lets the learning only add; it stays inside the solution set: nothing hard is added, the output is still the argmax of a softmax, the rule is only the INITIAL VALUE of a differentiable function"). **Status: draft, no critic pass yet, not built, nothing run.** Design, cache (B' = e292_a, K = 512), loss, decode of record, gate and controls are those of E365 (experiments/E365_e2e-token-champion-s1, sections 0 - 1) unless written here.

**Why:** E365 S1 (section 3) is worse than the rule `lo_near` on the same tokens (topo F1 0.5486 / 0.5743 against 0.6265 / 0.6281): the head learned a conservative operating point (precision 0.86, recall 0.40) because its loss scores every link alone while the topology metric scores whole chains (a missing link breaks a chain, the rule never says 'none'), and it did not even fit its training frames. A head that STARTS at the rule and is trained from there tests whether learning can ADD to the rule rather than whether it can find the rule.

**The design (the learned part of E365 unchanged; three prior terms added to its output logits; every prior term a differentiable function of the cached inputs with learnable scales; the learned logits start at exactly zero):**
- **Node existence prior:** the type logits get a prior on the none class: z_none += -(e_j) + log 6 (e = the B' existence logit after the layer), so that at step 0 p_none(j) = sigmoid(-e_j) and a node is kept exactly when B' keeps it (e >= 0), as the nodes of c-rule; the last Linear of the type head is zero-initialised (the learned type logits are 0 at step 0); one learnable scale on the prior term (init 1).
- **Short-link prior (the rule `lo`: nearest node ahead within 45 degrees of the heading, at most 8 m, heading difference at most 60 degrees):** z^short_ij += -(d_ij / s_S)^2 inside the rule's cone, a finite penalty -P_out outside it (learnable, init 20: the learning can open the cone), plus kappa x logsigmoid(beta x e_j) (the rule links only between kept nodes; init kappa 5, beta 4); s_S learnable, init 0.5 m (with d >= 2 m between neighbours the nearest candidate takes almost all the mass); the short none slot gets a prior logit equal to the cone boundary -(8 / s_S)^2 - 1, so 'none' wins exactly when no kept candidate lies in the cone.
- **Long-link prior (the rule `lo_near`: a node WITHOUT a short successor gets one long link, chord above 8 m, at most 40 m, inside the allowed gate, to the nearest node WITHOUT a short predecessor):** soft versions of the two set conditions, computed from the short prior itself: free_out(i) = p^short_prior(none | i), free_in(j) = exp(- sum_i p^short_prior(j | i)); z^long_ij += -(d_ij / s_L)^2 + log(free_in(j) + 1e-3) on the allowed long candidates within 40 m (penalty -P_out beyond), the long none slot prior = log(1 - free_out(i) + 1e-3) - log(free_out(i) + 1e-3) (a node with a short successor says 'none' in the long class); s_L learnable, init 1 m.
- **The learned part:** the S1 head (embedding, 3 self-attention layers with the soft key bias, the wiring layer, the type head) adds its logits to the prior; its last projections (wiring q / k, the destination evidence b, the type Linear) are zero-initialised, so at step 0 the output is the prior alone. Nothing is hard in the output path: the prior's cone and gate are masks on cached inputs (as the allowed gate of `jtok.pair_geometry` already is), the decode is the decode of record of E365 (named, on the output only). B' stays frozen (S4 is the later stage that opens it).
- **The FIXTURE (before any training, CPU, both seeds, written now):** the untrained (step-0) head's delivered graph on VAL equals c-rule's graph: topo F1 within 0.002 of c-rule (0.6265 / 0.6281) AND at least 99 % of c-rule's links identical (same source, same destination) AND the kept node sets identical for at least 99.5 % of the tokens. The prior's init values above are the starting point; if the fixture fails they may be changed ONCE before any training (the change and the fixture numbers are written into this card before the first training step; the training does not start until the fixture passes).

## 0. The card (what the tools run)
```json card
{
 "id": "E369",
 "card": "",
 "title": "E365 token head + the rule lo_near as a differentiable prior on its existence, short-link and long-link logits (learnable scales, learned logits zero-initialised: the step-0 head reproduces c-rule), one training stage of 12000 steps, two seeds, against c-rule (the floor), the E365 S1 heads and plain B",
 "kind": "variant",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.02, "vs": "ref"},
 "arms": {
  "ref": {"recipe": "c-rule = the rule links lo_near on the B' tokens with e >= 0.5, the delivered graph with every section 0 (runs/eval/e365_ctl/rule_e292_a_s<S>*.json): topo F1 0.6265 / 0.6281, F1@1.0 0.773 / 0.773, lanes found 0.586 / 0.586"},
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
 "kill": "a loss is not finite; the fixture fails (no training starts); or the TRAIN-sample topo F1 at step 3000 is below c-rule's TRAIN-sample topo F1 by more than 0.05 on both seeds (the learning destroys the prior: stopped, reported, not a verdict); a run cut by run.py or the GPU is resumed from its 500-step checkpoint by the supervisor (resume steps reported); `--timeout` from 0.65 s per step at the start, then from the speed measured over 1500 steps",
 "evaluate": {"split": "val"}
}
```
**The verdict (by hand, written before any run; `decide.py` reads only the JSON rule and is NOT the verdict):** (1) **FIXTURE** as above, before training; (2) **ADDS** = topo F1 of the trained head >= c-rule + 0.01 on BOTH seeds (0.6365 / 0.6381) at the decode of record; **NO-ADD** = below c-rule + 0.01 on both seeds; else MIXED; (3) the **okayish gate of D113 / E365 is unchanged** (G-topo >= 0.6359 / 0.6349, G-found: F1@1.0 >= 0.8182 / 0.8060 and lanes found >= 0.6066 / 0.5937, G-cost, G-simple; PASS / MISS / QUALITY-ONLY / INCONCLUSIVE as E365) and decides whether "a champion S1 exists" (then the confirmation of E367 applies: two fresh head seeds). The decode selected per head on its TRAIN sample (the rule of E366 / E367) is reported next to the decode of record, never the verdict here. Controls on the same tokens: c-rule (the floor), the E365 S1 heads (learned only), the best E367 arm when it exists, c-hi (0.819 / 0.820), plain B; the paired scene-bootstrap interval against c-rule and against plain B per seed.
**Pre-registered secondary numbers:** the TRAIN-sample topo F1 at steps 0 (= the fixture on TRAIN), 3000, 6000, 12000 (does the learning drift away from the rule toward the conservative point of S1?), the long-class delivered share (S1: 51 %), the learned scales s_S, s_L, P_out, kappa at the end, the share of the delivered links that are NOT rule links (what the learning added or changed).
**Prediction (guesses, with windows):** the fixture passes with the init values (80 %); trained topo F1 0.645 (0.60 - 0.69) mean over seeds; ADDS about 55 %, NO-ADD 30 % (the risk said now: the loss optimum of S1 is conservative, and the learning can move the head from the rule toward it; the step-3000 kill and the secondary numbers show it), MIXED 15 %; the okayish gate: PASS about 35 % (G-found: F1@1.0 of c-rule is 0.773, below the floor 0.818: the learned part has to lift F1@1.0 by about 0.045 as well, which is the harder half), MISS 40 %, INCONCLUSIVE / QUALITY-ONLY 25 %.

## 1. Card text
- **Statement:** a token head that starts at the rule and learns corrections beats the rule on the same tokens. **Mechanism:** the rule's links are the floor (it never refuses a link, so its chains stay whole); the learning can remove its false links (the rule's precision is moderate), add the long links it misses, and re-weight the existence. **Not claimed:** that the prior is the best prior (a learned rule, not a tuned one: its init values are fixed by the fixture, not by VAL); anything about S2 / S3 / S4 / TEST. **Simplicity:** one formula per prior term, four learnable scalars, the E365 head unchanged; the delivered graph has no rule STEP (the rule is the initial value of differentiable logits; the cone and the gate are masks on the cached inputs like the existing allowed gate). **Cost:** as E365 S1 (+ a few element-wise terms): 0.127 M parameters + 4 scalars; training about 40 min per pair.
- **Referee:** a short fresh `ar-critic` (cards) before the build; the code critic after the build.

## 2. Detail plan
| step | what exactly | where | done when |
|---|---|---|---|
| critic | fresh ar-critic on this card, v2 | Fable | committed |
| build | `--prior rule` in `training/grid/jtoken.py` (the three prior terms, zero-init of the learned last projections, the learnable scalars in the state dict), trainer + `load_model` (recipe), CPU tests with mutants (step-0 output = prior; prior = rule on hand frames; zero-init; gradient reaches the scalars and the learned part), a `fixture` command in `jtoken_eval.py` | coder | tests green, code critic |
| fixture | step-0 head vs c-rule on VAL, both seeds (CPU) | `run.py --id E369.fix_s<S>` | PASS written here |
| train | pr, seeds 0 and 1, the supervisor of E367 (`temp/e367/sup.sh` pattern), after the running E367 pairs, BEFORE c_long and the d 96 timing | `run.py --id E369.pr_s<S> --gpu --require-card` | done flags |
| read-out | dump, score (record + the TRAIN-selected decode), TRAIN sample at 3000 / 6000 / 12000, degraded | `run.py --id E369.read_s<S>` | cards |
| decide | the verdict above | by hand | result section, ledger notes, commit, one line to the lead |

## 3. Summary
(empty until the runs exist)
