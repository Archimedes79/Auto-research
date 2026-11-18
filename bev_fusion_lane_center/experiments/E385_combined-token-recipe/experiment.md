# E385 - the combined token recipe: the champion plus ONLY what survived its own written rule (distillation or the ensemble, S4, max-positive, the speed path), confirmed together on two fresh head seeds

Card of 2026-11-18 (system clock), **version 1 at 02:05, a DRAFT with the composition still open (filled from the verdicts BEFORE any run), before the critic pass**; written by the Spurmitte chat that Mirko addresses as "Fable" on the lane lead's order of 01:55 (D117 point 4 / D119: "ONE combined recipe card (what survived of max-positive loss, near-node none weight, S4, distillation, speed path / K 384) ... the rule written before any run, two seeds, VAL, a fresh ar-critic pass ... the card should name what is taken from each verdict and what the recipe is if each is a TIE"). Mirko, D119 (00:25): "bring this system to a finish line. Goal E2E differentiable. Fast. Simple & elegant, robust, good performing." D113 / D109 / D55 (E2E differentiable only, no rule step; at a tie the end-to-end variant, D109), D106 (the same cost: a tie takes the plain recipe), the six hard rules. TEST never touched.

**The question:** does the champion with all the parts that passed their own cards, trained TOGETHER in one recipe, keep (or add up) their gains on two fresh head seeds, at an unchanged or lower cost, with no rule step - i.e. is it the finished system of D119?

## The composition (each row filled from its card's written verdict BEFORE E385 starts; the cell 'taken' is decided by the rule in this table, not by hand)
| part | card and its verdict | taken into the recipe when | what the recipe has when NOT taken / a TIE | status at v1 (02:05) |
|---|---|---|---|---|
| base | E365 head, none weight 0.25 (E367 b_bal25, E373), D115 per-class decode (E375), K 512, top-K the named budget | always | - | the champion: VAL topo F1 0.6679 / 0.6762, junction F1 0.252 / 0.245, 16.6 - 16.8 ms whole path, 2.32 M parameters |
| max-positive loss (`--pos-loss max`) | E378 (confirmation of E376's side finding on fresh head seeds) | E378 ADOPT-LOSS | `--pos-loss sum` (the champion's); E378 TIE or NO: not taken (D106: same cost, a tie takes the plain recipe) | x_s0 passed clause (i) (topo F1 +0.0067 [+0.0013, +0.0146]); x_s1 trains tonight on slot 2; c (the twin) only on the ADOPT path |
| distillation (`--distill-from` two champion-recipe teachers) or the plain 2-head ensemble | E379: d (one distilled head) vs the champion; e2 (the ensemble) is a candidate only through the lead's D106 time gate (the stacked 2-head whole path <= 19.7 ms) | d ADOPT-DISTILL -> the recipe's training adds the KD term with two frozen champion-recipe teachers (a teacher stage: the recipe becomes 'two teacher heads, then one student'); e2 through its gate -> the delivered system is the 2-head ensemble (no student) | not taken: one head, no teacher | d_s0 trained (VAL read-out running), d_s1 trains until ~02:30; E379.time_e2 in the quiet gap (lane chat) |
| S4 (B''s node head trained with the head's loss, live tokens) | E372: S4-HOLDS / S4-GAINS / S4-LOSES | S4-HOLDS (D109 / D113: the end-to-end variant at a tie; the delivered system is then 'end to end: the loss reaches B''s node head') | S4-LOSES: the head on the frozen B' (stage 1 only) | seed 0 read: s4 - z topo F1 -0.0003 / junction F1 -0.0015, s4 - champion (live) +0.0002 / +0.0021 (within the guards); seed 1 trains tonight (slot 1) |
| the speed path (FastJTokenHead, fast_loss, the GPU-resident loader) | E383: parity of the speed path against the champion | E383 passes its parity rule -> the recipe trains with the speed path (same numbers by its rule, less time) | the plain trainer | E383 ACCEPTed as a card (7daf4851); its gate runs in the 04:30 quiet gap (lane chat) |
| K 384 (pruning) | E383 measures it as a diagnostic, not trained | NOT taken in E385 (no trained evidence; a later card) | K 512 | diagnostic only |
| near-node none weight | E377: TARGETING-ONLY (no adoption; it beat the scalar weight, not the champion) | never (closed) | - | removed from the code in the D117 simplification |
| recursive temporal fusion (D91 / Mirko 23:50: 'zeitliche Fusion ... nur rekursiv') | E384 (the lane chat: a recursive node state in the node head) | E384 passes its own rule | no temporal state | its own card; E385 waits for it only if it is decided before E385 starts, else it is the next card |

**If everything is a TIE or NO:** the recipe is the champion itself (+ the speed path if E383 passes, + S4 if it holds): E385 is then a confirmation that the combination of the cost-only / E2E parts keeps the champion's numbers (the 'hold' rule below), not a gain claim.

## 0. The card (what the tools run)
```json card
{
 "id": "E385",
 "card": "",
 "title": "E385 combined token recipe: the champion plus the parts that passed their own cards (composition table), trained together on two fresh head seeds (cache 0 / head 8, cache 1 / head 9), read at the pinned cell against the E373 champion heads at the D115 cell",
 "kind": "variant",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.005,
 "prediction": {"value": 0.004, "vs": "ref"},
 "arms": {
  "ref": {"recipe": "the champion: the E373 heads `runs/e373_cb_s0 / s1` at the D115 cell (short 0.3 / long 0.2 / keep 0.6 / none gate off): VAL topo F1 0.6679 / 0.6762, F1@1.0 0.8459 / 0.8467, lanes found 0.7191 / 0.7205, junction F1 0.2517 / 0.2452"},
  "c": {"recipe": "the combined recipe of the composition table (filled before the run; the exact command line written here then, run through `tools/jarm.py run E385 c_s<S> --seed S --head-seed H ...` with S = 0 / 1, H = 8 / 9: fresh head seeds, never used)", "run_ids": ["E385.c_s0", "E385.c_s1"]}
 },
 "control": "ref",
 "variant": "c",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "ref", "max_loss": 0.005},
  {"metric": "junction_F1", "vs": "ref", "max_loss": 0.01},
  {"metric": "F1@1.0", "vs": "ref", "max_loss": 0.005},
  {"metric": "lanes_found", "vs": "ref", "max_loss": 0.01}
 ],
 "budget_gpu_h": 3.0,
 "kill": "the kills of each taken part's card apply unchanged (E378: finite / learning; E379: the teachers' start check, KD in its band; E372: live_inputs asserted, the B-loss and head-loss rises, LabelOverflow; E383: its curve check); plus: a loss not finite; `--timeout` from the slowest taken part's measured speed x 1.3; one family on one slot with a supervisor and resume",
 "evaluate": {"split": "val"}
}
```
**The read-outs (named before any run; the standard scorer, all sections 0, no rule step, VAL 873, TEST never touched):** per head the pinned cell by `jtoken_eval.py choose-cell --runs E385.c_s0,E385.c_s1` (the mean of the two TRAIN tables; the E375 rule; the cell written here before its VAL score), the VAL score at that cell against the champion at D115 (cross-run paired interval), D115 reported; if S4 is taken: the LIVE read-out (the parity rule of E372); the whole-path ms on a quiet host (the E371 timing, batch 1) and the parameter count; the anatomy.
**The verdict (by hand: written before any run):** **ADOPT-RECIPE** (the finished system of D119) = on BOTH heads at the pinned cell against the champion at D115: topo F1 not lower by more than 0.005, junction F1 not lower by more than 0.010, F1@1.0 not lower by more than 0.005, lanes found not lower by more than 0.010; the D113 gate floors; the whole path at or below 19.7 ms (E324 s) and at most 1.1 x the champion's 16.8 ms unless a taken part's own card accepted a higher cost; no rule step; the kills did not fire. **ADOPT-RECIPE-GAIN** (reported with ADOPT-RECIPE): topo F1 higher by at least 0.005 with the paired interval above 0 on both heads (the parts added up). **NO** otherwise: the champion stays, and the composition is re-read part by part (which part broke it: one more read-out per taken part, reported). The recipe of an ADOPT becomes the champion's line in D114 through the lead (a new D<n>).
**Prediction (guesses, to be refined when the composition is known):** topo F1 c - ref +0.004 (window -0.005 to +0.013) if max-positive and distillation are both taken, about 0 if only cost / E2E parts are taken; ADOPT-RECIPE about 70 %, ADOPT-RECIPE-GAIN about 25 %, NO about 30 %.

## 1. Card text
- **Statement:** the champion's recipe with every part that passed its own card, trained together, is at least as good as the champion on two fresh head seeds and at most as costly (or it is the E2E variant at a tie), so that it can be delivered as the finished system.
- **Why one confirmation, not a factorial:** each part was tested alone against the champion by its own written rule (E372, E376 / E378, E379, E383); the combination can interfere (e.g. the max-positive loss and the KD term both shape the successor distribution; S4 moves the tokens the KD teachers were trained on); one run per seed of the combination answers 'do they add up' at a known cost; a broken combination is diagnosed part by part afterwards.
- **E2E / D55:** no part adds a rule step; the named hard steps stay: the top-K index choice, the allowed-pair gate on the inputs, the four output thresholds of the decode, the frozen pyramid; with S4 taken, B''s node head is in the loop (the label 'end to end: the loss reaches B''s node head').
- **Not claimed:** TEST; K 384 (a later card); temporal fusion unless E384 is decided before the start.
- **Inputs:** ready: the trainer options (`--pos-loss`, `--distill-from`, the S4 trainer jlive_train.py, the speed path of E383), `tools/jarm.py`, `jtoken_eval.py choose-cell / score / ensemble / --live`. Open: the verdicts of E372, E378, E379 (+ its time gate), E383.
- **Starts from:** the champion (E373 cb heads), B' e292_a, K 512, the D117 / D119 simplified code (landed 07:00 - 08:30).
- **Scheduling:** the day window 08:30 - 17:30 (start <= 15:30) after the verdicts and the simplification landed; one family on one slot.
- **Lens / group / sources / status:** D117 point 4 / D119; status DRAFT v1 (the composition open), awaiting the verdicts and the referee.
- **Referee:** pending (fresh `ar-critic`, cards, on this draft now; a second short look once the composition is filled).

## 2. Detail plan
| step | what exactly | where | done when |
|---|---|---|---|
| compose | fill the composition table from the verdicts by its rule; write the exact command line | this card | committed before any run |
| critic | a fresh ar-critic (cards) on the draft; a short second look on the filled composition | Fable | G2 yes |
| train | c_s0, c_s1 (fresh head seeds 8 / 9), one family, a supervisor and resume (S4 taken: the joint trainer after the head stage) | `jarm.py run E385 c_s<S>` | done flags |
| read-out | choose-cell (mean of the two TRAIN tables), VAL at the cell vs the champion, D115, live parity if S4, timing on a quiet host | `jtoken_eval.py`, the E371 timing | cards |
| decide | the verdict above | by hand | result section, ledger notes, commit, one line to the lead; the champion line through the lead (D<n>) |
