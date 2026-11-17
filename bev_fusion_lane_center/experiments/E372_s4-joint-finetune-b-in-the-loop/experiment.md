# E372 - S4: the joint fine-tune of the token champion with B' in the loop (the loss on the delivered graph reaches B's node head): from "staged, B frozen" to end to end

Card of 2026-11-17 (system clock). **Version 2 at 17:40, before the critic pass**: the design fixed on the lane lead's order of about 17:30 (D86: "the S4 joint trainer is yours ... E372 card update + trainer with the critic's safety points (live_inputs asserted every step, node-head grads clipped, LabelOverflow handling, update share measured) ... it needs the critic pass and a 50-step gate before the night start 22:00 on one slot ... The lane chat provides `lane_live_tok.py` (a03ef4e4) and the arm list proposal (z = frozen B' control vs s4 = trainable node head, two seeds each) ... If the trainer is not through its critic by 21:30, the night start slips to the next window, nothing is forced"). Version 1 (03:30, commit 0333d202) was text only: a plan against the E366 champion before the data path existed. Since then: the champion is the E373 b_bal25 head with the D115 per-class decode (E375 / D115), the live data path exists (`training/data/lane_live_tok.py`, a03ef4e4, with its selftest, 37 mutants, CPU fixture and gradient check), the head keeps its inputs' graph on request (`JTokenHead(live_inputs=True)`, baeb868d, code critic: the full gradient of all 9 float inputs against float64 finite differences, 866ba519). Written by the Spurmitte chat that Mirko addresses as "Fable". D113 / D109 / D55 (E2E differentiable only, no rule step), the six hard rules. TEST never touched.

**The question:** does ONE joint fine-tune of the champion head together with B''s node head (B' = runs/e292_a_s<S>: the E273 pyramid frozen, the node head with the soft Matrix-NMS layer TRAINABLE), with the gradient of the head's loss reaching B through every differentiable input of the head, keep the quality of the champion (topo F1 not lower than the matched frozen-B' control by more than 0.01 on both seeds) and so make the claim "a loss on the output reaches every module" (D55) true for B''s node head, except the named top-K index choice?

**Stages:** S1 (the head trained on the frozen cache) is stage 1; S4 is stage 2: the "at most two training stages" of G-simple holds.

## 0. The card (what the tools run)
```json card
{
 "id": "E372",
 "card": "",
 "title": "S4: joint fine-tune of the E373 champion head and B''s node head with B in the loop (live tokens, live labels, the head's loss reaching B through every token input; top-K 512 the named index budget), 3000 steps, two seeds, against the same fine-tune with B' frozen (z), read live at the D115 cell",
 "kind": "variant",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": -0.003, "vs": "z"},
 "arms": {
  "ref": {"recipe": "the champion as delivered (cache path): the E373 heads `runs/e373_cb_s0 / s1` at the D115 cell: VAL topo F1 0.6679 / 0.6762, F1@1.0 0.8459 / 0.8467, lanes found 0.7191 / 0.7205, junction F1 0.252 / 0.245 (reported; the live read of the same heads with B' frozen is the parity check)"},
  "z": {"recipe": "the matched control: the joint trainer with B''s node head FROZEN (`--train-nodes 0`): the champion head (start = runs/e373_cb_s<S>/model.pt, cache seed S, head seed 2 / 3) fine-tuned 3000 steps at lr 1e-4 on LIVE tokens and labels of the frozen B' (runs/e292_a_s<S>), the head's loss only", "run_ids": ["E372.z_s0", "E372.z_s1"]},
  "s4": {"recipe": "the same joint trainer with B''s node head TRAINABLE (`--train-nodes 1`): the same start, data order and schedule as z; loss = the head's loss (the champion's: none weight 0.25, --pos-loss sum) + 1.0 x B''s own node loss (`qgraph.node_loss`, its E292 form); the head 1e-4 and the node head 1e-4 (one schedule, cosine, 3000 steps, bs 8); one clip (3.0) over ALL trained parameters", "run_ids": ["E372.s4_s0", "E372.s4_s1"]}
 },
 "control": "z",
 "variant": "s4",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "z", "max_loss": 0.01},
  {"metric": "junction_F1", "vs": "z", "max_loss": 0.01}
 ],
 "budget_gpu_h": 2.2,
 "kill": "a loss is not finite; or no learning of the head (the mean logged head loss at steps 1800 - 2000 not below 0.95 x the mean at steps 100 - 200: a fine-tune from a trained head moves little, so the bar is low); or B''s own node loss rises by more than 20 % over its mean at steps 1 - 100 (B is being destroyed: stopped, reported); or the head is not live (the trainer asserts `head.live_inputs` every step in s4 and refuses a zero gradient norm of the head's loss on the node head at step 1); or more than 1 % of the steps hit `LabelOverflow` (each one skipped and counted); `--timeout` 90 min; one job family per slot with a supervisor and resume (a resumed run is reported as such)",
 "evaluate": {"split": "val"}
}
```
**The read-outs (named before any run; the standard scorer as E365, all sections 0, no rule step, VAL 873, TEST never touched):** per run a LIVE VAL dump (the run's B' node head + the run's head, under no_grad, through `lane_live_tok.LiveSource`: no cache), scored at the D115 cell (frozen: short 0.3 / long 0.2 / keep 0.6 / none gate off), the paired scene-bootstrap interval s4 minus z of the same seed (frames paired; the token sets differ, so the pairing is by frame, not by token), and each arm minus the champion's cache read-out (reported). **The parity check first** (before the runs are read): the champion heads read LIVE with the untouched B' (`e373_cb_s<S>` + `e292_a_s<S>`) against their cache read-out at D115: the topo F1 difference must lie within +-0.003 (the GPU fixture's float differences; the lane chat's `fixture --gpu` is the token-level diagnostic), else the live read-out is not trusted and the verdict waits.
**The verdict (by hand: written before any run; the JSON clauses mirror part of it):** **S4-HOLDS** = on BOTH seeds at D115: s4 minus z topo F1 not lower by more than 0.01 and junction F1 not lower by more than 0.01; the D113 gate floors hold for s4 (topo F1 >= 0.6359 / 0.6349, F1@1.0 >= 0.8182 / 0.8060, lanes found >= 0.6066 / 0.5937); the parity check passed; the kill did not fire: then the champion's description becomes "end to end: the loss reaches B''s node head" (hard and named: the top-K index choice, the allowed-pair gate on the inputs, the frozen pyramid, the four output thresholds), the recipe stays a two-stage one (S1, then S4). **S4-GAINS** (reported with S4-HOLDS): s4 minus z topo F1 >= +0.01 with the paired interval above 0 on both seeds. **S4-LOSES** otherwise: the champion stays "staged, B frozen". **Pre-registered secondary numbers:** the share of the node head's parameter UPDATE norm that comes from the head's loss (measured from the optimiser's steps: Adam's update with and without the head-loss gradient on a logged step every 200 steps), the gradient-norm share (0.28 - 0.32 at the start, lane chat), B''s node recall / precision (its node-loss read-out) at the start and the end, the count of LabelOverflow steps, the TRAIN-to-VAL drop, ms per frame (architecture unchanged: the champion's).
**Prediction (guesses):** s4 - z topo F1 **-0.003** (window -0.02 to +0.01), junction F1 -0.005; the update share of the head's loss 0.2 - 0.4; S4-HOLDS about 65 %, S4-GAINS about 10 %, S4-LOSES about 35 %. Reasons: the node head already serves the champion well (the champion was trained on its tokens); the head's gradient moves the tokens that the head's labels are recomputed from, which can chase its own targets (a moving-label risk the low lr and the 20 % kill bound); the z arm absorbs the effect of 3000 more head steps on live tokens.

## 1. Card text
- **Statement:** with the node head of B' trained jointly with the champion head (the head's loss + B''s own loss), the delivered graph's topo F1 and junction F1 stay within 0.01 of the same fine-tune with B' frozen on both seeds.
- **Mechanism / E2E (D55):** the token inputs (tok, pos, dirv, e, e_raw, s, uout, uin, flags) carry the graph from B''s node head into the champion head (`live_inputs=True`); the labels are recomputed from the DETACHED live tokens each step (`lane_tokcache.label_frame`: constants); the top-K index choice is an argsort on the detached existence logit (the named budget); the pyramid is frozen (eval, no_grad). Not differentiable, named: the index choice, the labels, the allowed-pair gate on the inputs, valid, the short / long split at 8 m (`cand`).
- **The trainer (to build, Fable, before the critic of the code):** `training/train/jlive_train.py` (design 'jlive', its own RECIPE incl. `live_inputs`, `train_nodes`, the start head and B' run, the weight of B''s loss): per step `no, tk = src.tokens(x)`, the head on `head_inputs(tk)`, then `src.labelled(tk, tables, token, row, token_drop=0, gate=False)`, the head's loss (`jtoken_loss`, the champion's options) + w x `b_loss(no, targets)`; one backward; the clip over head.parameters() AND src.trainable() (trainer.fit clips only model.parameters(): the joint module holds both so that the clip covers them, the critic's point); safety: `head.live_inputs` asserted every step in s4 (and False-equivalent in z: no gradient reaches B), the node-head gradient norm of the head's loss > 0 at step 1 in s4, LabelOverflow skipped and counted (kill above 1 %), B''s node loss logged against its start; the UPDATE share measured every 200 steps; checkpoint = head + node head (resume); a model.pt the read-out can load (the head's args incl. live_inputs, and the node head as a complete QGraph state for `qpredict.load_model`). The live read-out: `jtoken_eval.py dump --live` (the run's B' + head under no_grad through LiveSource; the dump carries its own tokens_np, so `score` reads them instead of the cache) and `--ref-run` pairing live dumps by frame.
- **The 50-step gate (before the night start):** z and s4 on seed 0 for 50 steps each on one GPU slot (`run.py --no-card`, diagnostic): finite losses, the s4 gradient check at step 1, s / step, GPU memory (planned ~4.5 GB), label seconds per step, LabelOverflow count, the update share at step 50; the timeout set from the measured speed x 1.3.
- **Not claimed:** the pyramid (frozen), B''s link / section / assignment heads (not in this path), TEST, S2 / S3 combined with S4.
- **Inputs:** ready: `lane_live_tok.py` (a03ef4e4; its GPU fixture is the lane chat's diagnostic before the runs), `JTokenHead(live_inputs)` (baeb868d / 866ba519), the champion heads E373 cb, B' runs e292_a_s0 / s1, the read-out tools. To build: the trainer and the live read-out above, with CPU tests and a code critic.
- **Starts from:** the champion (E373 cb heads, D115 decode), B' e292_a, K = 512, lane GT signature 9ad1d4ce30, no token drop.
- **Scheduling:** the night window from 22:00 on one slot (the lead), the other slot for E378's conditional twin c; if the trainer is not through its critic by 21:30 the start slips to the next window.
- **Lens / group / sources / status:** D113 S4 (the E2E claim), the lead's order of about 17:30; status proposed, v2 before the critic.
- **Referee:** none yet (fresh `ar-critic`, cards, now).

## 2. Detail plan
| step | what exactly | where | done when |
|---|---|---|---|
| card v2 | the design above, a fresh critic | Fable | committed, critic through |
| build | `jlive_train.py` (trainer + safety + update share), `jtoken_eval.py dump --live` and the live pairing, CPU tests, a code critic | coders + Fable | tests green, critic accepts |
| gate | 50 steps of z and s4 (seed 0), diagnostic | `run.py --no-card` | numbers in section 3 |
| parity | the champion heads read live with the untouched B' vs their cache read-out at D115 | `run.py --id E372.parity_s<S>` | within +-0.003 |
| runs | z_s0, s4_s0, z_s1, s4_s1 (one family, one slot, night) | `run.py --id E372.<arm>_s<S> --gpu --require-card --timeout 90` | done flags |
| read-out | live VAL dumps, scores at D115, s4 minus z per seed | `run.py --id E372.read_<arm>_s<S>` | cards |
| decide | the verdict above | by hand | result section, ledger notes, commit, one line to the lead |

## 3. Summary
(empty until the runs exist)
