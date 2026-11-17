# E372 - S4: the joint fine-tune of the token champion with B' in the loop (the loss on the delivered graph reaches B's node head): from "staged, B frozen" to end to end

Card of 2026-11-17 (version 1, 03:30 system clock, TEXT ONLY: not built, no critic yet, nothing run; written by the Spurmitte chat that Mirko addresses as "Fable" on the lane lead's order of 03:15: "S4 is now the next E2E-claim step: with the S1 recipe as the champion, S4 (joint fine-tune through the soft existence gate into B's node head, B in the loop online, the cache no longer enough) turns 'staged, B frozen' into 'the loss reaches B'"). **Precondition:** the champion candidate E366 (the E365 S1 head + the decode chosen on the TRAIN sample) is CONFIRMED by E371 (two fresh head seeds) and its G-cost is closed on a quiet host; until then this card is a plan.

**The question:** does ONE joint fine-tune of the S1 head together with B''s node head (B' = runs/e292_a_s<S>: the E273 pyramid, the node head with the soft Matrix-NMS layer), with the gradient of the head's loss reaching B through every differentiable input of the head, keep the quality of the S1 champion (topo F1 not lower by more than 0.01 on both seeds) and so make the claim "a loss on the output reaches every module" (D55) true, except the named top-K index choice?

**Stages:** S1 (the head trained on the frozen cache) is stage 1; S4 is stage 2: the "at most two training stages" of G-simple holds (E370's two-stage recipe would make S4 a third and is therefore not the base of this card).

**The design (written now; details fixed in v2 after the critic and the build plan of the lane chat):**
- **Data path (B in the loop, online; the lane chat builds it, it knows B's trainer):** per step a batch of TRAIN frames: the 39 frozen BEV input channels of B (as `lane_tok_data` reads them) -> B''s pyramid (frozen in the arm of record; the last pyramid layers unfrozen only in a later arm) -> B''s node head (TRAINABLE) -> the top-K 512 cells by B''s existence (the named budget: an index choice, no gradient through the choice) -> the token fields of the cache computed LIVE (tok, pos, dirv, e, e_raw, flags, s, uout, uin) -> the S1 head -> the S1 loss. The labels (assignment, types, token-level targets with the contraction) are recomputed LIVE from the live tokens and the frame's GT table with the cache's code (`lane_tokcache.label_frame`), because the tokens move when B moves.
- **The gradient reaches B** through every input of the head that B produces: the token features `tok`, the positions and headings (they enter the relative-geometry bias and the allowed gate's features), the existence and flag logits (the embedding, the layer-1 key bias, the pair features e_i, e_j). No new gate is added to the head (the head's architecture = the champion's, so the start is the champion bit for bit; a "token x sigmoid(e)" gate would change the architecture and is a later arm if the gradient into the existence proves too weak: the share of B's parameter update norm that comes from the head's loss is a pre-registered secondary number).
- **Loss:** the S1 head loss + B''s own node-head loss (its E273 / E292 training loss, unchanged, weight 1: it keeps B's existence, offsets and flags calibrated while the head's gradient moves it); learning rates: the head 1e-4, B''s node head 1e-4 (both low), cosine, 3000 steps (a fine-tune), the champion's weights as the start (E366 heads, seeds 0 / 1, with their B' of the same seed).
- **Read-out:** the delivered graph of the jointly trained system on VAL at the champion's decode (frozen: link 0.1, keep 0.6, none gate off); the decode re-selected on the TRAIN sample by the E366 rule is reported as a diagnosis; dumps from the LIVE path (no cache).

## 0. The card (what the tools run)
```json card
{
 "id": "E372",
 "card": "",
 "title": "S4: joint fine-tune of the E366 champion head and B''s node head with B in the loop (live tokens, live labels, the head's loss reaching B through every differentiable token input; top-K 512 the named index budget), 3000 steps, two seeds, read at the champion's frozen decode, against the champion",
 "kind": "variant",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.0, "vs": "ref"},
 "arms": {
  "ref": {"recipe": "the confirmed champion: E366 (VAL topo F1 0.6617 / 0.6767 at link 0.1 / keep 0.6 / none gate off), read from the LIVE path with B' frozen (a parity check against the cache read-out first); the verdict is by hand"},
  "s4": {"recipe": "the joint fine-tune (trainer and data path to be built: lane chat + coder), 3000 steps, lr 1e-4 for the head and B''s node head, B''s pyramid frozen", "run_ids": ["E372.s4_s0", "E372.s4_s1"]}
 },
 "control": "ref",
 "variant": "s4",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "ref", "min_effect": 0.01}
 ],
 "budget_gpu_h": 3.0,
 "kill": "a loss is not finite; or no learning; or B''s own node loss rises by more than 20 % over its start (B is being destroyed: stopped, reported); a run cut by run.py or the GPU is resumed from its checkpoint by the supervisor",
 "evaluate": {"split": "val"}
}
```
**The verdict (by hand, written before any run):** S4-HOLDS = the okayish gate of E365 section 0 holds on both seeds AND the topo F1 is not lower than the champion's (read on the live path) by more than 0.01 on either seed; then the champion becomes "end to end: the loss reaches B" (only the top-K index choice, the allowed-pair gate on the inputs and the output thresholds are hard, named); S4-LOSES = the quality gate fails or the loss exceeds 0.01 on a seed: the champion stays "staged, B frozen". **Pre-registered secondary numbers:** the share of B''s parameter update norm that comes from the head's loss; B''s node recall / precision before and after; the TRAIN-to-VAL drop; the ms of the whole path (unchanged architecture: equal to the champion's).
**Prediction (guesses):** S4-HOLDS about 55 %; topo F1 change -0.005 (-0.03 to +0.02).

## 1. Card text
- **Referee:** none yet; a fresh ar-critic in cards mode after v2 (the data path of the lane chat known).
- **Open before v2:** the live data path (the 39-channel inputs and B''s trainer), the cost of the live labels per batch (the cache's label code on CPU), the memory of B' + head at bs 8, whether B''s loss needs its own targets per batch (the lane chat knows).

## 2. Detail plan
| step | what exactly | where | done when |
|---|---|---|---|
| plan | the live data path and B''s loss: a short plan from the lane chat (it knows B's trainer) | lane chat | message |
| card v2 | the design fixed, a fresh critic | Fable | committed |
| build | the joint trainer (the S1 head's trainer + B' live + live labels), a parity check (B' frozen, live path = cache read-out), CPU tests | lane chat (data path) + coder (trainer) | tests green, code critic |
| runs | s4_s0, s4_s1 after the champion is confirmed | `run.py --id E372.s4_s<S> --gpu --require-card` | done flags |
| read-out | live dumps, score at the frozen decode | `run.py --id E372.read_s<S>` | cards |
| decide | the verdict above | by hand | result section, ledger notes, commit, one line to the lead |

## 3. Summary
(empty until the runs exist)
