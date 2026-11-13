# E325 - the learned wiring head: which delivered piece a connector end is joined to, chosen by a small softmax head instead of the nearest-piece rule (E2E replacement of the output selection, D87)

Card of 2026-11-13 20:50, written by the lane chat after D87 (the build card of E322: `experiments/E322_soft-wiring/experiment.md`, whose gate G0 is the result that this card starts from). **Status: card only, no code, nothing started.** D87: the one-piece nearest-end rule is the only rule step of the delivered graph that has no differentiable replacement yet, and E2E replacements without an end-to-end alternative are kept and refined, not cut; the bar is therefore EQUAL accuracy, not a gain (G0: the oracle choice at the GT end gains -0.0007 / +0.0013 topo F1 over the rule on E312, so a head that imitates it can at best tie), and the price is stated: parameters, MACs, ms, host time.

## 0. The card (what the tools run)
```json card
{
 "id": "E325",
 "card": "",
 "title": "a learned softmax head over the delivered piece ends in reach (4 m / 45 degrees) chooses the piece each connector end of E312 is wired to; the hard nearest-piece rule is replaced at the read-out by the head's argmax (the weights are the edge probabilities of the output)",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.0, "vs": "c"},
 "better": {"dup_frac": "lower", "vert100_dp": "lower"},
 "arms": {
  "c": {"recipe": "control = E312 of the same seed, snapped read-out at p >= 0.1 with the nearest-piece rule (`runs/e312_p_s<S>/ends_card_snap_thr0.1.json`): topo F1 0.6825 / 0.6864, junction F1 0.3194 / 0.3143, dup_frac 0.1747 / 0.1557, F1@1.0 0.8669 / 0.8685, lanes found 0.7376 / 0.7423, vertices 19.55 / 20.10"},
  "w": {"recipe": "the same E312 dump (keypoints, pairs, handle scales), the same delivered graph of B + `close:1.5@0.5`, the same snapped drawing; only the choice of the piece per end is the argmax of the head `Wire` (section 1) instead of the rule; the head is trained on TRAIN frames (seed S: B = `runs/e273_p_s<S>`, keypoints of `runs/e312_p_s<S>`); run ids E325.w_s<S>", "run_ids": ["E325.w_s0", "E325.w_s1"]}
 },
 "control": "c",
 "variant": "w",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.005},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.01},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03},
  {"metric": "lanes_found", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 1.0,
 "kill": "the loss is not finite; or the head does not learn: its agreement with the oracle label on the TRAIN ends with a real choice (two or more places) stays below the rule's own agreement (75 %, `jwiring.py score`, stat multi_same / multi); or on the VAL ends with a real choice the head's agreement with the oracle label is below the rule's 75.0 % / 74.5 % on either seed (a head that imitates the label worse than the rule is not worth its cost)",
 "evaluate": {"split": "val"}
}
```
**Verdict rule (written before the run; the training runs go through `run.py`, the scores of stored dumps are 'by hand: replay'):** **KEEP (the head replaces the nearest-piece rule at the output)** when, on BOTH seeds, snapped topo F1 of w is at least c - 0.005 and every guard holds. **TIE, kept and refined (D82 / D87):** a bound or a guard is missed but the kill did not fire: the head is the differentiable replacement of the rule, the miss and its price are reported and a follow-up card removes the defect (this is the D87 reading; the written token is not changed). **CUT** only by the kill (the idea is shown impossible: no learning). Reported next to it, not in the rule: the agreement with the oracle label on the real choices, the calibration of the weights as edge probabilities (the share of ends where the weight of the chosen piece is above 0.5 and how often those are right), parameters, MACs and ms.

## 1. Card text
- **Statement (falsifiable):** a head with about 1 k parameters that sees the geometry of each candidate piece end and the pair it belongs to reproduces the nearest-piece choice of the snap at equal topology (snapped topo F1 within 0.005 of c on both seeds), so that the last selection rule of the delivered graph has a learned, differentiable replacement whose weights are the edge probabilities for the planner.
- **What G0 says and does not say:** the choice among the delivered pieces in reach has no accuracy headroom for a head that imitates the rule at the GT end (E322 section 2: o - c -0.0007 / +0.0013, a wrong choice on purpose costs 0.03, reach 4 m is the best). So this card is an end-to-end card, not an accuracy card: its prediction is a tie. G0 does not cover ties among co-located ends (12 - 14 % of the oracle ends), wiring both of them, or a dustbin; the follow-up cards are named below.
- **The head `Wire` (new module `training/grid/jwire.py`):** for every connector end i (entry or exit) of a pair and every candidate piece end c in reach (the same gate as the rule: within 4 m and 45 degrees, at most 8, an index choice) the features: distance d, offsets along and across the end's heading, the heading difference (cos, sin), the kind of the piece (lane / link), the logarithm of its length, the existence and link probabilities that B gives its end node, the keypoint's kind and probability, and the partner end (chord length, heading difference of the pair, kp probability of the partner). A 2-layer MLP (hidden 32, about 1.2 k parameters, about 1.2 kMACs per candidate) gives one logit per candidate; the softmax over the candidates is the weight (edge probability) of each piece; a fixed dustbin logit (-R^2 / (2 sigma^2) with R = 4 m, sigma = 1.2 m) keeps the rule's behaviour for the no-candidate case and never wins while a candidate is in reach, so the argmax is a choice among the candidates. Loss: the cross entropy against the label, over the ends with a label. **Label:** the piece end the rule picks at the matched GT end (G0's oracle; matching as `qendnet.match_keypoints`), ends that match no GT end or find no piece at the GT end carry no label in this card (the dustbin as a target is the follow-up: the oracle is not an upper bound there, E322 section 3). The head is trained on frozen inputs; joint training with the pair head (the keypoints get a gradient from the wiring loss through the distance features) is the next refinement.
- **Data (to be built, section 2):** B's delivered pieces and E312's keypoints are needed on TRAIN frames: B's forward and read-out on the TRAIN frames (GPU, 8414 frames, about 4 minutes per seed), the E312 model's dump on the TRAIN frames (about 6 minutes per seed), the GT ends of the TRAIN frames from the training loader. **Caveat:** both are IN-SAMPLE on TRAIN (better pieces and keypoints than on VAL: the VAL keypoint error is 1.2 - 1.4 m); the head is trained with a keypoint jitter drawn from the measured VAL error (position sigma 1.0 m, heading sigma 6 degrees: from `jdiag.py kperr`) and judged on VAL only. The head sees no VAL frame and no TEST frame.
- **Read-out:** `add_connectors` gets a `wire_choice` option (default off = the rule, bit for bit): a per-pair list of the chosen piece ends, used where the snap now takes the nearest. The jwiring-style trick of G0 (put the chosen piece end into the pair as its end and let `snap=True` re-select it at distance 0) is NOT used for the read-out because of co-located ends (it re-selects by list order); the option passes the piece index directly.
- **Named hard steps that remain:** the argmax of the head per end (the dustbin = not wired, here never chosen), the candidate gate and its cut at 8, the p >= 0.1 threshold, the keypoint peak choice, the close-node step and the rest of B's read-out (E292), the Hermite sampling and Douglas-Peucker in the scorer.
- **Controls and confound statement:** c (rule) against w (head) on the SAME E312 dump and the same graph, so only the choice differs; the oracle o and the positive control `second` from E322 are the ceiling and the floor of the choice (topo F1 c, o 0.6817 / 0.6878 and `second` 0.6513 / 0.6563 on E312); the head is trained once per seed, its training noise is one draw (the bootstrap of the scene resamples is the interval); the jitter parameters come from the VAL error statistics (a summary of VAL, no VAL label enters the head).
- **Prediction (numbers, written before the run; a hit is within +-0.003 for topo F1, +-0.05 for agreements):** w - c = 0.000 topo F1 (an imitation of a label that ties the rule); agreement with the oracle label on the real choices 0.78 (the rule 0.75: the head also sees heading, length and the partner end; if it does not beat the rule the kill says so); lanes found, F1@1.0, dup_frac, junction F1 within 0.003 of c.
- **Price (to be filled by the run):** parameters about 1.2 k (+ the dustbin constants), MACs about 1.2 kMACs x 8 candidates x about 23 ends = about 0.2 MMACs per frame (against 7.96 GMACs), ms: the head is batched over the ends of a frame, the host-side candidate lists are the cost (the pair head is already host-bound at 9.7 ms); measured on the CPU and, in the window, on the idle GPU.
- **Not claimed:** any gain; anything about ties (12 - 14 % of the oracle ends), a dustbin, or k = 2 pieces per end (follow-up cards); anything on TEST (D84 covers only E312).

## 2. What exists and what has to be built (after the critic pass)
Exists: `evaluation/jwiring.py` (`rule_candidates`, `rule_end`, `match`, the oracle, the GT-ends extraction, selftest), `qends_eval.add_connectors` and `make_pairs`, `jends_eval.load_model`, `qgraph_readout.dump_full`, `lane_gt_data` loaders with the GT ends. To build: (1) `training/grid/jwire.py` (features, head, loss, dataset from cached TRAIN dumps) with its test (the shapes, the softmax, the gradient, the dustbin never winning, the rule as a special case: with the logit = -distance the argmax equals `rule_end`, mutations killed); (2) the TRAIN dumps (B pieces per frame, E312 keypoints per frame) as a job of `jtest_eval.py`-style code; (3) `add_connectors(wire_choice=...)` with a flag-off identity test against the E312 cards; (4) `jwiring.py score --head <run>` writing its own card file (never the card of record); (5) the training script and the run through `run.py` (ids E325.w_s0 / E325.w_s1), READY file after a fresh critic pass on the code. A zero-parameter twin (softmax of the negative squared distance, the same dustbin; argmax = the rule exactly) is the interface option of E322 section 1 and the first thing to build as a test fixture: with it the read-out path is verified bit for bit before the head is trained.

## 3. Risks
(1) The in-sample TRAIN inputs make the head overconfident on VAL (the jitter and the VAL-only judgment are the guard). (2) Features that leak the label (the distance to the GT end is not a feature; only the keypoint-relative geometry is). (3) The head learns the rule and nothing else (agreement equal to the rule's): then the kill fires by the 'not above the rule' condition and the honest statement is that the choice is not learnable beyond the distance rule from these features; D87 keeps the replacement as a documented twin, not as a delivered path. (4) The host time of the candidate lists may exceed the rule's: stated as price.

## 4. Referee
The fresh `ar-critic` pass (cards mode) is written below when it has run.
