# E287 Junction pair head v2: entry-to-exit pairs from the junction maps, with the order rule and the evidence along the curve (D82, step 3 of the thread)

Card of 2026-11-11, written by the lane chat on Mirko's word (D82: "wenn 2 Ein und 2 Ausgaenge da sind, dann ist die richtige Loesung eingang 1 auf ausgang 1 ... Ggf. bei Tie Loesung nicht verwerfen, sondern verfeinern ... Und bleibe differenzierbar"). Builds on E285 (GT layer) and E286 (junction maps); it refines E284 (the end head on node tokens, TIE on both seeds: topo F1 -0.005 / -0.003 against the plain delivered graph, pair recall 0.31, precision 0.64, end recall 0.58): a TIE is kept and refined, not dropped, but must not get worse (D82). Literature of the day (`literature/scan_2026-11-11_junction_topology.md`, `..._pruning_uncertainty_time.md`): independent pair logits are what the field does; no published method puts an order constraint on entry x exit; the rule control of Nilsson 2025 (angle bin gives the exit arm, lane order the exit lane) is the control to beat.

## 0. The card (what the tools run)
```json card
{
 "id": "E287",
 "card": "",
 "title": "junction pair head v2 (`jends.py`, trained by `jends_train.py` on the frozen junction maps of E286): keypoints from the 1 m entry / exit heat, pair logits with the order-rule features, the evidence along the Hermite curve and a soft crossing penalty; its connectors are added to the delivered graph of B + the hard step of E275, against that plain graph",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.02,
 "prediction": {"value": 0.015, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = the plain delivered graph of B + the hard step of E275 (runs e273_p_s0 / e273_p_s1, close:1.5@0.5), the same as the control of E284; no new training"},
  "p0": {"recipe": "pair head on the new keypoints, no order features, no evidence, no crossing penalty (`--no-order --no-evidence --cross-w 0`): the E284 head on better pixels"},
  "p1": {"recipe": "+ the order rule: rank features of the entry / exit groups and the soft crossing penalty (`--no-evidence`)"},
  "p2": {"recipe": "+ the evidence along the Hermite curve (lane-centre probability of B, zone, conflict core, stop line, crossing): the full head"},
  "q2": {"recipe": "the full head of p2 on the maps of the JOINT arm jj of E286 (the pyramid trained together with the junction head); the delivered graph under the connectors is still B's own"}
 },
 "control": "c",
 "variant": "p2",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "min_effect": 0.02},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_gain": 0.02},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 1.5,
 "kill": "a loss that is not finite; fewer than 1 % positive pairs among the allowed ones in the first 200 steps (the keypoints do not match the GT ends: the maps run is broken); the maps run of E286 missing",
 "evaluate": {"split": "val"}
}
```
**By hand (the tools cannot read the scorer's cards; the verdict script is `evaluation/jends_eval.py score` = `qends_eval.cmd_score`, the rule is D79's, applied to the plain delivered graph of the same seed, VAL 873 frames):** KEEP when topo F1 rises by more than +0.02 on BOTH seeds with F1@1.0 not lower by more than 0.01, dup_frac not higher by more than 0.02, vertices per 100 m (Douglas-Peucker) not higher by more than 1.0, junction F1 not lower by more than 0.03; CUT when topo F1 falls by more than 0.02 on both seeds or a guard breaks on both; TIE otherwise: **a TIE is kept and refined (D82)**, the follow-up card takes the TIE arm as its control. Arms are compared with each other by the head's own numbers (pair recall / precision, end recall / precision, order violations) and by topo F1; the arm that goes on is the one with the highest mean topo F1 (a tie inside 0.005: the one with fewer parts). Reported, not judged: lanes found, Q, topo P / R, junction P / R, the share of delivered pairs that violate the order rule (non-crossing inside one entry group x exit group), ms of the head, E284's numbers as a second comparison.

## 1. The head (`training/grid/jends.py`, tested by `python jends.py`)
Keypoints: the 3 x 3 peaks of the 1 m entry / exit heat of the junction maps (K = 32 per kind; in training the cells of the GT ends are always candidates: the D7 selection convention, the gradient flows through the values), position by the soft-argmax of the 3 x 3 window (differentiable in the heat), heading and probability at the peak. Tokens: the 1 m features of the maps head at the peak + position + heading + p, two layers of self-attention over the keypoints (as E284). Pair logit for every allowed (entry, exit) pair (`qends.allowed` rule) from [token a, token b, geometry (13), evidence (6), rank features (5)]. **Order features:** the entry / exit groups of the keypoints (`jgt._groups`: heading within 25 degrees, at most 6 m along and across) give each keypoint a rank (0 = leftmost) and a group size; the pair gets rank_in / (k_in - 1), rank_out / (k_out - 1), k_in, k_out and "same rank, same size" (the identity solution of D82). **Crossing penalty:** with P = sigmoid(logit) on the allowed pairs, the sum of P(i -> j) P(i' -> j') over the pairs whose ranks cross inside one pair of groups, normalised by the mass of P, added to the loss: soft, differentiable. **Evidence:** 12 points on the Hermite curve (tangent = chord), bilinear samples of the lane-centre probability of B (mean, minimum), the zone and core maps (mean), the stop-line and crossing maps (maximum). Loss: BCE (positive weight 3) + 0.5 x dual softmax with a 'none' entry (as E284) + the crossing penalty. Hard steps, named: the choice of the K strongest peaks (index choice), the group / rank bookkeeping (a function of the keypoint positions), the delivered pairs at p >= 0.5 and the connector curves (the read-out).
Not built here (the degree statistic comes first): a Sinkhorn assignment with a dustbin. A one-to-one transport is wrong where an entry feeds two connectors (27 % of the entries have more than one exit, 33 % of the exits more than one entry: E285 section 4); a capacitated version with learned row masses is the follow-up if the pair numbers say the competition is the limit.

## 2. Design
Two seeds, 3000 steps, lr 1e-3, batch 8, on the maps run of E286 with the best context arm (the base of E286, same seed), the threshold 0.5 and the keep integration of E284 (`qends_eval.add_connectors`) fixed before any VAL number of the head exists. VAL 873 frames; TEST untouched. Controls: c (plain graph), E284's h2 head (reported), the oracle numbers of E283 section 8 for what ends and pairs are worth.
**Prediction (a number):** pair recall 0.45 / precision 0.70 (E284: 0.31 / 0.64), end recall 0.70 (E284 0.58), topo F1 +0.015 over c (a TIE by the rule, E284 was -0.004: the head must not be worse, D82). A KEEP needs the head to beat the plain graph clearly: end recall above about 0.8 with few false ends.

## 3. Command and records
`wsl bash temp/e287/start_pairs.sh <p0|p1|p2> <seed>` (scratch wrapper around `run.py --require-card`, the exact command is in the run record), then `jends_eval.py dump --run e287_<arm>_s<S>` and `score --run e287_<arm>_s<S> --seed <S>`. Result files: `runs/e287_<arm>_s<S>/ends_head.json`, `ends_card.json`. GPU: nights 22 - 07 and weekdays 08:30 - 17:30 only.

## 4. Controls measured before the runs (2026-11-11, CPU, `evaluation/jrule_control.py`, the keypoints of the E284 head h2, VAL 873, 11083 true pairs)
The rule control (written in the file's docstring before the numbers; groups and ranks of the keypoints, a connected exit group per turn class, lane matching by normalised rank, extreme lanes for the turns) on the SAME keypoints as the E284 head:
| | pair recall | pair precision | pairs delivered |
|---|---|---|---|
| rule, seed 0 / seed 1 keypoints | 0.378 / 0.365 | 0.140 / 0.128 | 29948 / 31682 |
| E284 head (logit >= 0), same keypoints | 0.286 / 0.292 | 0.588 / 0.584 | 5399 / 5542 |
Reading: the rule delivers 5 - 6 times as many pairs and is wrong for 86 - 87 % of them: as a decision it is far behind the head (precision 0.13 against 0.59); but its recall 0.37 - 0.38 shows the CEILING of the keypoints: with the E284 end recall of 0.58 two ends of a pair are both found for about 0.34 - 0.38 of the true pairs, so the pair recall of the E284 head (0.29) is already close to what its keypoints allow. **Consequence for this card: the lever is the end recall (E286), not only the pair decision**; the head p0 / p1 / p2 is compared with the rule at equal keypoints (`jrule_control.py score --run e287_<arm>_s<S>`), and the rank features of p1 give the head the rule's information as features. The first temporal measurement is in E290 section 4 (pair flips 20 %, 52 - 54 % of the true pairs delivered in neither of two frames).
