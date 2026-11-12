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
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.02},
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

## 4. Controls measured before the runs (2026-11-11, recomputed 2026-11-12 after the critic's findings; CPU, `evaluation/jrule_control.py`, the keypoints of the E284 head h2, VAL 873, 11083 true pairs)
The rule control (written in the file's docstring before the numbers; groups and ranks of the keypoints, a connected exit group per turn class, lane matching by normalised rank, extreme lanes for the turns) on the SAME keypoints as the E284 head. Matching of keypoints to GT ends is STRICT one-to-one (4 m, 30 degrees; a keypoint serves one GT end only) and the rule sees only the keypoints the head also sees (probability >= the threshold); the first version of this section (11 Nov: rule recall 0.37 / precision 0.13) let the rule use all keypoints and counted a GT pair as found when ANY keypoint near the end took part, which flattered the rule's recall and understated its precision.
| keypoint threshold | seed | ceiling: both ends of a pair found | rule recall / precision (pairs) | E284 head (logit >= 0) recall / precision (pairs) |
|---|---|---|---|---|
| 0.5 | 0 | 0.429 | 0.357 / 0.337 (11733) | 0.289 / 0.741 (4329) |
| 0.5 | 1 | 0.417 | 0.346 / 0.332 (11583) | 0.294 / 0.746 (4370) |
| 0.3 | 0 | 0.516 | 0.403 / 0.263 (16988) | 0.311 / 0.680 (5061) |
| 0.3 | 1 | 0.510 | 0.392 / 0.259 (16789) | 0.321 / 0.678 (5243) |
Reading: the rule delivers 2.7 - 3.4 times as many pairs and is wrong for two thirds of them (precision 0.33 against 0.74): as a decision it is far behind the head. But its recall (0.35 - 0.40) is higher than the head's (0.29 - 0.32) at the same keypoints, and the CEILING (both ends of the pair found) is 0.42 - 0.52: the head delivers 60 - 70 % of what its keypoints allow, the rule 78 - 83 %. **Two consequences for this card: (1) the keypoints (end recall, E286) are the first lever: the ceiling is 0.42 at threshold 0.5; (2) the pair decision still leaves 0.12 - 0.20 of the pairs on the table at logit >= 0, which is what the precision-recall curve (`jpair_pr.py`, E302) shows and what the rank features of p1 / p2 give the head the information for.** The head p0 / p1 / p2 is compared with the rule at equal keypoints (`jrule_control.py score --run e287_<arm>_s<S>`). The first temporal measurement is in E290 section 4 (pair flips 20 %, 52 - 54 % of the true pairs delivered in neither of two frames).

## 5. Result (2026-11-12; the final checkpoints, VAL 873, scored by `jends_eval.py score` against the plain delivered graph of the same seed; `runs/e287_<arm>_s<S>/ends_head.json` / `ends_card.json`; p2 seed 1 was dumped on the CPU because the GPU dump died with "Cannot allocate memory", the scored numbers do not depend on it)
| arm | seed | entry / exit recall | pair recall / precision (logit >= 0) | topo F1 (vs plain) | F1@1.0 (vs plain) | lanes found (vs plain) | junction F1 (vs plain) | junction recall (vs plain) | dup_frac (vs plain) |
|---|---|---|---|---|---|---|---|---|---|
| p0 | 0 | 0.652 / 0.675 | 0.141 / 0.549 | 0.6275 (+0.0016) | 0.8343 (+0.0059) | 0.6440 (+0.0173) | 0.2638 (-0.0075) | 0.6127 (+0.0573) | 0.2046 (-0.0299) |
| p0 | 1 | 0.669 / 0.678 | 0.181 / 0.598 | 0.6235 (-0.0014) | 0.8242 (+0.0083) | 0.6358 (+0.0222) | 0.2551 (+0.0058) | 0.6009 (+0.1004) | 0.1732 (-0.0281) |
| p1 | 0 | 0.652 / 0.675 | 0.182 / 0.602 | 0.6313 (+0.0055) | 0.8349 (+0.0066) | 0.6497 (+0.0230) | 0.2632 (-0.0080) | 0.6185 (+0.0630) | 0.2037 (-0.0308) |
| p1 | 1 | 0.669 / 0.678 | 0.186 / 0.599 | 0.6267 (+0.0018) | 0.8252 (+0.0093) | 0.6375 (+0.0238) | 0.2588 (+0.0094) | 0.6076 (+0.1070) | 0.1771 (-0.0241) |
| p2 | 0 | 0.652 / 0.675 | 0.158 / 0.598 | 0.6320 (+0.0061) | 0.8353 (+0.0070) | 0.6466 (+0.0199) | 0.2594 (-0.0119) | 0.5992 (+0.0438) | 0.2108 (-0.0237) |
| p2 | 1 | 0.669 / 0.678 | 0.177 / 0.638 | 0.6286 (+0.0037) | 0.8267 (+0.0107) | 0.6402 (+0.0265) | 0.2521 (+0.0028) | 0.5866 (+0.0861) | 0.1795 (-0.0218) |
Plain graph (control, seed 0): topo F1 0.6259, F1@1.0 0.8283, lanes found 0.6267, junction F1 0.2712, junction recall 0.5554, dup_frac 0.2345, 19.28 vertices / 100 m.

**Verdict by the rule above: TIE on all three arms** (topo F1 rises by +0.0016 / -0.0014 (p0), +0.0055 / +0.0018 (p1), +0.0061 / +0.0037 (p2), the +0.02 of the rule is not reached on any seed; no guard breaks: junction F1 loses at most 0.012, F1@1.0 gains, dup_frac falls). Kept and refined (D82). The arm that goes on by the written rule: mean topo F1 over the seeds p2 +0.0049, p1 +0.0037, p0 +0.0001 over the plain graph, p2 and p1 differ by 0.0013, inside the 0.005 tie, so the one with fewer parts: **p1**. By hand: the follow-up cards written before this result (E298, E302, E303, E305, E308) build on p2 (the full head: they need the evidence input and the cost of an arm is the same); that is kept, and p1 is the fallback if a follow-up shows the evidence to be the problem. What the arms say: the order features help (p1 over p0: +0.004 / +0.003 topo F1, +0.04 pair recall at the same precision); the evidence along the curve does not add to it (p2 against p1: +0.0006 / +0.0019, inside the band). All three deliver the same gain over the plain graph, and it is of the SAME kind: lanes found +0.02 to +0.027, junction recall +0.04 to +0.11, duplicates -0.022 to -0.031, topo F1 +0.00 to +0.006: the connectors are right where they are delivered, but few are delivered.

**The weak spot is the pair decision, not the ends.** Pair precision at the recall of the E284 head (`jpair_pr.py score`, same VAL frames, 11083 true pairs): p0 0.524 / 0.559, p1 0.566 / 0.579, p2 0.568 / 0.598 against the E284 head 0.741 / 0.746 (seed 0 / seed 1; gain -0.148 to -0.217). The strict control has not changed (rule recall 0.45 / precision 0.33 on the keypoints of these runs; ceiling both ends found 0.528 / 0.533). The AUC of the pair logit among candidates with BOTH ends matched to GT ends (`temp/gt2/pair_diag.py`, 20249 candidates, 5798 true): p0 0.912, p1 0.926, p2 0.934 (seed 0 / 1: 0.934 for p2 s1) against 0.974 for the E284 head on its own keypoints: the new head ranks pairs of true ends worse than the old one although it sees better ends. E308 (node tokens of B) was built for this; E302 (capacity) attacks the other half (a degree budget per end).

Cost: 41 - 44 minutes per run, two lanes in parallel on the GPU. Ledger rows E287.<arm>_s<S>: result and "TIE by rule of this card" in the notes. Rows of the arm q2 (the head on the joint maps of E286 jj) do not exist: jj was stopped, see the E286 card section 8.
