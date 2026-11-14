# E346 - the ceiling of the one-graph code on all 873 VAL frames: B's node + link code against a clean control, where its loss sits, and whether the 1 m grid or node chains through the junction are options (CPU, no training)

Card of 2026-11-14 19:25, revised 19:45 after the first critic pass (REVISE: the main arm and its adapter exist already and were measured on a subset; the control was ambiguous; the chain code was under-specified; the long-link code was misdescribed) and 19:55 after the second (REVISE, narrow: the chain count was not decisive as defined; details of the breakdown and of arm g1). Written by the lane lead on Mirko's order of 2026-11-14 ("Plane 2 Experimente", after his question for a good, cheap, simple, end-to-end head). It is stage Q0 of the plan E344 (section 4), cut down to what is not known yet. Plan card: nothing is started; the lane experiments chat owns the code.

**Why:** the delivered system is two systems (lane head B and the junction layer) joined by a rule on the output (snap + one-piece wiring, about two thirds of the topology gain). A head that outputs ONE graph needs a code in which a junction connector is an edge between two nodes of the same node set, so the wiring is an index and not a search. That code exists: B's target code (`training/grid/qtargets.py`, E273) has nodes on the 2 m grid (at most one per cell, event points win), short links, and long links. **What is known:** `evaluation/qgraph_eval.py` already encodes the lane GT (`qformat.oracle_prediction`), reads it out (`readout`: wiring by node index) and scores it with the standard graph scorer (`graph_metrics`); `evaluation/tests/test_qgraph_eval.py` asserts topo F1 > 0.95 for it; the E273 card reports topo F1 0.97 for the oracle graph on every third VAL frame (291 frames, scratch of the tag `lcr/e273-scratch-report`), and the critic's hand check on 31 VAL frames gives topo F1 0.9684, F1@1.0 0.9977, lanes found 0.9918, junction F1 0.968 (by hand, not a run record). **What is not known:** (1) the number on all 873 frames against a control that separates the loss of the CODE from the mismatch of the two ground truths (the scorer's reference is the old graph GT, the code is built from the GtGenerator lane GT; on the 31 frames the undecoded lane GT scores 0.9854 against the old GT, so the code loses 0.017, not 0.032); (2) where the loss sits (lanes, short connectors, long connectors); (3) whether a 1 m grid buys anything; (4) whether node CHAINS through the junction (the code of E344 section 4) are possible with one node per cell. The stored counts do not settle it: with the connectors kept as nodes 18.5 % of the occupied 2 m cells would hold two nodes and 4 % three (E273 section 2.8); 18.4 % of the lane cells hold two or more pieces at 2 m and 12.1 % at 1 m, but only 3.7 % / 1.9 % hold CROSSING pieces (headings more than 30 degrees apart; E273 l.44), and two connectors that leave one entry side by side are one chain with a split, not a collision. **The value of this card** is therefore not the pass of l2, which is nearly known (losses of 0.017 / 0.008 / 0.002 against a bar of 0.03 on the 31 frames), but the full-frame confirmation with the clean control, the breakdown, the 1 m arm and a collision count with a definition that decides.

## 0. The card (what the tools run)
```json card
{
 "id": "E346",
 "card": "",
 "title": "representation ceiling of B's node + link code on the VAL ground truth (873 frames): the lane GT encoded by `qformat.oracle_prediction`, read out by `qgraph_eval.readout`, scored by `graph_metrics` against the old graph GT, on the 2 m grid (as it is) and on a 1 m grid, against the undecoded lane GT; plus the cell-collision count of a chain code at 2 m and 1 m",
 "kind": "block",
 "deterministic": true,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": -0.017, "vs": "g1"},
 "arms": {
  "g0": {"recipe": "harness check: the old graph GT scored against itself through `graph_metrics` (must give 1.000 on topo F1, F1@1.0 and lanes found)", "run_ids": ["E346.g0"]},
  "g1": {"recipe": "control: the UNDECODED lane GT polylines of GtGenerator (`frame.lane()`), edges = the lane GT's own `succ` relation (not a proximity join), scored against the old graph GT: the part of the loss that is the difference of the two ground truths, not the code", "run_ids": ["E346.g1"]},
  "l2": {"recipe": "the code as it is, 2 m grid: `qformat.oracle_prediction(frame.lane())` -> `readout` -> `graph_metrics` (nodes by location, at most one per cell, event points win their cell, connector points inside `in_intersection` suppressed; a link with a chord of at most 8 m (`SHORT`) is a Hermite curve with chord / 3 handles, a longer one is a long link with two handle scales fitted on the 6 x 6 grid 0.1 - 0.85)", "run_ids": ["E346.l2"]},
  "l1": {"recipe": "the same code on a 1 m grid: `N`, `RES` (module constants of `qtargets.py` today) as parameters, `SHORT` stays 8 m, the fit grid unchanged", "run_ids": ["E346.l1"]}
 },
 "control": "g1",
 "variant": "l2",
 "pairing": "position",
 "seeds": [0],
 "rule": [
  {"metric": "topo_F1", "vs": "g1", "max_loss": 0.03},
  {"metric": "lanes_found", "vs": "g1", "max_loss": 0.03},
  {"metric": "F1@1.0", "vs": "g1", "max_loss": 0.03}
 ],
 "budget_gpu_h": 0.0,
 "kill": "arm g0 does not score 1.000 on the three metrics, or the degraded copies of `evaluation/tests/test_qgraph_eval.py` do not fall, or `training/grid/test_qtargets.py` is not green with the cell size as a parameter at its default: then the harness is wrong and no arm is read",
 "evaluate": {"split": "val"}
}
```
**The rule in words (written before the run):** the code PASSES on a grid when the decoded GT loses at most 0.03 against the undecoded lane GT (arm g1) on topo F1, lanes found and F1@1.0, all three, VAL, 873 frames; deterministic, so no seeds: the numbers are exact for the stored GT. The absolute numbers are reported next to the losses. **Why 0.03:** it is about a tenth of the gap between the delivered system (snapped topo F1 0.70) and the ground truth; a code that loses less is not what limits a head. The bar is not trivial for the absolute number (the stored 0.97 against the old GT is a loss of 0.03 against arm g0) and the full-frame result may differ from the 31-frame hand check. **Choice, fixed here:** l2 when it passes (B's code, exists, fewest cells); l1 replaces it only when l2 fails and l1 passes; l1's gain over l2 is reported either way, and a gain below 0.01 topo F1 is written down as "the 1 m grid buys nothing for the code" (question 1 of E344 section 7). When neither passes no head is trained on the code: the breakdown names the part that loses and the code is changed first (a second slot per cell is the reserve the E273 card names).
**The chain code (E344 section 4) is not built in this card.** It is judged by a count on the GT first, per grid, with the connector points kept (no intersection suppression). **Definition:** a COLLISION CELL is a cell that holds points of two different pieces which cannot be one chain node: the two points are more than 0.3 m apart or their headings more than 30 degrees apart (pieces that run together after a shared entry or into a shared exit are one node and do not count). **The rule quantity** is the share of GT connectors that have at least one collision cell along them (a connector that loses a node is a broken or bridged chain). **Rule:** above 5 % of the connectors the chain code is rejected on that grid without being built; at or below 5 % on a grid the chain code gets its OWN card, which must specify what this one leaves open (the number and layout of the successor slots, two successors at a split, whether the losing line of a shared cell is bridged or cut, pieces without section ids, the curve between chain nodes). Reported next to it, with occupied cells as the denominator: the raw share (two or more pieces in a cell) and the crossing share (more than 30 degrees), which must reproduce E273's 18.4 % / 3.7 % at 2 m and 12.1 % / 1.9 % at 1 m within a point (its window and GT version may differ: a difference is reported, not hidden).
**Prediction (numbers):** l2 loses 0.017 topo F1 against g1 (the 31-frame hand check) and passes; l1 loses 0.012 (a guess); connectors with a collision cell: about 30 % at 2 m (E273: 31 % of the connectors collide mid-length with another piece) and about 15 % at 1 m (a guess: crossing connectors cross whatever the cell size, only the side-by-side cases go away): the chain code is rejected on both grids.
**Reported with the arms (where the code loses):** (a) the three rule metrics, junction F1 and vertices per 100 m, absolute and against g1 and g0; (b) F1@1.0 and lanes found split into lane and connector parts: by segment on the lane-GT side (a third of its pieces mix lane and connector segments) and by `kind` on the old-GT side; topo F1 is read for the whole graph only; (c) connectors split into short (chord at most 8 m, chord / 3 handles) and long (fitted scales): the distance of the decoded curve to its GT polyline, median and 90th percentile; (d) the share of GT connector length that is suppressed inside `in_intersection` and comes back only as a link; (e) the outputs per cell a head has to predict, and the CPU time of the read-out.

## 1. Card text
- **Statement:** B's node + link code on the 2 m grid reproduces the lane GT graph within 0.03 on all 873 VAL frames; node chains through the junction are not possible with one node per cell on either grid.
- **Mechanism:** a link has no nodes inside the junction, so crossing connectors never compete for a cell; a chain puts a node into every cell a connector crosses.
- **Inputs:** ready: `evaluation/qgraph_eval.py` (oracle prediction, read-out, scorer, the 873 VAL frames with both ground truths), `training/grid/qtargets.py`, `qformat.py`, the tests named in the kill line. To build (scratch `temp/e346/`): arm g1 (the lane GT polylines as a `LaneGraph`), the cell size as a parameter of `qtargets.py` and its pass-through in `qformat.oracle_prediction` (defaults unchanged), the breakdown (b) - (d), the collision count. No new metric.
- **Starts from:** `qgraph_eval.py` and `qtargets.py` at HEAD, the stored VAL GT.
- **Cost:** CPU only, 2 threads (shared box), no GPU, no checkpoint. Each arm one `run.py` job without `--gpu`. Estimate: a day of code (the 1 m parameter and the breakdown), minutes per arm.
- **What it decides:** whether the one-graph head predicts B's code as it is (2 m, links), and that the chain variant of E344 is closed or gets its own card. It does not say that a head can LEARN good links: B's learned long links are the known weak point (junction F1 0.13 against 0.63 for the oracle with the head's node recall, E273), which is why the junction layer exists; E347 takes the first step of that question (the ends on the node grid).
- **Not claimed:** anything about a trained model; anything on TEST.
- **D88:** a measurement on the ground truth, no step added to any delivered path.
- **Lens / status:** structure (one graph, one head), D86 weights; status proposed.
- **Referee:** first pass 2026-11-14 (fresh `ar-critic`, mode cards): REVISE, seven findings, all taken into this revision (the existing scorer and numbers cited, two controls g0 / g1 and the rule against g1, code C not built but judged by a count, code L described from `qtargets.py`, the stored 18.5 % cited, the kill on the right tests). Second pass 2026-11-14 (fresh critic): REVISE, narrow; arms g0 / g1 / l2 / l1 hand-over ready (it re-ran 31 frames: g0 1.0000, g1 0.9854, l2 0.9684 topo F1), the chain count not: taken into this revision (collision defined, the rule on connectors, the stored 1 m numbers cited, breakdown by segment / kind, g1 edges = `succ`, the `qformat` pass-through). Before the runs: `ar-critic` mode code on the diff of `qtargets.py` / `qformat.py` (B's target code: the default must stay bit-identical).

## 2. Detail plan
| step | what exactly | where | done when | budget |
|---|---|---|---|---|
| code | arm g1; cell size parameter; breakdown; collision count | `temp/e346/`, `training/grid/qtargets.py` (parameter only) | the kill checks pass | CPU, a day |
| run | arms g0, g1, l2, l1 | `run.py --id E346.<arm> --require-card --split val` (no `--gpu`) | done flag, metric card per arm | minutes each |
| decide | the rule above from the metric cards; the chain rule from the count | `decide.py --experiment E346`, else "by hand: GT replay" with the reason | verdict in the ledger; the chosen code in HANDOVER.md and E344 section 4 | - |

## 3. Summary
(empty until the runs exist)
