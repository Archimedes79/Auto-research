# E356 - node spacing (D104): the spacing of the delivered nodes and the collapse of a post-hoc radius, and the GT-coding ceiling of B's node + link code on a 2.5 m and a 3.125 m grid (CPU, no training)

Card of 2026-11-15 19:55, version 2 of 20:00 after the first critic pass (REVISE: part A had been measured by E275 section 7 and the mechanism of version 1 was false: B's nodes sit at sub-cell positions, so 'neighbours at exactly 2.0 m survive radius 2.0' is wrong; part B did not test D104's constraint, because one node per cell at the true position gives no minimum spacing; the rule had holes (lanes found unguarded, 'taken' undefined, dup_frac direction); windows too wide; E346's open item not carried over; all taken below). Written by the lane experiments chat on D104 (the owner 2026-11-15 about 19:30: "Don't set nodes closer than 2 m, consider 2.5 m ..."; `experiment_execution/decisions/D104.md` point 1: measure first, on the CPU, without training). **What is known already (E275 section 7, quoted, not re-argued):** a close-node radius of 2.0 m on B's plain graph (thr 0.5) takes topo F1 from 0.626 / 0.625 to 0.238 / 0.220 and node recall to 0.545 / 0.549; 2.5 m (thr 0.4, E274 / E275) 0.204: the node code collapses. **Why (this card, measured on the stored B dumps, `temp/e356/nn_spacing.py`, VAL 873 frames):** B puts at most one node into each 2 m cell at its true sub-cell position, so neighbouring nodes are about 2 m apart and often closer: after the delivered step (`close:1.5@0.5`) a node has another node closer than 2.0 m in 70 % / 69 % of the cases (seed 0 / 1), closer than 2.5 m in 93 % / 92 %, the nearest-neighbour distance has the quartiles 1.94 / 1.99 / 2.00 m (median 1.99); the 156 nodes per frame become 106 at radius 2.0 (-32 %) and 82 at 2.5 (-48 %) with the nearest-neighbour distance never below 2.00 / 2.68 m. A read-out cannot give a spacing that the links, which join neighbouring cells, do not know about: B's short links connect nodes about 2 m apart; once every second node is gone the survivors are 3 - 4 m apart and no link joins them. **So a minimum node spacing has to be a property of the node CODE, not of the read-out.** That is what part B measures: the ceiling of the decoded GT on coarser grids, together with the spacing statistics of the decoded nodes. D86 / D88 hold: nothing is trained, no step is added. **Not claimed:** that a trained B predicts the coarser code (a trained arm is a later card); anything on TEST; anything on ms.

## 0. The card (what the tools run)
```json card
{
 "id": "E356",
 "card": "",
 "title": "(A) the nearest-neighbour spacing of B's delivered nodes and the E275 collapse reproduced on the stored B dumps (close-node radius 2.0 and 2.5 m instead of 1.5 m: B plain graph and the E324 s snapped read-out, VAL 873 frames, both seeds, with the vertices per 100 m); (B) the GT-coding ceiling of B's node + link code on a 2.5 m and a 3.125 m grid (E346 tool, VAL 873 frames) against the 2 m result, with the spacing statistics of the decoded nodes",
 "kind": "variant",
 "deterministic": true,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": -0.025, "vs": "c"},
 "arms": {
  "c": {"recipe": "control (A) = the delivered read-out `close:1.5@0.5` of E324 s as stored (`runs/e324_s_s<S>/ends_card_snap_thr0.1.json`: snapped topo F1 0.6981 / 0.7010, B plain graph = `numbers.base` 0.6259 / 0.6249, `vert100_dp` 18.97 / 19.28 in the same cards); control (B) = E346 g1 (undecoded lane GT) and l2 (2 m grid): `runs/eval/e346_g1/val.json`, `e346_l2`; the tagged-score path is checked by ONE recomputation of the control, `E356.a15chk_snap_s0` (`--spec close:1.5@0.5 --tag c15`), which must reproduce the stored snapped card to 2e-4 topo F1 on base and head"},
  "a20": {"recipe": "(A) `jends_eval.py score --run e324_s_s<S> --seed <S> --spec close:2.0@0.5 --tag c20 --snap --thr 0.1` on the stored B dump (the default `--full`): B's node set is thinned with radius 2.0 m (strictly closer than 2.0 m) before B's graph and the connectors are read out (the attached ends and keypoints of the junction head come from the model dump and are not thinned); the card of B plain is `numbers.base` of this card", "run_ids": ["E356.a20_snap_s0", "E356.a20_snap_s1"]},
  "a25": {"recipe": "(A) the same with `--spec close:2.5@0.5 --tag c25`; plus the control recomputation", "run_ids": ["E356.a25_snap_s0", "E356.a25_snap_s1", "E356.a15chk_snap_s0"]},
  "sp": {"recipe": "(A) `temp/e356/nn_spacing.py`: nodes per frame, the share of kept nodes with a neighbour closer than 2.0 / 2.5 m, the quartiles of the nearest-neighbour distance of the stored B nodes at e >= 0.5 and after `close:1.5 / 2.0 / 2.5` (a plain count on the stored dumps, run once, numbers in the Reading below)", "run_ids": ["E356.nn_spacing"]},
  "l25": {"recipe": "(B) the E346 driver with a 2.5 m cell (N = 40): the lane GT encoded by `qformat.oracle_prediction`, read out by `qgraph_eval.readout`, scored by `graph_metrics` against the old graph GT, `SHORT` = 8 m, the fit grid unchanged; `e346_arm.py l25`; the node spacing columns (`nn_share_lt20`, `nn_share_lt25`, quartiles) and `dropped_candidates_per_node` are in its card", "run_ids": ["E356.l25"]},
  "l31": {"recipe": "(B) the same with a 3.125 m cell (N = 32): plain 3.0 m is refused by the tool (it does not divide the 100 m window); 3.125 m is 4 % COARSER than 3.0 m, so slightly harder: labelled so; E355 (card version 3) has no 3.0 m arm any more: its v31 / s31 arms are 3.125 m and the dense reference of this lattice is this arm l31", "run_ids": ["E356.l31"]}
 },
 "control": "c",
 "variant": "a25",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.01},
  {"metric": "lanes_found", "vs": "c", "max_loss": 0.03},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.01},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 0.0,
 "kill": "(A) the recomputation of the control (`E356.a15chk_snap_s0`) differs from the stored snapped card by more than 2e-4 topo F1 on the plain graph or the head (the tagged-score path is not the stored one); or a card is not made from 873 frames; (B) the E346 kill line: arm g0 does not score 1.000, the self-test of the driver fails, the cell size is not seen by every consumer (`check_cell_size`), or the defaults of qtargets (2 m) are not restored after the arm",
 "evaluate": {"split": "val"}
}
```
**The rule (written before the run; by hand: replay of the cards with `temp/e356/e356_table.py`; `dup_frac` is read as 'rises by at most', not as a loss):** **(A) A measurement, with a pre-registered reading:** the E275 collapse is CONFIRMED when the radius 2.0 m and 2.5 m lose more than 0.10 topo F1 on B's plain graph (`numbers.base`) AND on the E324 s snapped numbers (`numbers.head`) on both seeds; then 'a post-hoc radius is not a way to a minimum spacing' and the spacing has to come from the code. If a radius passes the BARS of the old read-out rule (topo F1 and lanes found lose at most 0.01 / 0.03, dup_frac rises at most 0.01, junction F1 loses at most 0.03, both seeds, both numbers sets) AND `vert100_dp` of E324 s snapped falls by at least 0.5 (mean of the seeds), it is a CANDIDATE for the delivered read-out: 'candidate' means recorded and run through the stored tools with `--spec` (the TEST freeze of E343 covers `qgraph_readout.py` and `qgraph_eval.py`: no frozen file is edited, and nothing changes in the delivered system by this card). **(B)** the E346 rule: a grid PASSES the code ceiling when the decoded GT loses at most 0.03 against the undecoded lane GT (g1) on topo F1, lanes found and F1@1.0, all three, VAL, 873 frames; reported next to it for l25 and l31 against l2 (topo F1 0.9669, lanes found 0.9853, F1@1.0 0.9957, junction F1 0.9535, dup_frac 0.0223, vertices per 100 m after Douglas-Peucker 11.45 (raw 33.1), 180 nodes and 40 000 outputs per frame): the differences, junction F1, dup_frac, `vert100` raw and `vert100_dp`, nodes per frame, outputs per frame (cells x 16 channels), the collision shares, the read-out time, **the spacing of the decoded nodes (share with a neighbour closer than 2.0 m and 2.5 m, the quartiles of the nearest-neighbour distance) and `dropped_candidates_per_node` (E346's open item: 1.11 % at 2 m against a 1 % gate; `test_qtargets.py` is red at HEAD for a reason that is not this card's: named in E346 section 4)**. A grid answers D104's constraint only if the share of decoded nodes with a neighbour closer than 2.0 m (2.5 m) is small: the table says it. **Vertex count next to every accuracy number: every table has the `vert100_dp` column.**

**D106 line (the owner 2026-11-15 ~19:54, `decisions/D106.md`: at similar performance (the band 0.01 topo F1 on both seeds and read-outs, guards held) the system that is simpler AND faster is chosen without asking again; simpler-but-slower or faster-with-more-parts stays the owner's case):** every ADOPT / TIE row of this card states explicitly 'simpler AND faster?' with its four facts: the parts (named selections / parts: E356 changes no system: part A is a read-out measurement (a radius is a candidate only; the delivered read-out stays unless a radius passes the bars AND the vertex bar), part B is a ceiling measurement of a code), the parameters, the GMACs and the quiet-host ms (paired, with its interval; 'not above' = the upper end of the interval is not above 0); the card applies D106 by this written rule and does not decide a borderline case itself.

## 1. Card text
- **Statement:** (A) a post-hoc close-node radius of 2.0 / 2.5 m collapses B's graph (the E275 result, reproduced on E324 s) because B's nodes are about 2 m apart by construction; (B) B's node + link code reproduces the lane GT within 0.03 on a 2.5 m grid and probably not on a 3.125 m grid, and a coarser grid reduces, but does not remove, the share of nodes with a close neighbour.
- **Mechanism (B):** E346: the code loses in the connectors and the junction (junction F1 -0.046 at 2 m, lanes exact); a coarser grid puts more connector points into one cell (the collision share 59.7 % at 2 m) and fewer cells per metre, so the connector part loses more; the cost of going from 1 m to 2 m was 0.0025 topo F1, so 2.5 m should cost a few thousandths more.
- **Inputs (ready):** the stored B dumps `temp/e276/predfull_e273_p_s<S>.pkl` (scratch input of the stored scorer, not in git, present), the stored E324 s dumps, `jends_eval.py score --spec / --tag` (E292 mechanism; a tagged card is a NEW file in the stored run folder, as in E348 stage 0, never overwritten, `spec` and `tag` are fields of the card), `qgraph_readout.py`, the E346 driver `temp/e346/e346_arm.py` / `e346_lib.py` (scratch of E346; arms l25 / l31 added by a coder, the spacing columns added; second use: its promotion into `evaluation/` with a test is a named cleanup item, to be done before a third use (E355 step 0)), `temp/e356/e356_table.py`, `temp/e356/nn_spacing.py`.
- **Prediction (numbers; guesses; windows):** (A) topo F1 of the BASE graph at radius 2.0: 0.24 (window 0.20 to 0.30), at 2.5: 0.20 (window 0.15 to 0.25) (E275: 0.238 / 0.220 at 2.0 and 0.204 at 2.5 with thr 0.4); the E324 s snapped topo F1 loses 0.35 at radius 2.0 (window -0.45 to -0.25) and 0.40 at 2.5; `vert100_dp` of E324 s snapped falls (a thinner graph) but the bars are missed; nodes per frame 106 / 82 (seed 0; measured by the spacing script: 105.6 / 81.8 and 106.6 / 82.4); the share of nodes with a neighbour closer than 2.0 m: 0 after radius 2.0 by construction. (B) l25: topo F1 -0.025 against g1 (window -0.032 to -0.020), lanes found -0.018 (window -0.025 to -0.010), F1@1.0 -0.006, junction F1 -0.065; l31: topo F1 -0.030 (window -0.040 to -0.022: at the bar), junction F1 -0.085; nodes per frame about 144 / 115 (l25 / l31), outputs per frame 25 600 / 16 384; the share of decoded nodes with a neighbour closer than 2.0 m: l2 about 0.70 (like the delivered graph), l25 0.30 (window 0.15 to 0.50), l31 0.10 (window 0.03 to 0.25); closer than 2.5 m: l25 0.75, l31 0.30.
- **Power:** deterministic replays of the same stored predictions or of the stored GT: the only noise is scene sampling (22 scenes, paired half-width about 0.004 for a topo F1 difference of this kind, E348); the effects predicted in (A) are two orders of magnitude larger; in (B) the bar is 0.03 and the predictions sit 0.003 - 0.01 from it, so a pass or fail at l25 / l31 is read with that margin.
- **In-sample:** nothing is chosen on VAL beyond the two radii and two grids of D104; no number of this card feeds TEST.
- **D55 / D86 / D88:** no step is added; the close-node radius is an existing constant of a named step. **Referee:** first pass 2026-11-15 19:50: REVISE (above), taken in version 2.

## 2. Detail plan
| step | what exactly | where | done when | budget |
|---|---|---|---|---|
| spacing | `nn_spacing.py` on both stored B dumps (done once 19:45: numbers in section 1) | `run.py` job `E356.nn_spacing` (CPU) | table | 2 CPU min |
| control | `E356.a15chk_snap_s0` (the tagged path reproduces the stored snapped card to 2e-4) | `run.py` job, CPU, nice 19 | the kill of (A) | 5 CPU min |
| A | 4 scoring jobs: radius 2.0 / 2.5 x seeds 0 / 1, snapped | `run.py` jobs (CPU, nice 19, one stream) | tagged cards `runs/e324_s_s<S>/ends_card_c20_snap_thr0.1.json`, `c25_*` | 4 x 4.5 min |
| B | `e346_arm.py l25` and `l31` (full 873 frames) | `run.py` jobs (CPU, nice 19) | `runs/eval/e346_l25/val.json`, `e346_l31` | 2 x 5 min |
| decide | `e356_table.py`: the tables with vertices and spacing, the rule of section 0 | by hand: replay | result section, ledger notes, HANDOVER | - |

## 3. Summary (2026-11-15 21:16; CPU replays and GT replays, no training; by hand: replay of the cards with `temp/e356/e356_table.py`; chain `temp/day31/day31.sh`, run ids `E356.*`; VAL 873 frames, both seeds in part A)
**Result: (A) a post-hoc close-node radius of 2.0 or 2.5 m collapses B's graph and the system of record (the E275 result, reproduced to three decimals); (B) B's node + link code reproduces the lane GT on a 2.5 m and on a 3.125 m grid almost as well as on 2 m (both PASS the E346 bar; the loss against 2 m is 0.001 - 0.002 topo F1, the junction F1 falls by 0.007 / 0.017), but NEITHER grid gives a minimum spacing: 36 % (2.5 m) and 24 % (3.125 m) of the decoded GT nodes still have a neighbour closer than 2.0 m (92 % at 2 m).** D104's constraint ('no nodes closer than 2 m') is therefore not met by a coarser grid; it has to be a property of the node set itself (a spacing term, a merge or cell-centre positions: E355's arms m / c).
**(A) The read-out radius (the stored B dumps, `close:R@0.5`, B plain = `numbers.base`, E324 s snapped = `numbers.head`; vertices per 100 m after Douglas-Peucker next to every number):**

| seed | numbers set | radius | topo_F1 | lanes_found | F1@1.0 | dup_frac | junction_F1 | vert100_dp |
|---|---|---|---|---|---|---|---|---|
| 0 | B plain | 1.5 | 0.6259 | 0.6267 | 0.8283 | 0.2345 | 0.2712 | 19.28 |
| 0 | B plain | 2.0 | 0.2377 | 0.2182 | 0.6010 | 0.1928 | 0.2352 | 28.67 |
| 0 | B plain | 2.5 | 0.1603 | 0.1600 | 0.3993 | 0.2347 | 0.1520 | 22.54 |
| 0 | E324 s snap | 1.5 | 0.6981 | 0.7604 | 0.8762 | 0.1781 | 0.3238 | 18.97 |
| 0 | E324 s snap | 2.0 | 0.4041 | 0.4471 | 0.7421 | 0.1678 | 0.3869 | 23.71 |
| 0 | E324 s snap | 2.5 | 0.3244 | 0.3892 | 0.6210 | 0.2376 | 0.3041 | 19.51 |
| 1 | B plain | 1.5 | 0.6249 | 0.6137 | 0.8159 | 0.2012 | 0.2493 | 20.08 |
| 1 | B plain | 2.0 | 0.2195 | 0.1994 | 0.5800 | 0.1658 | 0.2090 | 30.77 |
| 1 | B plain | 2.5 | 0.1389 | 0.1425 | 0.3637 | 0.2010 | 0.1274 | 24.79 |
| 1 | E324 s snap | 1.5 | 0.7010 | 0.7594 | 0.8763 | 0.1522 | 0.3232 | 19.63 |
| 1 | E324 s snap | 2.0 | 0.3990 | 0.4420 | 0.7428 | 0.1530 | 0.3985 | 24.46 |
| 1 | E324 s snap | 2.5 | 0.3118 | 0.3769 | 0.6111 | 0.2239 | 0.3302 | 20.34 |

The control recomputation (`E356.a15chk_snap_s0`, the tagged-score path with `--spec close:1.5@0.5 --tag c15`) reproduces the stored snapped card with a largest difference of 0.0 (kill row passed). **Reading, pre-registered: the E275 collapse is CONFIRMED** on all four (seed x numbers set) cells at both radii (every cell loses more than 0.10 topo F1): B plain topo F1 0.626 / 0.625 -> 0.238 / 0.220 at 2.0 m (E275: 0.238 / 0.220, identical) and 0.160 / 0.139 at 2.5 m; the E324 s snapped read-out 0.698 / 0.701 -> 0.404 / 0.399 and 0.324 / 0.312 (the junction connectors are snapped to a graph whose lanes are broken). The vertices per 100 m do NOT fall (B plain 19.3 -> 28.7 at 2.0 m: a broken graph has more short pieces), so neither radius is a candidate read-out; the delivered read-out stays `close:1.5@0.5`. **Why (measured on the stored B dumps, `temp/e356/nn_spacing.py`):**

| seed | read-out | nodes per frame (mean / p95 / max) | share with a neighbour < 2.0 m | share < 2.5 m | NN distance quartiles 25 / 50 / 75 % (m) |
|---|---|---|---|---|---|
| 0 | e >= thr (no step) | 172.2 / 298 / 344 | 0.772 | 0.961 | 1.80 / 1.97 / 2.00 |
| 0 | close:1.5 | 156.5 / 277 / 300 | 0.698 | 0.926 | 1.94 / 1.99 / 2.00 |
| 0 | close:2.0 | 105.6 / 183 / 205 | 0.000 | 0.576 | 2.01 / 2.09 / 3.23 |
| 0 | close:2.5 | 81.8 / 141 / 159 | 0.000 | 0.000 | 3.13 / 3.68 / 3.97 |
| 1 | e >= thr (no step) | 173.3 / 309 / 343 | 0.767 | 0.959 | 1.79 / 1.97 / 2.00 |
| 1 | close:1.5 | 157.3 / 282 / 297 | 0.690 | 0.924 | 1.93 / 1.99 / 2.00 |
| 1 | close:2.0 | 106.6 / 189 / 219 | 0.000 | 0.581 | 2.01 / 2.08 / 3.13 |
| 1 | close:2.5 | 82.4 / 144 / 163 | 0.000 | 0.000 | 3.05 / 3.67 / 3.97 |

B's nodes sit at sub-cell positions, one per 2 m cell, so about 70 % of the delivered nodes have a neighbour closer than 2.0 m and the median nearest-neighbour distance is 1.99 m; radius 2.0 removes a third of the nodes (156 -> 106 per frame), 2.5 removes half (82), and B's short links, which join cells about 2 m apart, no longer find a partner.
**(B) The GT-coding ceiling on a coarser grid (E346 tool, arms l25 / l31 new, l2 = the stored E346 card; the decoded lane GT against the old graph GT; g1 = the undecoded lane GT):**


| arm | topo_F1 | lanes_found | F1@1.0 | junction_F1 | dup_frac | vert100 | vert100_dp |
|---|---|---|---|---|---|---|---|
| g1 | 0.9858 | 1.0000 | 1.0000 | 0.9995 | 0.0063 | 10.3334 | 10.3334 |
| l2 | 0.9669 | 0.9853 | 0.9957 | 0.9535 | 0.0223 | 33.0697 | 11.4489 |
| l25 | 0.9656 | 0.9836 | 0.9953 | 0.9467 | 0.0213 | 27.0009 | 11.4147 |
| l31 | 0.9646 | 0.9791 | 0.9944 | 0.9363 | 0.0208 | 21.7635 | 11.3771 |
| l1 | 0.9693 | 0.9899 | 0.9969 | 0.9670 | 0.0241 | 62.4309 | 11.3520 |
**Pass rule (loss against g1 at most 0.03 on topo F1, lanes found, F1@1.0): l2 -0.0190 / -0.0146 / -0.0043 PASS; l25 -0.0202 / -0.0163 / -0.0047 PASS; l31 -0.0212 / -0.0209 / -0.0056 PASS.** The raw vertices per 100 m fall with the grid (33.1 -> 27.0 -> 21.8), the Douglas-Peucker vertices (the delivered number) do not change (11.45 / 11.41 / 11.38): a coarser grid is not a route to fewer delivered vertices. **Sizes and spacing of the decoded nodes:**

| arm | nodes per frame (mean / p95 / max) | outputs per frame | share of decoded nodes with a neighbour < 2.0 m / < 2.5 m | NN median (m) | dropped candidates per node (E346 gate 1 %) | raw collision share of occupied cells | events per frame / merged | read-out ms |
|---|---|---|---|---|---|---|---|---|
| l2 | 180.5 / 311 / 346 | 40000 | 0.923 / 0.949 | 1.98 | 1.11 % | 0.149 | 24.4 / 0.85 | 1.72 |
| l25 | 146.9 / 254 / 280 | 25600 | 0.356 / 0.908 | 2.48 | 1.33 % | 0.168 | 24.1 / 1.13 | 1.23 |
| l31 | 117.8 / 204 / 221 | 16384 | 0.241 / 0.331 | 2.97 | 2.61 % | 0.199 | 23.7 / 1.50 | 0.99 |

Reading: the coarser grid saves nodes (-19 % / -35 % per frame; p95 and maximum 311 / 346 -> 254 / 280 -> 204 / 221, the sizes to dimension a token set) and outputs per frame (-36 % / -59 %) at 0.001 - 0.002 topo F1, but the share of nodes with a close neighbour falls only to 36 % / 24 % (below 2.0 m) and the median nearest-neighbour distance is the cell size minus a little (1.98 / 2.48 / 2.97 m): **no grid gives the guarantee**; the dropped candidates per node (E346's open item: 1.11 % against a gate of 1 %) are 1.11 % / 1.33 % / 2.61 %: the code at 3.125 m drops twice as many candidate nodes into occupied cells (the gate is exceeded on all three grids; the test of E346 stays red for the reason named there); events merged per frame rise 0.85 / 1.13 / 1.50 (an event place sharing its cell with another loses its own node).
**Predictions (written before; hit windows):** (A) B plain at 2.0 m 0.24 (window 0.20 to 0.30): HIT (0.238 / 0.220, both inside); at 2.5 m 0.20 (0.15 to 0.25): HIT on seed 0 (0.160), seed 1 just below the window (0.139); E324 s snapped loss at 2.0 m -0.35 (-0.45 to -0.25): HIT (-0.294 / -0.302); nodes per frame 106 / 82: HIT. (B) l25: topo F1 -0.025 against g1 (-0.032 to -0.020): HIT at the edge (-0.0202); lanes found -0.018 (-0.025 to -0.010): HIT (-0.0163); l31 topo F1 -0.030 (-0.040 to -0.022): MISSED on the good side (-0.0212); junction F1 predicted -0.065 / -0.085: smaller (-0.053 / -0.063); share of decoded nodes with a neighbour < 2.0 m: l2 0.70 MISSED (0.923), l25 0.30 (0.15 to 0.50) HIT (0.356), l31 0.10 (0.03 to 0.25) HIT at the edge (0.241).
**D106 line ('simpler AND faster?'):** this card changes no system. For the follow-up cards: a 3.125 m node code would have -35 % nodes, -59 % outputs and the same parts at the GT-code level; whether it is simpler AND faster than the 2 m code after training needs a trained arm (parts, parameters, GMACs of the link head, quiet-host ms): not claimed here.
**In-sample / caveats:** deterministic replays; the only noise is scene sampling (22 scenes). The 3.125 m grid is 4 % coarser than the 3 m of D104 (3.0 m does not divide the 100 m window); the l31 arm gets 5 long links on plain lanes (SHORT stays 8 m), named in the coder's notes (`temp/e356/build_notes.md`). `qnodes.py`, `qzones` and `qraster` assume N = 50: a trained head on these grids needs them changed (new files under the E343 freeze).
