# E355 step 0 results (lane experiments chat, 2026-11-16 02:34; CPU GT replay, no training; by hand: GT replay; nothing is written into E355's card)

Source: the nine arm cards `runs/eval/e355_<arm>/val.json` (jobs `E355.s20 ... E355.c25`, run.py, 873 VAL frames, 22 scenes) next to the stored dense cards l2 / l25 / l31 (E346 / E356) and the undecoded lane GT g1; the table is `wsl bash tools/env/_py14.sh e346_arm.py report355` (full output: `temp/e356/e355_step0_report.txt`). Tool: the E346 driver with the arms of `temp/e346/e355_lib.py` (spec `temp/e356/e355_arms_spec.md`, notes `temp/e356/build_notes.md`); checks before the arms: selftest, l2 against the stored card 104 of 104 numbers at 1e-6, l25 / l31 at 31 frames 116 / 120 of 120 at 1e-6.

## Table (g1 = the undecoded lane GT; every other column a decoded arm; '-' = undefined for the arm)
| metric | g1 | l2 | l25 | l31 | s20 | s25 | s31 | v20 | v25 | v31 | m20 | m25 | c25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| topo_F1 | 0.9858 | 0.9669 | 0.9656 | 0.9646 | 0.9668 | 0.9656 | 0.9646 | 0.9685 | 0.9652 | 0.9673 | 0.9617 | 0.9591 | 0.7744 |
| lanes_found | 1.0000 | 0.9853 | 0.9836 | 0.9791 | 0.9853 | 0.9836 | 0.9791 | 0.9998 | 0.9993 | 0.9963 | 0.9846 | 0.9764 | 0.8346 |
| F1@1.0 | 1.0000 | 0.9957 | 0.9953 | 0.9944 | 0.9957 | 0.9953 | 0.9944 | 0.9999 | 0.9997 | 0.9992 | 0.9951 | 0.9937 | 0.8699 |
| junction_F1 | 0.9995 | 0.9535 | 0.9467 | 0.9363 | 0.9535 | 0.9467 | 0.9363 | 0.9896 | 0.9855 | 0.9763 | 0.9235 | 0.9112 | 0.9441 |
| vert100_dp | 10.33 | 11.45 | 11.41 | 11.38 | 11.45 | 11.41 | 11.38 | 13.66 | 13.57 | 13.39 | 11.67 | 11.65 | 17.44 |
| nodes_per_frame | - | 180.5 | 146.9 | 117.8 | 78.7 | 74.3 | 75.7 | 118.7 | 109.9 | 107.5 | 98.2 | 81.4 | 146.9 |
| nodes_per_frame_p95 | - | - | 254 | 204 | 141 | 138 | 140 | 254 | 235 | 227 | 176 | 147 | 254 |
| nodes_per_frame_max | - | - | 280 | 221 | 159 | 154 | 151 | 276 | 268 | 245 | 193 | 157 | 280 |
| junc_nodes_per_connector | - | - | - | - | 0.60 | 0.59 | 0.57 | 4.30 | 3.87 | 3.52 | 0.58 | 0.55 | 0.59 |
| nn_share_lt20 | - | - | 0.356 | 0.241 | 0.061 | 0.049 | 0.032 | 0.238 | 0.156 | 0.091 | 0.000 | 0.000 | 0.000 |
| nn_share_lt25 | - | - | 0.908 | 0.331 | 0.100 | 0.094 | 0.074 | 0.334 | 0.286 | 0.178 | 0.208 | 0.000 | 0.000 |
| wbl_connector_share | - | - | - | - | 0.127 | 0.127 | 0.130 | 0.686 | 0.691 | 0.708 | 0.127 | 0.127 | 0.127 |
| wbl_connector_share_excl_adjacent | - | - | - | - | 0.036 | 0.038 | 0.042 | 0.485 | 0.510 | 0.541 | 0.047 | 0.047 | 0.047 |
| conn_cov_03 | - | - | - | - | 0.916 | 0.910 | 0.904 | 0.992 | 0.987 | 0.980 | 0.900 | 0.891 | 0.385 |
| link_all_d_p90 | - | - | - | - | 0.049 | 0.050 | 0.050 | 0.002 | 0.003 | 0.004 | 1.473 | 1.495 | 1.122 |
| merge_event_loss_share | - | - | - | - | - | - | - | - | - | - | 0.015 | 0.031 | - |
| merge_node_share | - | - | - | - | - | - | - | - | - | - | 0.456 | 0.446 | - |

## Rule readings (the card's rules, applied by hand)
**Bar (loss against g1 at most 0.03 on topo F1, lanes found, F1@1.0):** every arm PASSES except c25 (topo F1 0.2114 below g1, F1@1.0 0.1301 below).
**X2, kept nodes inside the junction (YES at a lattice only when v_L passes the bar AND junction F1 is at least 0.01 above s_L AND at most 5 % of the connectors have a wanted-but-lost node): NO at all three lattices.**
- 2.0 m: v20 junction F1 0.9896 against s20 0.9535 = +0.0361; wanted-but-lost connectors 0.686 (spec-literal) / 0.485 (without losses to an adjacent piece).
- 2.5 m: v25 0.9855 against s25 0.9467 = +0.0388; wanted-but-lost 0.691 / 0.510.
- 3.125 m: v31 0.9763 against s31 0.9363 = +0.0400; wanted-but-lost 0.708 / 0.541.
- The junction gain is real (+0.036 .. +0.040 junction F1, lanes found +0.015 at 2.5 m, connector coverage within 0.3 m 0.910 -> 0.987), but it costs: kept nodes per connector 0.59 -> 3.87, nodes per frame 74.3 -> 109.9 (p95 138 -> 235), delivered vertices per 100 m 11.41 -> 13.57, and 69 - 71 % of the connectors still lose a wanted node in the one-node-per-cell contest (the limit of any lattice token set), 49 - 54 % when the loss to the adjacent piece of a shared cell is not counted. The rule's 5 % is far from both readings.
**Spacing, reading (ii) (AFFORDABLE: passes the bar and loses at most 0.01 topo F1 and 0.02 junction F1 against the dense code of the same lattice): NOT AFFORDABLE at both lattices.**
- m20 against l2: topo F1 -0.0052 (bar -0.01), junction F1 -0.0300 (bar -0.02), nodes per frame 180.5 -> 98.2; event nodes lost 0.015; merges between pieces that are not linked 1075 of 71836 (773 of them between pieces that share a predecessor or successor: near-split branches).
- m25 against l25: topo F1 -0.0065, junction F1 -0.0355, nodes per frame 146.9 -> 81.4; event nodes lost 0.031; unlinked merges 1066 of 57123.
- The spacing is bought with junction F1 (the survivors are at least R apart by construction: nn_share_lt20 of m20 and lt25 of m25 are 0 by construction, not a result). The topo F1 cost alone would be affordable (-0.005 / -0.007).
**c25, cell-centre positions (the price of lattice-fixed positions):** F1@1.0 -0.1254, topo F1 -0.1912, lanes found -0.1490 against l25; connector coverage within 0.3 m 0.385; p90 distance of the delivered curve to its own path 1.12 m (median 0.54 m). **A token set needs free offsets from the cell centre (the card's token design has them: cell centre + offset); lattice-fixed positions are not an option.**
**Sparse set (s arms, suppression ON):** s20 / s25 / s31 agree with the dense l2 / l25 / l31 to four decimals on the scorer numbers (topo F1, lanes found, F1@1.0) while delivering 79 / 74 / 76 nodes per frame instead of 180 / 147 / 118 (-56 % / -49 % / -36 %): the sparse set is free in the ground truth (E276 arm s found it not learnable by a per-cell target; X1 needs a loss without one).

## Predictions of section 0 against the numbers (hit windows from the card)
- v25 loses 0.03 topo F1 against g1 (window 0.02 - 0.045): 0.0206 HIT at the edge. Its junction F1 NOT 0.01 above s25 (window -0.02 .. +0.005): +0.0388 MISS (the gain is large). More than 5 % wanted-but-lost (window 15 - 45 %): 69 % MISS above the window, the direction (X2 not worth it by the rule) HIT.
- m25 loses 0.01 - 0.03 topo F1 against l25: 0.0065 MISS (smaller); merges 5 - 15 % of the events: 3.1 % lost MISS; not affordable or borderline: NOT AFFORDABLE (by the junction F1) HIT. m20 affordable: NOT AFFORDABLE MISS (junction F1 0.0300 below l2, bar 0.02). c25 loses 0.02 - 0.06 F1@1.0: 0.1254 MISS (twice as large).

## Deviations and caveats (named)
- **The kill row of s20 (E276 P1) was run with the print resolution as tolerance:** the card says 5e-5; the first full s20 run read topo F1 0.96682 against the PRINTED 0.9669 (7.9e-5) and vert100_dp 11.449 against the printed 11.5 (0.051), while the other five numbers (lanes found, junction F1, Q, F1@1.0, vert100 14.41) reproduce; the stored dense l2 is 0.966853 / 11.449, so sparse and dense differ by 3e-5 in the fifth decimal (E276 section 7a states agreement to four decimals). Tolerances were widened to 1e-4 and 0.1 at 00:15 (by hand, `temp/e346/e346_arm.py` line 41), after the first run, before any other arm was run; the sparse pass-through is shown by vert100 14.41 against the dense 33.07.
- `E355.s25` died at 01:11 with OSError 12 (WSL page cache, a gzip read of a data loader); the lead resubmitted s25 ... c25 at 01:25 with --retries 1 after dropping the page cache; the arm numbers are deterministic replays, nothing changed. A page-cache guard runs for the night.
- Counts as built (coder's notes, `temp/e356/build_notes.md`, critic pass `a83b34a927d579e8b`): the merge merges when d < R (survivors at least R apart); `wbl_connector_share` is spec-literal and counts benign joins at shared cells, so the column without losses to an adjacent piece is printed next to it; `dropped_candidates_per_node` is undefined for the s / v / m arms (their numerator comes from the dense contest); `junc_nodes_per_connector` is not 0 with the suppression ON (event nodes inside the junction).
- 22 scenes; deterministic replays; the only noise is scene sampling (no seeds).
- The 3.125 m lattice is 4 % coarser than 3 m (3.0 does not divide the 100 m window); the l31 / v31 / s31 arms get 5 long links on plain lanes (SHORT stays 8 m).
- Compute (D106): none of these arms is a system; a trained token design is judged by parts, parameters, GMACs and quiet-host ms in its own card.
