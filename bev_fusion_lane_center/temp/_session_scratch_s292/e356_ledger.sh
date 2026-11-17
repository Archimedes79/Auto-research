#!/bin/bash
# E356 ledger notes (card section 3)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
L="python3 ../autoresearch/tools/ledger.py note"
$L E356.a15chk_snap_s0 --result "control recomputation of the stored snapped card through the tagged path (--spec close:1.5@0.5 --tag c15): largest difference 0.0" --decision "kill row of part A passed (by hand: replay)"
$L E356.nn_spacing --result "B's delivered nodes: nodes per frame 156 (close:1.5), 70 % / 69 % with a neighbour < 2.0 m, 93 % / 92 % < 2.5 m, NN median 1.99 m; close:2.0 -> 106 nodes, close:2.5 -> 82 nodes" --decision "measurement, no verdict"
$L E356.a20_snap_s0 --result "close:2.0: B plain topo F1 0.2377 (1.5 m: 0.6259), E324 s snapped 0.4041 (0.6981), vert100_dp 28.7 / 23.7" --decision "E275 collapse confirmed (by hand: replay): not a candidate read-out"
$L E356.a20_snap_s1 --result "close:2.0: B plain 0.2195 (0.6249), E324 s snapped 0.3990 (0.7010), vert100_dp 30.8 / 24.5" --decision "E275 collapse confirmed (by hand: replay)"
$L E356.a25_snap_s0 --result "close:2.5: B plain 0.1603, E324 s snapped 0.3244, vert100_dp 22.5 / 19.5" --decision "E275 collapse confirmed (by hand: replay)"
$L E356.a25_snap_s1 --result "close:2.5: B plain 0.1389, E324 s snapped 0.3118, vert100_dp 24.8 / 20.3" --decision "E275 collapse confirmed (by hand: replay)"
$L E356.l25 --result "GT code on a 2.5 m grid: topo F1 0.9656 (-0.0202 vs g1), lanes found 0.9836, F1@1.0 0.9953, junction F1 0.9467, nodes per frame 147 (p95 254, max 280), share of decoded nodes with a neighbour < 2.0 m 0.356 / < 2.5 m 0.908, dropped candidates per node 1.33 %" --decision "PASS the E346 bar (by hand: GT replay); no spacing guarantee"
$L E356.l31 --result "GT code on a 3.125 m grid: topo F1 0.9646 (-0.0212 vs g1), lanes found 0.9791, F1@1.0 0.9944, junction F1 0.9363, nodes per frame 118 (p95 204, max 221), share with a neighbour < 2.0 m 0.241 / < 2.5 m 0.331, dropped candidates per node 2.61 %" --decision "PASS the E346 bar (by hand: GT replay); no spacing guarantee; 3.125 m is 4 percent coarser than 3 m"
echo "ledger notes done"
