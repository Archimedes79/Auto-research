#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
L="python3 ../autoresearch/tools/ledger.py"
$L note E353.rule_c --result "n1 (paint branch 25 % width) minus E324 s: snapped -0.0004 / -0.0002, PRIMARY -0.0015 / +0.0003, attach -0.0007 / +0.0007 topo F1; guards held (30 clauses)" --decision "ADOPT (by hand: replay, jrule_removal.py, rule of the card section 0); D106 ms pending (E353.time_s0, quiet window 05:00-07:00)" --notes "2026-11-16 02:55 lead: result section 3 of the card written; -1.1059 GMACs (-13.5 %), -27 664 parameters"
$L note E353.rule_cp --result "n1 minus cp (equal-budget continuation): snapped -0.0008 / -0.0007, PRIMARY -0.0012 / +0.0006, attach -0.0014 / +0.0013; guards held" --decision "ADOPT (by hand: replay, jrule_removal.py); bar (ii) holds on both seeds, all three read-outs" --notes "2026-11-16 02:55 lead: cp_s0 data order differs after step 1000 (resume, named in the card)"
