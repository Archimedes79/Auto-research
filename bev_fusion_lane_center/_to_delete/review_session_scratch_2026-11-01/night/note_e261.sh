cd /mnt/c/Auto-research/bev_fusion_lane_center
L="python3 ../autoresearch/tools/ledger.py note"
S="python3 ../autoresearch/tools/schedule.py set"
$L E261.cv --result "20 scene-half splits (choice on half A, read on half B): rc2 has an admissible point on half A in 13 of 20; held-out mean a-c lanes found +0.029 (SD 0.010), R@1 far +0.021, Q -0.0075 (> the 0.005 guard), topo F1 -0.0085, FL +0.013; a-t lanes found +0.0157; DP of a > 14.3 in 10 of 13 held-out halves (rule: at most 5); in-sample choice under the DP-halves rule: rc2 tau 0.9 / pruning 0.5" --decision "fails the DP-halves check and the held-out Q guard: rc2's extra coverage is bought with vertices and Q"
$L E261.ev --result "VAL 873 (selection-CV, in-sample choice): recall6 (rc2 0.9 / 0.5) vs recall: lanes found +0.0133, R@1 far +0.0054, Q +0.0023, topo F1 -0.0037, dup +0.0295 (limit 0.030), DP 13.08; recall_thr (rc1 0.2 / 0.4) vs recall: lanes found +0.0105, R@1 far +0.0096, Q -0.0049, DP 13.48" --decision "CUT by decide.py (lanes found +0.0133 < +0.015, R@1 far +0.0054 < +0.010); the card's own text would call a-c in (-0.010, +0.015) a TIE, same consequence: recall (rc1) stays, rc2 not adopted, E262 cancelled, candidate W out (decide.py has no TIE-window clause: tools note)"
$S E261.reg done
$S E261.dec done
$S E262.a1 cancelled
$S E262.ev cancelled
$S E262.dec cancelled
$S E265.dec done
