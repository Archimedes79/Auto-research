#!/bin/bash
cd "$(dirname "$0")/../.."
L=../autoresearch/tools/ledger.py
python3 $L note E240.c3 --result "pair 3 (stored map, LCR_DIST_DIR): Q j1rc 0.7391 / j1eq 0.7316 (+0.0075), lanes found 0.7393 / 0.6815 (+0.058); four pairs: Q a-c +0.0062, lanes found +0.0596, seed SD Q 0.0030 / 0.0020, paired SD 0.0043" --decision "E240 TIE by decide.py on four pairs (prediction +0.006 hit)"
python3 $L note E255.a0 --result "matched 14.3 vertices (tau 0.2, vtx_r 3.69): Q 0.6520 vs controls 0.7241 / 0.7252 (a-c -0.0726), lanes found 0.551 vs 0.675 / 0.687, topo F1 0.610 vs 0.737; held-out 623 Q 0.6519" --decision "E255 CUT (decide.py: Q -0.0726 <= -0.020, topo F1 guard violated); prediction -0.045 missed by -0.028; K193 closed"
python3 $L note E256.a0 --result "by city, five D4 candidates: Q Boston 0.7757 fast_recall / 0.7700 recall / 0.7661 control / 0.7625 champion / 0.7515 fast; Singapore 0.7137 recall / 0.7119 control / 0.7101 champion / 0.7064 fast / 0.7061 fast_recall; no pair flips by >0.010 in both cities" --decision "no flip, order not established; fast_recall advantage over fast is Boston-only (+0.024 / -0.000), lanes found +0.065 in both cities"
