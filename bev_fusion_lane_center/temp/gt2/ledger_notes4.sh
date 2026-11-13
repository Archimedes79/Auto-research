cd "$(dirname "$0")/../.."
L="python3 ../autoresearch/tools/ledger.py"
$L note E303.s_s0 --result "unsnapped p>=0.5: F1@1.0 +0.0080 over plain (+0.0010 over p2, +0.0008 over the chord control), topo F1 +0.0077; snapped p>=0.1: topo F1 +0.0320 (chord control +0.0265)" --decision "TIE by hand: card rule (F1@1.0 +0.01 over c not reached), kept and refined; the scales help on the snapped read-out (E311)" 2>&1 | tail -n 1
$L note E303.s_s1 --result "unsnapped p>=0.5: F1@1.0 +0.0139 over plain (+0.0032 over p2, +0.0023 over the chord control), topo F1 +0.0085; snapped p>=0.1: topo F1 +0.0356 (chord control +0.0251)" --decision "TIE by hand: card rule (F1@1.0 +0.01 over c not reached), kept and refined; the scales help on the snapped read-out (E311)" 2>&1 | tail -n 1
