cd "$(dirname "$0")/../.."
L="python3 ../autoresearch/tools/ledger.py"
$L note E313.s_s0 --result "kp_f1 0.743 vs 0.723 (E307), crowded entry recall 0.856 vs 0.789, isolated 0.691 vs 0.714, best-thr F1 +0.012" --decision "KEEP by hand: card rule on both seeds; isolated recall -0.02 / -0.03" 2>&1 | tail -n 1
$L note E313.s_s1 --result "kp_f1 0.732 vs 0.705 (E307), crowded entry recall 0.844 vs 0.789, isolated 0.676 vs 0.710, best-thr F1 +0.010" --decision "KEEP by hand: card rule on both seeds; isolated recall -0.02 / -0.03" 2>&1 | tail -n 1
