cd "$(dirname "$0")/../.."
L="python3 ../autoresearch/tools/ledger.py"
$L note E307.n_s0 --result "kp_f1 0.723 vs 0.630 (j1), crowded entry recall 0.789 vs 0.542, entry >35 m 0.652 vs 0.551, zone IoU 0.767 vs 0.760" --decision "KEEP by hand: card rule (kp F1 > +0.01, crowded recall > +0.04, zone IoU within 0.01) on both seeds; base for later maps" 2>&1 | tail -n 1
$L note E307.n_s1 --result "kp_f1 0.705 vs 0.642 (j1), crowded entry recall 0.789 vs 0.576, entry >35 m 0.651 vs 0.562, zone IoU 0.761 vs 0.759" --decision "KEEP by hand: card rule (kp F1 > +0.01, crowded recall > +0.04, zone IoU within 0.01) on both seeds; base for later maps" 2>&1 | tail -n 1
$L note E304.f_s0 --result "stopped at step 200 of 3000" --decision "stopped by hand: E307 (sigma 1.0) made the single-factor test moot, see E304 card section 2" 2>&1 | tail -n 1
$L note E304.f_s1 --result "stopped at step 200 of 3000" --decision "stopped by hand: E307 (sigma 1.0) made the single-factor test moot, see E304 card section 2" 2>&1 | tail -n 1
