cd "$(dirname "$0")/../.."
L="python3 ../autoresearch/tools/ledger.py"
$L note E304.f_s0 --result "stopped at step 200 of 3000" --decision "stopped by hand: E307 (sigma 1.0) made the single-factor test moot, see E304 card section 2" 2>&1 | tail -n 1
$L note E304.f_s1 --result "stopped at step 200 of 3000" --decision "stopped by hand: E307 (sigma 1.0) made the single-factor test moot, see E304 card section 2" 2>&1 | tail -n 1
