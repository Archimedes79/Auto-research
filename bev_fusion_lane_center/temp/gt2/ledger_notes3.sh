cd "$(dirname "$0")/../.."
L="python3 ../autoresearch/tools/ledger.py"
$L note E305.e_s0 --notes "the launcher (run.py) was killed with the day-3 chain at 09:03 while the training kept running to step 3000 (log: done 3000, model.pt complete, 09:24); the row was written by ledger sync" 2>&1 | tail -n 1
$L note E305.e_s1 --notes "the launcher (run.py) was killed with the day-3 chain at 09:03 while the training kept running to step 3000 (log: done 3000, model.pt complete, 09:25); the row was written by ledger sync" 2>&1 | tail -n 1
$L note E286.jj_s0 --result "not judged: stopped at step 1000 of 3000 (11 s per step), ckpt kept" --decision "stopped by hand: speed, see E286 card section 8" 2>&1 | tail -n 1
$L note E286.jj_s1 --result "not judged: stopped at step 1000 of 3000 (11 s per step), ckpt kept" --decision "stopped by hand: speed, see E286 card section 8" 2>&1 | tail -n 1
