#!/bin/bash
# ledger rows for the speed runs that have none (b: process killed by me in teardown after its 70 steps; c, a2: launchers killed before they started a run) + result notes
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
L="python3 ../autoresearch/tools/ledger.py"
DRY=$1
$L add speed.b $DRY --status stopped --hyp "diagnostic (Mirko's mini measurement 2026-11-01/02, no verdict): HEAD code + batched loss read-back + 30-step torch.profiler window, 70 steps of t10_dual_s0" 2>&1 | tail -n 3
$L add speed.c $DRY --status stopped --hyp "diagnostic: launcher stopped by me before it started a run (repeated as speed.c2, 8-step profile window)" 2>&1 | tail -n 3
$L add speed.a2 $DRY --status stopped --hyp "diagnostic: launcher stopped by me before it started a run (repeated as speed.a3)" 2>&1 | tail -n 3
