#!/bin/bash
# E291 ledger notes (card section 5; numbers from the cards and the run logs of E291.rule_*)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
L="python3 ../autoresearch/tools/ledger.py note"
$L E291.c_s0 --result "recipe twin (prior flag off), 3000 steps; snapped topo F1 0.6789 (stored E312 0.6825: -0.0035), unsnapped 0.6431" --decision "control of the rule (by hand: replay)" --notes "restarted from step 0 after a WSL out-of-memory error at step 230 of the first attempt (same seed, same data order); staged code temp/_session_scratch_e291"
$L E291.c_s1 --result "recipe twin, 3000 steps; snapped topo F1 0.6870 (stored E312 0.6864: +0.0006), unsnapped 0.6475" --decision "control of the rule (by hand: replay)" --notes "staged code temp/_session_scratch_e291"
$L E291.b_s0 --result "prior trained with corrupted counts, 3000 steps; snapped topo F1 clean 0.6838, noise30 0.6836, u 0.6830, s 0.6824, sc 0.6831; prior term / token 0.07, gate mean 0.68" --decision "see E291.rule_*_01: G- with R+ = TIE, kill (5) MET (by hand: replay, card section 5)" --notes "prior parameters 577"
$L E291.b_s1 --result "prior trained with corrupted counts, 3000 steps; snapped topo F1 clean 0.6892, noise30 0.6891, u 0.6880, s 0.6877, sc 0.6887" --decision "see E291.rule_*_01 (by hand: replay, card section 5)" --notes "prior parameters 577"
$L E291.rule_G_01 --result "b clean - twin, snapped, pooled over 2 seeds: +0.0035 [+0.0019, +0.0054] (seed 0 +0.0049, seed 1 +0.0022)" --decision "G- (point and upper end below +0.01) (by hand: replay)"
$L E291.rule_Gunsnap_01 --result "b clean - twin, unsnapped, pooled: +0.0014 [-0.0015, +0.0047]" --decision "non-inferiority holds (by hand: replay)"
$L E291.rule_R_01 --result "b at 30 % wrong counts - twin, snapped, pooled: +0.0034 [+0.0020, +0.0051]" --decision "R+ (lower end above -0.01) (by hand: replay)"
$L E291.rule_C_01 --result "b clean - b U (all counts unknown, geometry kept), pooled: +0.0010 [+0.0001, +0.0018]" --decision "C+ by the letter (interval excludes 0 by 0.0001); the count is worth about 0.001 (by hand: replay)"
$L E291.rule_Ks_01 --result "b shuffled table s - twin, pooled: +0.0021 [+0.0005, +0.0040]" --decision "kill (5) MET (lower end above 0): the input carries something other than the table's count (by hand: replay)"
$L E291.rule_Ksc_01 --result "b sc (own geometry, another frame's counts) - twin, pooled: +0.0029 [+0.0012, +0.0049]" --decision "kill (5) MET (by hand: replay)"
$L E291.rule_splits_01 --result "gain b clean - twin by arm: single-lane arms +0.0047 [+0.0029, +0.0067], multi-lane +0.0033 [+0.0013, +0.0056]; 25 - 40 m +0.0060, < 25 m +0.0036, >= 40 m +0.0010" --decision "analysis, no verdict: the gain is not larger where the count is above one"
$L E291.rule_gate_s0 --result "gate against count correctness (b noise30 vs clean): term / token right 0.0767, wrong 0.0903, correlation with 'right' -0.139" --decision "analysis, no verdict" --notes "the tool exited 2 after printing the result (cause not checked)"
$L E291.rule_gate_s1 --result "term / token right 0.0913, wrong 0.1047, correlation -0.109" --decision "analysis, no verdict"
echo "ledger notes done"
