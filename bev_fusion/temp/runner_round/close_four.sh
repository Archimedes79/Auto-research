#!/bin/bash
# scratch: close the four remaining held runs on Mirko's word 'Ja, alle vier' (2026-11-09, answer to the BEV system chat's question); notes are appended, HOLD markers stay (the refusal)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
L=../autoresearch/tools/ledger.py
W="CLOSED 2026-11-09 on Mirko's word ('Ja, alle vier', his answer in the BEV system chat to the question whether the four holds on the current line should also be closed): "
T="The HOLD marker stays as the refusal (hold.py clear would allow a restart); nothing deleted."

python3 $L note smoke.t22_B_s0 --notes "${W}a 7 min smoke run of t22 stage B (student encoder) of 2026-11-05, stopped; superseded by the real stage B run t22.B_s0 and later smokes. ${T}" || exit 1

python3 $L note smoke.t22_B_s0b --notes "${W}the repeat of that smoke (4 min, 2026-11-05), stopped; superseded by t22.B_s0 and smoke.t22_B2_s0. ${T}" || exit 1

python3 $L note t20.resume_s0 --notes "${W}a 3 min resume attempt of the t20 run (2026-11-03), stopped; the t20 reference t20.base_s0 was trained to the end (step 6000) and is the base of the later work. ${T}" || exit 1

python3 $L note t21.rec_s0 --notes "${W}the recurrent arm of t21, cut at step 72 of 1500 for speed (53 s/step) on 2026-11-04 and repeated as t21.rec_s0b; this id is not resumed. ${T}" || exit 1
echo done
