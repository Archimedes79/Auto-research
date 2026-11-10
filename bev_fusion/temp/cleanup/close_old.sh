#!/bin/bash
# scratch: close the three held runs that rest on the old (pre-D48) network, on Mirko's word of 2026-11-09; notes are appended, HOLD markers stay (the refusal)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
L=../autoresearch/tools/ledger.py
W="CLOSED 2026-11-09 on Mirko's word (typed in the BEV system chat: 'Raeum erstmal auf, wenn die auf alten Netzwerken beruhen'): "
T="The HOLD marker stays as the refusal (hold.py clear would allow a restart); nothing deleted."

python3 $L note profile.t10 --notes "${W}a 2.5 min profile run on the t10 network (the network before the D48 redesign; the reference is now t20.base_s0), no result, never to be restarted. ${T}" || exit 1

python3 $L note r2.raster64_s0 --notes "${W}run of the r2 radar-slot chain on the t10-era network, stopped at step 113 on 2026-11-01 (its checkpoint was deleted on 2026-11-02 at Mirko's word); the radar-slot question comes back as schedule item r2.radar_again on the new baseline with a fresh card. ${T}" || exit 1

python3 $L note t10.dual_long_s0 --notes "${W}the 6000-step run of the t10 dual network, stopped at step 5030 (Mirko had already chosen a complete network next); lever 1 stays unmeasured, the SOA-reachable claim neither confirmed nor dropped; its checkpoints ckpt_00005030.pt / ckpt_00005000.pt stay where they are (hard rule 5: archive or slim only on his word). ${T}" || exit 1
echo done
