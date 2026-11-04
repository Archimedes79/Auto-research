#!/bin/bash
# scratch: note timing.idle3c in the ledger, then queue the idle-GPU timing with the second patch (timing.idle3d; a diagnostic run: --no-card, no verdict)
cd /mnt/c/Auto-research || exit 1
export AR_PROJECT=../bev_fusion
python3 autoresearch/tools/ledger.py note timing.idle3c --result "no number: the patched guard (own pid dropped) still refused the second look after the model load (utilisation 6 - 29 percent right after the load, no other process listed)" --decision "diagnostic run, feeds no verdict; repeated as timing.idle3d with temp/t16/timing_patched2.py (own pid dropped and a 3 s settle before the second look)"
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
ps -eo pid,args | grep -E "[b]ev_suite|[t]rain" | cut -c1-120
nohup python3 /mnt/c/Auto-research/autoresearch/tools/run.py --id timing.idle3d --gpu --split val --no-card -- bash temp/t16/timing_all2.sh > runs/timing_idle3d.launcher.log 2>&1 &
sleep 4
ps -eo pid,etime,args | grep "[t]iming.idle3d" | cut -c1-200
