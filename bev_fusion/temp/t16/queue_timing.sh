#!/bin/bash
# scratch: note the failed timing.idle3b in the ledger, then queue the idle-GPU timing of the three systems with the patched own-pid guard (timing.idle3c; a diagnostic run: --no-card, no verdict)
cd /mnt/c/Auto-research || exit 1
export AR_PROJECT=../bev_fusion
python3 autoresearch/tools/ledger.py note timing.idle3b --result "no number: all three timings were refused by the guard's second check (nvidia-smi lists the timing process itself as a GPU user once its model is loaded, so 'the GPU got busy while the model was loading' always fires; the GPU was idle); the job then failed (exit 2) because I edited its script while it ran" --decision "diagnostic run, feeds no verdict; repeated as timing.idle3c with temp/t16/timing_patched.py (the own pid dropped from the guard's list, every other check kept)"
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
ps -eo pid,args | grep -E "[b]ev_suite|[t]rain" | cut -c1-120
nohup python3 /mnt/c/Auto-research/autoresearch/tools/run.py --id timing.idle3c --gpu --split val --no-card -- bash temp/t16/timing_all.sh > runs/timing_idle3c.launcher.log 2>&1 &
sleep 4
ps -eo pid,etime,args | grep "[t]iming.idle3c" | cut -c1-200
