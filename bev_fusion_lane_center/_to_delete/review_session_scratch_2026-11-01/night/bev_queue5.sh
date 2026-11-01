# overnight GPU queue part 5 (D28), replaces part 4: after part 2 (t10.dual_s1 + its eval) -> t11 B' -> B2 (accum 2) -> their evals.
# The t8_ctl_s1 eval is left to the run.py item t8.ctl_s1.eval queued by another chat (pid 220506); this queue waits until it is over.
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY="$HOME/miniconda3/envs/gcar/bin/python"
export OMP_NUM_THREADS=2
ram() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
wait_ram() { local ok=0; while [ $ok -lt 2 ]; do if [ "$(ram)" -ge "$1" ]; then ok=$((ok + 1)); else ok=0; echo "waiting for RAM: $(ram) GB, need $1 ($(date +%T))"; fi; sleep 60; done; }
evalrun() { [ -f runs/$1/.done ] || { echo "$1: no .done, no eval"; return; }
  wait_ram 10; $PY -u evaluation/bev_suite.py runs/$1 --workers 2 eval --n 1000 > runs/$1.eval.log 2>&1
  echo "$1 eval exit $? $(date +%T): $(tail -n 1 runs/$1.eval.log)"; }
arm() {  # arm <id> <config> <hyp>
  wait_ram 14
  AR_PROJECT=../bev_fusion python3 ../autoresearch/tools/run.py --id $1 --gpu --split train --hyp "$3" \
    --pred "card t11_vel_hurdle sections 1 + 2: vehicle ave_skill_moving 0.00 -> 0.25 (0.12-0.40), AVE 2.38 -> 1.95, static error <= 0.30 m/s, maps and front mAP2 within the t10 bands of t10_dual_s0 / s1" \
    --rule "card t11_vel_hurdle section 2 (arms B' and B2): an arm with vehicle ave_skill_moving >= 0.12 and maps / mAP2 within the bands is the velocity fix and the bev prototype candidate (t11_motion_field next); B2 alive and B' dead -> the effective batch is the lever; both dead -> not learned in 1000 steps (not 'no motion in the input')" \
    -- bash temp/train/run_ar.sh training/configs/$2.json > runs/$2.launcher.log 2>&1
  echo "$1 launcher exit $? $(date +%T): $(tail -n 1 runs/$2.launcher.log)"; }
while pgrep -f "[b]ev_queue2.sh" > /dev/null; do sleep 30; done
echo "part 2 over $(date +%T)"
while pgrep -f "run.py --id t8.ctl_s1.eval " > /dev/null || pgrep -f "[b]ev_suite.py runs/t8_ctl_s1" > /dev/null; do sleep 30; done
echo "t8_ctl_s1 eval (other chat's item) over $(date +%T)"
arm t11.vel_hurdle t11_vel_hurdle "arm B' of t11_vel_hurdle: t10_dual_s0 warm start, fresh vel branch (dead ReLU, t11_vel_cause), moving / static logit + vel L1 on moving centres + decode gate, batch 3, 1000 steps"
arm t11.vel_hurdle_acc2 t11_vel_hurdle_acc2 "arm B2 of t11_vel_hurdle: as B' with gradient accumulation 2 (effective batch 6; all batch-3 runs have a dead velocity output, the live ones had >= 6)"
git -C .. diff --stat bf/t11.vel_hurdle bf/t11.vel_hurdle_acc2 -- bev_fusion/training/*.py bev_fusion/autogt
grep -h gt_signature runs/t11_vel_hurdle/gt.json runs/t11_vel_hurdle_acc2/gt.json
evalrun t11_vel_hurdle
evalrun t11_vel_hurdle_acc2
nvidia-smi --query-gpu=memory.used --format=csv,noheader
