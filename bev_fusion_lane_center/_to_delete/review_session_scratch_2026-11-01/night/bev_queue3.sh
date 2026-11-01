# overnight GPU queue part 3 (D28), after part 2 (t10.dual_s1 + its eval): t8_ctl_s1 eval -> t11 arms A' then B' -> their evals -> t7 re-evaluation (finish4)
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
    --pred "card t11_vel_hurdle section 1 + 2: B' vehicle ave_skill_moving 0.00 -> 0.25 (0.12-0.40), AVE 2.38 -> 1.95, static error <= 0.30 m/s, maps and mAP2 equal to A' within the bands; A' kills its fresh vel branch again (t11_vel_cause P4)" \
    --rule "card t11_vel_hurdle rules V1-V4 (B' against A', same start, same seed): V1 B' skill >= 0.12 and maps / mAP2 within bands -> the hurdle is the velocity fix, next t11_motion_field; V3 no gain and mov logit no better than its prior -> the features carry no motion, next lever the input" \
    -- bash temp/train/run_ar.sh training/configs/$2.json > runs/$2.launcher.log 2>&1
  echo "$1 launcher exit $? $(date +%T): $(tail -n 1 runs/$2.launcher.log)"; }
while pgrep -f "[b]ev_queue2.sh" > /dev/null; do sleep 30; done
echo "part 2 over $(date +%T)"
evalrun t8_ctl_s1
arm t11.vel_ctl t11_vel_ctl "arm A' of t11_vel_hurdle: t10_dual_s0 warm start, fresh vel branch (dead ReLU, t11_vel_cause), today's L1 on all centres, 1000 steps"
arm t11.vel_hurdle t11_vel_hurdle "arm B' of t11_vel_hurdle: t10_dual_s0 warm start, fresh vel branch, moving / static logit + vel L1 on moving centres + decode gate, 1000 steps"
git -C .. diff --stat bf/t11.vel_ctl bf/t11.vel_hurdle -- bev_fusion/training bev_fusion/autogt
grep -h gt_signature runs/t11_vel_ctl/gt.json runs/t11_vel_hurdle/gt.json
evalrun t11_vel_ctl
evalrun t11_vel_hurdle
wait_ram 10
T7_NO_WAIT=1 bash temp/t7/finish4.sh > /dev/null 2>&1
echo "finish4 exit $? $(date +%T): $(tail -n 2 temp/t7/finish4.log)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
