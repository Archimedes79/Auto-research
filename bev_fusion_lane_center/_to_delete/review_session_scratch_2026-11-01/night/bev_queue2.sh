# overnight GPU queue part 2 (D28): t8.ctl_s1 (running) -> its eval -> t10.dual_s1 resumed (OOM-killed at step 760; ckpt 500) -> its eval.
# A RAM guard before every GPU step: the OOM killer took t10.dual_s1 at 00:09 (WSL 23 GB; a training needs ~9 GB + workers).
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY="$HOME/miniconda3/envs/gcar/bin/python"
export OMP_NUM_THREADS=2
ram() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
wait_ram() { local ok=0; while [ $ok -lt 2 ]; do if [ "$(ram)" -ge "$1" ]; then ok=$((ok + 1)); else ok=0; echo "waiting for RAM: $(ram) GB free, need $1 ($(date +%T))"; fi; sleep 60; done; }
wait_launcher() { while pgrep -f "run.py --id $1 " > /dev/null; do sleep 30; done; }
evalrun() {  # evalrun <run>: cold VAL 1000 as the board's other arms
  [ -f runs/$1/.done ] || { echo "$1: no .done, no eval"; return; }
  wait_ram 10
  $PY -u evaluation/bev_suite.py runs/$1 --workers 2 eval --n 1000 > runs/$1.eval.log 2>&1
  echo "$1 eval exit $? $(date +%T): $(tail -n 1 runs/$1.eval.log)"
}
wait_launcher t8.ctl_s1
echo "t8.ctl_s1 ended $(date +%T): $(tail -n 1 runs/t8_ctl_s1.launcher.log)"
evalrun t8_ctl_s1
wait_ram 14
AR_PROJECT=../bev_fusion python3 ../autoresearch/tools/run.py --id t10.dual_s1 --gpu --split train \
  --hyp "seed twin of t10.dual_s0 (its rule E1 held: maps of t8_bevfeat kept, front mAP2 0.33 >= ctl - 0.054); resumed after the OOM kill at step 760 (00:09, RAM)" \
  --pred "cold VAL 1000 within the t10 bands of s0: lane divider@0.5 0.34, drivable 0.80, walkway 0.33, front mAP2 0.33" \
  --rule "s1 agrees with s0 (E1 again) -> the dual-feature chain is the new reference (board base proposal; promotion stays Mirko's); s1 fails E1 -> t10 is seed luck, keep t8_bevfeat / t8_ctl" \
  -- bash temp/train/run_ar.sh training/configs/t10_dual_s1.json > runs/t10_dual_s1.launcher2.log 2>&1
echo "t10.dual_s1 launcher exit $? $(date +%T): $(tail -n 1 runs/t10_dual_s1.launcher2.log)"
evalrun t10_dual_s1
nvidia-smi --query-gpu=memory.used --format=csv,noheader
