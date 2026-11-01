# overnight GPU queue part 1 (D28): after t10.dual_s1 -> its eval -> t8_ctl_s1 resumed (base seed 2) -> its eval
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY="$HOME/miniconda3/envs/gcar/bin/python"
export OMP_NUM_THREADS=2
while pgrep -f "run.py --id t10.dual_s1 " > /dev/null; do sleep 30; done
echo "t10.dual_s1 ended $(date +%T): $(tail -n 1 runs/t10_dual_s1/train.log)"; ls runs/t10_dual_s1/.done 2>&1
[ -f runs/t10_dual_s1/.done ] && $PY -u evaluation/bev_suite.py runs/t10_dual_s1 --workers 2 eval --n 1000 > runs/t10_dual_s1.eval.log 2>&1
echo "t10_dual_s1 eval $(date +%T): $(tail -n 1 runs/t10_dual_s1.eval.log)"
AR_PROJECT=../bev_fusion python3 ../autoresearch/tools/run.py --id t8.ctl_s1 --gpu --split train \
  --hyp "resume of the crashed base seed t8_ctl_s1 (CUDA error at step 2500 of 3000): gives the board's base t8_ctl its second seed; the last 500 steps run on the snapshot code of this run, not 96a02be (default-off options only in between)" \
  --pred "cold VAL 1000 within the t8 bands of t8_ctl_s0 (drivable 0.716, lane divider 0.191, mAP 0.251)" \
  --rule "measurement, no decision of its own: the board uses s0 + s1 for its base band; t10 and the t7 arms are then read against two base seeds" \
  -- bash temp/train/run_ar.sh training/configs/t8_ctl_s1.json > runs/t8_ctl_s1.launcher.log 2>&1
echo "t8.ctl_s1 launcher exit $? $(date +%T)"; ls runs/t8_ctl_s1/.done 2>&1
[ -f runs/t8_ctl_s1/.done ] && $PY -u evaluation/bev_suite.py runs/t8_ctl_s1 --workers 2 eval --n 1000 > runs/t8_ctl_s1.eval.log 2>&1
echo "t8_ctl_s1 eval $(date +%T): $(tail -n 1 runs/t8_ctl_s1.eval.log 2>/dev/null)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
