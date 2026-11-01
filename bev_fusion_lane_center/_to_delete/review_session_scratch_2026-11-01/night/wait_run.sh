# wait_run.sh <run.py id> [max minutes]: returns when no run.py launcher of that id is alive (the run ended: done, failed or stalled) or after max minutes
id=$1; max=${2:-240}
for i in $(seq 1 $((max * 2))); do
  pgrep -f "run.py --id $id " > /dev/null || { echo "launcher $id ended"; break; }
  sleep 30
done
pgrep -f "run.py --id $id " > /dev/null && echo "launcher $id still running after $max min"
cd /mnt/c/Auto-research/bev_fusion 2>/dev/null; date; nvidia-smi --query-gpu=memory.used --format=csv,noheader
