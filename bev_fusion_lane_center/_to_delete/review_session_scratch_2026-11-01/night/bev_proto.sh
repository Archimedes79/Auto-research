# bev prototype of B2 (t11_vel_hurdle_acc2): idle-GPU timing, then two val scenes with memory carried + dashboard video
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY="$HOME/miniconda3/envs/gcar/bin/python"
export OMP_NUM_THREADS=2
while pgrep -f "run.py --id E268.p" > /dev/null; do sleep 20; done
# idle GPU: no training / eval process of any chat, < 2 GB used (desktop baseline ~1.3-1.6 GB)
ok=0; while [ $ok -lt 2 ]; do
  if ! pgrep -f "[b]ev_suite.py|[e]valuate.py|[t]rain.py|[p]16_train" > /dev/null && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)" -lt 2000 ]; then ok=$((ok + 1)); else ok=0; fi; sleep 20; done
echo "GPU idle $(date +%T)"
for r in t11_vel_hurdle_acc2 t10_dual_s0; do
  $PY -u evaluation/bev_suite.py runs/$r time > runs/$r.time.log 2>&1
  echo "$r time exit $? $(date +%T): $(tail -n 2 runs/$r.time.log | tr '\n' ' ')"
done
$PY -u evaluation/bev_prototype.py runs/t11_vel_hurdle_acc2 --scenes scene-0003 5 --video --device cuda > runs/t11_vel_hurdle_acc2.proto.log 2>&1
echo "prototype exit $? $(date +%T): $(tail -n 3 runs/t11_vel_hurdle_acc2.proto.log | tr '\n' ' ')"
