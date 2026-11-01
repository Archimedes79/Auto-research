# re-time t10_dual_s0 and B2 alternately on the idle GPU (the first t10 session was flagged idle false); keep each latency.json per pass
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY="$HOME/miniconda3/envs/gcar/bin/python"
export OMP_NUM_THREADS=2
mkdir -p temp/retime_2026-11-01
for pass in 1 2; do
  for r in t10_dual_s0 t11_vel_hurdle_acc2; do
    while pgrep -f "[b]ev_suite.py runs/t(8|10|11)|[e]valuate.py|[t]rain.py|[p]16_train|[b]ev_prototype" > /dev/null; do sleep 15; done
    $PY -u evaluation/bev_suite.py runs/$r time > temp/retime_2026-11-01/$r.$pass.log 2>&1
    cp runs/$r/eval/latency.json temp/retime_2026-11-01/$r.$pass.json
    echo "$r pass $pass: $(grep -E '"(bev_dt0_median|median_ms|idle)"' runs/$r/eval/latency.json | tr -s ' ' | tr '\n' ' ') $(date +%T)"
  done
done
