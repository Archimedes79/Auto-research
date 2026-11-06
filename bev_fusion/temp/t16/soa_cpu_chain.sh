#!/bin/bash
# scratch: the CPU re-scorings against the state of the art, one python job at a time, 2 threads, low priority (the t18.fix training runs on the GPU and slows down by ~25 % while they run):
# 1 wait for the DualBEV record, 2 the front-sector comparison (DualBEV vs t16 vs t17), 3 GaussianCaR vs t16 on the maps, 4 the same for t17
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=/home/ml/miniconda3/envs/gcar/bin/python
while pgrep -f "soa_front_v8.py --n 1000" > /dev/null; do sleep 15; done
echo "== front compare $(date +%T)"
if [ -f temp/t16/soa/dualbev_front_v8_n1000.pkl ]; then
  bash temp/t16/soa_front_v8_run.sh evaluation/bev_suite.py front dualbev_front=temp/t16/soa/dualbev_front_v8_n1000.pkl runs/t16_full_s0 runs/t17_data --n 1000 --x-max 50 --boot 1000 --out temp/t16/soa/front_v8_n1000.md 2>&1 | tail -5 | cut -c1-200
else
  echo "no DualBEV record"
fi
for RUN in t16_full_s0 t17_data; do
  echo "== maps $RUN $(date +%T)"
  nice -n 15 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY temp/t16/soa_maps_v8.py runs/$RUN temp/t16/soa/soa_maps_v8_$RUN.json 400 --align 2>&1 | tail -4 | cut -c1-200
done
echo "== done $(date +%T)"
