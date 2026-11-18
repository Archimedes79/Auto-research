#!/usr/bin/env bash
# E59 (Erweiterung): alternative Navigationsziele an Verzweigungen (make_route_alt.py) gegen die gefahrene Route derselben Samples
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
./run_gcar.sh make_route_alt.py 2>&1 | tail -n 1
for S in 0 1.0 2.0; do
  R=1; [ "$S" = "0" ] && R=0
  ./run_gcar.sh assoc_eval.py --sets test --sigma $S --reg $R --rt_file route_alt_val.pkl --workers 10 --out e59alt_s$S 2>&1 | grep -E "^\[|Traceback|Error" | cut -c1-420
  ./run_gcar.sh assoc_eval.py --sets test --sigma $S --reg $R --rt_file route_drv_sub_val.pkl --workers 10 --out e59drv_s$S 2>&1 | grep -E "^\[|Traceback|Error" | cut -c1-420
done
echo "== $(date +%H:%M:%S) done"
