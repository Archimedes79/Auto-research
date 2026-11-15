#!/bin/bash
# one-off (E274): after the p runs are done, one GPU process at a time: score r and p on VAL 873, the verdicts of the card, then the read-out sweeps (c seed 1 full, NMS rules for c seed 0, r)
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
L=temp/e274/chain.log
echo "chain start $(date)" > $L
for k in 0 1; do
  echo "== score r_s$k $(date)" >> $L
  bash tools/env/_py14.sh qgraph_run_eval.py score --run e274_r_s$k --seed $k --exp e274 >> $L 2>&1
done
for k in 0 1; do
  echo "== score p_s$k $(date)" >> $L
  bash tools/env/_py14.sh qgraph_run_eval.py score --run e274_p_s$k --seed $k --exp e274 >> $L 2>&1
done
echo "== verdict r against c $(date)" >> $L
bash tools/env/_py14.sh qgraph_run_eval.py verdict274 --var e274_r_s0 e274_r_s1 --ctl e273_p_s0 e273_p_s1 >> $L 2>&1
echo "== verdict p against r (junction_F1, band 0.03) $(date)" >> $L
bash tools/env/_py14.sh qgraph_run_eval.py verdict274 --var e274_p_s0 e274_p_s1 --ctl e274_r_s0 e274_r_s1 --exp-ctl e274 --metric junction_F1 --band 0.03 >> $L 2>&1
echo "== verdict p against c (topo_F1) $(date)" >> $L
bash tools/env/_py14.sh qgraph_run_eval.py verdict274 --var e274_p_s0 e274_p_s1 --ctl e273_p_s0 e273_p_s1 >> $L 2>&1
touch temp/e274/SCORED
echo "== sweeps (hard stop 15:20 for the BEV chat) $(date)" >> $L
left() { echo $(( $(date -d 15:20 +%s) - $(date +%s) )); }
run_sweep() { t=$(left); if [ "$t" -gt 120 ]; then timeout ${t}s bash tools/env/_py14.sh sweep_gpu.py "$@" >> $L 2>&1; else echo "skipped (past 15:20): $*" >> $L; fi; }
run_sweep e273_p_s1
run_sweep e273_p_s0 --skip-plain
run_sweep e274_r_s0 --skip-plain
run_sweep e274_r_s1 --skip-plain
echo "chain done $(date)" >> $L
touch temp/e274/CHAIN_DONE
