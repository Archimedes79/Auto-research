#!/usr/bin/env bash
# E55: Seed-Streuung: (a) Referenz-Dichtekopf (dec32 + logit7, 4207 Zeilen, 3000 Schritte) mit Seeds 22 und 23 (Referenz e30_base2 = Seed 21), (b) Schluesselpunkt-Kopf mit Seed 1 (E44 = Seed 0) + Standard-Schnitt auf dem Graphen
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
echo "== $(date +%H:%M:%S) start"
for SD in 22 23; do
  ./run_gcar.sh dense_head.py --name e55_seed$SD --fg gcar --feats dec32,logit7 --steps 3000 --bs 8 --lr 6e-4 --beta2 0.98 --eps 1e-6 --workers 3 --dir 0.5 --lam 1 --wpos 1 --train_stride 2 --eval train_eval,val,test --seed $SD > $O/e55_seed$SD.out 2>&1 &
done
./run_gcar.sh node_head.py --name e55_nodes_s1 --seed 1 --workers 3 > $O/e55_nodes_s1.out 2>&1 &
wait
echo "== $(date +%H:%M:%S) trainings done"
C2='{"tau":0.5,"par":{"cut":1,"r_s":16.0,"r_b":8.0,"cos_min":0.5,"d_c":1.5,"prune":6.0}}'
./run_gcar.sh node_assemble.py --graphs e43/graphs_rule_e41_A --nodes e55_nodes_s1 --sets val,test --out e55_cons_s1 --par_json "$C2" --workers 6 2>&1 | grep -E "^\[(val|test)\]|Traceback|Error"
echo "== $(date +%H:%M:%S) done"
