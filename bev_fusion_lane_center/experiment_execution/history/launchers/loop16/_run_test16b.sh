#!/bin/bash
# TEST, once, with the user's OK (loop prompt of 2026-10-24 evening: "TEST (21 scenes): OK"):
# the route head along the edges (points every <= 4 m, seeds 0 / 1) on the champion f1 graph, the single-frame configuration j1 192 / 96 and the control E94 (same treatment);
# graph metrics of j1 192 / 96; planner consumer on the champion chain with the new head. Nothing is selected on TEST: thresholds and settings were fixed on validation.
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/diag_test16b
L=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e123_ladder
mkdir -p $O
rm -f $O/done
ASM="cov_r=0.5 vtx_r=1.5 junc_thr=0.5 reloc_thr=0.4 reloc_shift=12 reloc_tol=1.6"
export NMAXV=560
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_tok_resample.py tok_p16_f2h_test_clean tok_p16_f2hr_test_clean 4.0 > $O/resample.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_tok_resample.py tok_e94_a_test_clean tok_e94_ar_test_clean 4.0 >> $O/resample.log 2>&1
for SD in "p16_f2hr_route_s0" "p16_f2hr_route_s1"; do
  bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_eval2.py --tok tok_p16_f2hr_test_clean --split test_clean --model $SD --kind x --taus 0.5 --cells exact,s1,s2,s3,wrong,none --tag test > $O/champ_$SD.log 2>&1
done
for SD in "e104br" "e104br_s1"; do
  bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_eval2.py --tok tok_e94_ar_test_clean --split test_clean --model $SD --kind x --taus 0.5 --cells exact,s1,s2,s3,wrong,none --tag test > $O/ctrl_$SD.log 2>&1
done
# single-frame configuration j1 192 / 96: graph metrics and route heads
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_final_eval.py --design D7 --name e123_D7_r3_j1 --split test_clean --set M=192 K0=96 --asm $ASM --tau 0.7 --tag finalJ192 --keep > $O/j1_graph.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_tokens.py --design D7 --name e123_D7_r3_j1 --out tok_p16_j1 --split test_clean --tau 0.7 --set M=192 K0=96 --asm $ASM > $O/j1_tokens.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_tok_resample.py tok_p16_j1_test_clean tok_p16_j1r_test_clean 4.0 >> $O/resample.log 2>&1
for SD in "p16_f2hr_route_s0" "p16_f2hr_route_s1"; do
  bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_eval2.py --tok tok_p16_j1r_test_clean --split test_clean --model $SD --kind x --taus 0.5 --cells exact,s1,s2,s3,wrong,none --tag test_j1 > $O/j1_$SD.log 2>&1
done
# planner consumer on TEST, champion chain with the new route head (3 seeds)
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_chain_data.py --tok tok_p16_f2hr --route p16_f2hr_route_s0 --out p16chain_f2hr --draws 3 --workers 4 --splits test > $O/chain_data_test.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p14_chain_consumer.py --data p16chain_f2hr --out p16_constest_f2hr --modes route+ego --seeds 3 --eval_split test > $O/consumer_test.log 2>&1
echo done > $O/done
