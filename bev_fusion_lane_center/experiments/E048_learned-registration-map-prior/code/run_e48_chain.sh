#!/usr/bin/env bash
# E48: Kette Grobstufe -> Feinstufe: (a) Korrelationssuche (E40) + gelernte Feinstufe, (b) gelernte Grob- + Feinstufe; Registrierungsfehler und Wirkung auf den Vorwissens-Kopf (e40_clean) bei sigma 1 / 2 / 3 m
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
G="/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh"
export NO_SLOT=1
until [ -f $O/e48_fine/align_eval_chain.json ]; do sleep 15; done
echo "== $(date +%H:%M:%S) chain eval corr+fine"
$G learned_align.py eval --name e48_chain2 --chain corr,e48_fine --sigmas 0.5,1,2,3 --n 300 2>&1 | grep -E "^sigma|Traceback|Error"
run() { # name sxy sdeg aligner
  $G dense_ext.py --name $1 --steps 0 --init_from e40_clean --prior map --prior_xy 0 --prior_deg 0 --eval_prior_xy $2 --eval_prior_deg $3 --eval_align 2 --aligner $4 --eval val,test --workers 3 > $O/$1.out 2>&1
}
run e48_clean_s1_corrfine 1 2 corr,e48_fine
run e48_clean_s2_corrfine 2 4 corr,e48_fine
run e48_clean_s3_corrfine 3 6 corr,e48_fine
run e48_clean_s05_corrfine 0.5 1 corr,e48_fine
run e48_clean_s1_chain 1 2 e48_align,e48_fine
run e48_clean_s2_chain 2 4 e48_align,e48_fine
echo "== $(date +%H:%M:%S) e48 chain done"
