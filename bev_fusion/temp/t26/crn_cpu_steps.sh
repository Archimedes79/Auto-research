#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
CRNPY=~/miniconda3/envs/crn/bin/python; GCARPY=~/miniconda3/envs/gcar/bin/python; OUT=~/work/CRN/outputs/t26_crn_lead
mkdir -p $OUT
echo "== dry-run"; nice -n 15 $CRNPY evaluation/crn/crn_export.py --dry-run 2>&1 | tail -6
echo "== oracle"; nice -n 15 $CRNPY evaluation/crn/crn_export.py --oracle --out $OUT/oracle 2>&1 | tail -4
echo "== convert"; nice -n 15 $GCARPY evaluation/crn/crn_convert.py $OUT/oracle/results_nusc.json $OUT/oracle/rec.pkl 2>&1 | tail -6
echo "== overlay"; nice -n 15 $GCARPY evaluation/crn/crn_gates.py overlay $OUT/oracle/rec.pkl $OUT/oracle/overlay.png 2>&1 | tail -4
cp $OUT/oracle/overlay.png /mnt/c/Auto-research/bev_fusion/temp/t26/crn_oracle_overlay.png 2>/dev/null; ls -la $OUT/oracle | head
