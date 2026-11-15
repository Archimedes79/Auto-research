#!/bin/bash
# t26.ref_crn_smoke: CRN R50 on 50 of our frames twice (TF32 off, then on), then gate b (centres within 0.1 m), the partial conversion and the overlay of gate a; evaluation/crn/RUN.md steps 2 - 4
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2
CRNPY=~/miniconda3/envs/crn/bin/python
GCARPY=~/miniconda3/envs/gcar/bin/python
OUT=~/work/CRN/outputs/t26_crn
set -e
$CRNPY evaluation/crn/crn_export.py --limit 50 --workers 2
$CRNPY evaluation/crn/crn_export.py --limit 50 --tf32 --workers 2
$GCARPY evaluation/crn/crn_gates.py tf32 $OUT/tf32_off_n50/crn_dump.pkl $OUT/tf32_on_n50/crn_dump.pkl
$GCARPY evaluation/crn/crn_convert.py $OUT/tf32_off_n50/results_nusc.json $OUT/tf32_off_n50/rec.pkl --partial
$GCARPY evaluation/crn/crn_gates.py overlay $OUT/tf32_off_n50/rec.pkl $OUT/tf32_off_n50/overlay.png --n 20
echo "crn smoke done"
