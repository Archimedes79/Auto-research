#!/bin/bash
# t26.ref_crn: the full CRN export on the 1706 samples (TF32 off), gate c (devkit mAP within 0.473 +- 0.05, light tables) and the conversion to our frames (evaluation/crn/RUN.md steps 5 - 7)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2
CRNPY=~/miniconda3/envs/crn/bin/python
GCARPY=~/miniconda3/envs/gcar/bin/python
OUT=~/work/CRN/outputs/t26_crn
$CRNPY evaluation/crn/crn_export.py --workers 2 || exit 1
$CRNPY evaluation/crn/crn_devkit_gate.py $OUT/tf32_off/results_nusc.json --light || echo "GATE C DID NOT PASS (read the lines above)"
$GCARPY evaluation/crn/crn_convert.py $OUT/tf32_off/results_nusc.json $OUT/crn_front.pkl --expect-signature a107e0b4e8 || exit 1
echo "crn full done"
