#!/bin/bash
# one-off: the ORACLE prediction (what a perfect network delivers) as a picture and a video of the same scene, for the report (CPU)
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
D=$PWD/temp/e273/report/oracle
mkdir -p $D
nice -n 12 bash tools/env/_py14.sh qgraph_view.py frame scene-0221:19 --out $D --pred oracle 2>&1 | tail -2
nice -n 12 bash tools/env/_py14.sh qgraph_view.py scene 11 --out $D/scene --video --pred oracle 2>&1 | tail -2
ls -R $D | head -20
