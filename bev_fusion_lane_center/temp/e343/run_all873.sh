#!/bin/bash
# E343 code step: the whole `all` path on real VAL data in a scratch folder (--max 873 = all frames, files in temp/e343/val_max873/): stored dumps given/copied, so no dump is made; control = E312's stored VAL dumps and card (defaults of the VAL split)
cd /mnt/c/Auto-research/bev_fusion_lane_center
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 10 bash tools/env/_py14.sh jtest_eval.py all --split val --seeds 0 1 --run-prefix e324_s_s --max 873 --boot 1000 --device cpu --b-dump 'temp/e276/predfull_e273_p_s{seed}.pkl' > temp/e343/all873.log 2>&1
echo "exit $?" >> temp/e343/all873.log
