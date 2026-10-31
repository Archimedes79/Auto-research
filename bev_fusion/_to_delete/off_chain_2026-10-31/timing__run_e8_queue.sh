#!/usr/bin/env bash
# Sequential queue for the first idle-GPU window after the training rounds (one job at a time, never touches another job):
# 1 wait until no t1 round runs and the GPU is free (tests/run_after_t1.sh), 2 the idle-GPU timing of the S1 chain incl. the light
# encoders (retried every 10 min for 2 h until the GPU is really idle), 3 the S1 synthetic gate, 4 the distillation of the light
# encoder for ResNet-18, EfficientNet-B0, EfficientNet-B2 (each waits for a free GPU). Usage: bash temp/timing/run_e8_queue.sh
cd "$(dirname "$0")/../.." || exit 1
PY=/home/ml/miniconda3/envs/gcar/bin/python
export OMP_NUM_THREADS=2
bash tests/run_after_t1.sh true || exit 1
for i in $(seq 1 12); do
  $PY temp/timing/time_s1_chain.py > temp/timing/time_s1_chain.log 2>&1 && break
  sleep 600
done
$PY -u tests/test_s1_pipeline.py > temp/checks/s1_pipeline_log.txt 2>&1
mkdir -p runs
for b in resnet18 efficientnet_b0 efficientnet_b2; do
  bash tests/run_when_gpu_free.sh $PY -u training/distill_light_encoder.py --backbone $b >> runs/distill_queue.log 2>&1
done
