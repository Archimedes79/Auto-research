cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONPATH=/mnt/c/Auto-research
PY=~/miniconda3/envs/gcar/bin/python
echo "--- refused:"; nice -n 15 $PY evaluation/soa_maps.py runs/t20_base_s0 temp/t26_b/soa_live_v13.json --no-model --frames-from temp/t16/soa/soa_maps_v8_t16_full_s0.json 2>&1 | tail -3; echo "exit $?"
echo "--- allowed:"; nice -n 15 $PY evaluation/soa_maps.py runs/t20_base_s0 temp/t26_b/soa_live_v13.json --no-model --frames-from temp/t16/soa/soa_maps_v8_t16_full_s0.json --allow-gt-mismatch 2>&1 | tail -9
