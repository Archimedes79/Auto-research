cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONPATH=/mnt/c/Auto-research
PY=~/miniconda3/envs/gcar/bin/python
nice -n 15 $PY evaluation/soa_maps.py runs/t16_full_s0 temp/t26_b/soa_repro_noodel2.json --no-model --frames-from temp/t16/soa/soa_maps_v8_t16_full_s0.json --gt-signature 6a7aa7864a --check temp/t16/soa/soa_maps_v8_t16_full_s0.json 2>&1 | tail -2
echo "--- nonexistent run:"; nice -n 15 $PY evaluation/soa_maps.py runs/no_such_run temp/t26_b/x.json --no-model --frames-from temp/t16/soa/soa_maps_v8_t16_full_s0.json --gt-signature 6a7aa7864a 2>&1 | tail -2
$PY -c "
import json
d=json.load(open('temp/t26_b/soa_repro_noodel2.json'))
print({k: d['meta'][k] for k in ('frames_requested','frames_missing','frames_missing_tokens')})
"
