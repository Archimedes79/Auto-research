cd /mnt/c/Auto-research/bev_fusion_lane_center
python3 - <<'PY'
import json
d=json.load(open('data/splits.json'))
print(type(d))
for k,v in d.items():
    print(k, type(v), len(v) if hasattr(v,'__len__') else v, v[:3] if isinstance(v,list) else '')
PY
grep -n 'class EvalSet' -A25 training/data/p10_infer.py | head -40
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
ps aux --sort=-%cpu | head -8
free -g
cat tools/env/_py14.sh | head -30
