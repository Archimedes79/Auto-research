#!/bin/bash
# does the pinned code of a stored run import now that autogt/ is gone? (one run per process)
cd /mnt/c/Auto-research/bev_fusion || exit 1
RUN=runs/${1:-t20_base_s0}
PYTHONPATH=/mnt/c/Auto-research OMP_NUM_THREADS=1 nice -n 15 ~/miniconda3/envs/gcar/bin/python - "$RUN" <<'EOF' 2>&1 | tail -n 12 | cut -c1-230
import sys, importlib
sys.path.insert(0, "evaluation")
import bev_adapter as A
code = A.setup_paths(sys.argv[1])
print("code folder:", code)
bad = []
for m in ("gt_bev", "layers", "losses", "nuscenes_dataset", "precompute", "train"):
    try:
        importlib.import_module(m)
        print("import ok:", m)
    except Exception as e:
        bad.append(m)
        print("import FAILED:", m, "-", type(e).__name__, str(e)[:120])
print("failed:", bad)
EOF
