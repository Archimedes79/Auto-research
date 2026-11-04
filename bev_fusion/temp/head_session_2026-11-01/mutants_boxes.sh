#!/usr/bin/env bash
# mutants of the vectorised box extractor: every one must fail extract.py's self-test (CPU, seconds)
cd /mnt/c/Auto-research/bev_fusion/temp/head_session_2026-11-01 || exit 1
PY=~/miniconda3/envs/gcar/bin/python; SRC=../../training/extract.py; mkdir -p m; n=0; killed=0
mut() { n=$((n+1)); sed "$2" $SRC > m/extract.py
  if cmp -s $SRC m/extract.py; then echo "MUTANT $1: sed changed nothing"; return; fi
  if PYTHONPATH=/mnt/c/Auto-research/bev_fusion/training:/mnt/c/Auto-research/bev_fusion/evaluation OMP_NUM_THREADS=2 nice -n 10 $PY m/extract.py > m/out.log 2>&1; then echo "MUTANT $1: SURVIVED"
  else killed=$((killed+1)); echo "MUTANT $1: killed ($(grep -E 'Error' m/out.log | tail -1 | cut -c1-90))"; fi; }
mut xy_swapped   's/p = _params(sub, torch.div(cell, Y, rounding_mode="floor").double(), (cell % Y).double(), grid, soft=False)/p = _params(sub, (cell % Y).double(), torch.div(cell, Y, rounding_mode="floor").double(), grid, soft=False)/'
mut mov_flipped  's/gate = torch.sigmoid(obj\["mov"\]) if soft else (obj\["mov"\] >= 0).to(v.dtype)/gate = torch.sigmoid(obj["mov"]) if soft else (obj["mov"] < 0).to(v.dtype)/'
mut dir_flipped  's/back = obj\["dir"\]\[:, 0\] < 0/back = obj["dir"][:, 0] > 0/'
mut no_peaks     's/    peaks = hm \* (hm == F.max_pool2d(hm, 3, 1, 1)).to(hm.dtype)\n//; s/scores, idx = peaks.reshape(B, -1).topk/scores, idx = hm.reshape(B, -1).topk/'
mut group_wrong  's/return scores, torch.div(idx, X \* Y, rounding_mode="floor"), idx % (X \* Y)/return scores, idx % G, idx % (X * Y)/'
mut size_swapped 's/out\["length"\], out\["width"\] = torch.exp(obj\["size"\]\[:, 0\]), torch.exp(obj\["size"\]\[:, 1\])/out["length"], out["width"] = torch.exp(obj["size"][:, 1]), torch.exp(obj["size"][:, 0])/'
echo "killed $killed of $n"; rm -f m/extract.py m/out.log
