#!/bin/bash
# run evalsuite + trainsuite selftests in gcar env, 2 threads; cwd = autoresearch/
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 MPLBACKEND=Agg
OUT="$1"
mkdir -p "$OUT"
which ffmpeg > "$OUT/env.txt" 2>&1; python -c "import numpy, scipy, matplotlib; print(numpy.__version__, scipy.__version__, matplotlib.__version__)" >> "$OUT/env.txt" 2>&1
( time timeout 1500 nice python -m evalsuite selftest ) > "$OUT/eval_selftest.txt" 2>&1; echo "exit $?" >> "$OUT/eval_selftest.txt"
( time timeout 600 nice python -m trainsuite selftest ) > "$OUT/train_selftest.txt" 2>&1; echo "exit $?" >> "$OUT/train_selftest.txt"
echo ALLDONE > "$OUT/done.txt"
