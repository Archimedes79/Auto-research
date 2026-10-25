#!/usr/bin/env bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# Skript in der Umgebung 'gcar' ausfuehren: run_gcar.sh <skript.py> [argumente]   (Protokoll auf stdout; Rasterizer-Ordner im Suchpfad)
# GPU-Auftraege (Training/Inferenz) belegen einen von GPU_SLOTS (Standard 5) Plaetzen (flock), damit die 12 GB der Grafikkarte nicht ueberlaufen; GPU_PRIO=low darf nur die ersten GPU_SLOTS-2 Plaetze belegen.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
export CUDA_HOME="$CONDA_PREFIX" PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR="$HOME/.numba_cache_gcar"
export PYTHONPATH="$HOME/work/gaussiancar:$HOME/work/gaussiancar/gaussiancar/ops/diff-gaussian-rasterization:${PYTHONPATH:-}"
f="$1"; shift
GPU=0
case "$f" in
  dense_head.py|dense_ext.py|refiner.py|temporal_b.py|sys_eval.py|final_system.py|lane_former.py|bench_sys.py|slot_test.py|p10_train.py|p10_infer.py) GPU=1 ;;
  link_head.py) [ "$1" != "build" ] && GPU=1 ;;
esac
if [ "$GPU" = 1 ] && [ -z "$NO_SLOT" ]; then
  NSLOT=${GPU_SLOTS:-5}; MAXI=$NSLOT
  [ "$GPU_PRIO" = "low" ] && MAXI=$((NSLOT-2))
  while true; do
    for i in $(seq 0 $((MAXI-1))); do
      flock -n -E 99 /tmp/gpuslot.$i python "$BFLC/scripts/$f" "$@" 2>&1
      rc=$?
      if [ $rc -ne 99 ]; then exit $rc; fi
    done
    sleep $((4 + RANDOM % 6))
  done
else
  python "$BFLC/scripts/$f" "$@" 2>&1
fi
