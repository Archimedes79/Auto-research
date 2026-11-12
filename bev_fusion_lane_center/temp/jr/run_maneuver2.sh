cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
bash tools/env/_py14.sh ../jr/maneuver.py e308_t_s0 0 plain 'p>=0.5' 'p>=0.5 snap' 'p>=0.1' 'p>=0.1 snap' oracle 'oracle snap' 'oracle+fit snap' > temp/jr/maneuver_snap.log 2>&1
echo MANEUVER_SNAP_DONE >> temp/jr/maneuver_snap.log
