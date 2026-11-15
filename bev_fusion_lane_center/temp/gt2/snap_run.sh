cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
until tail -n 1 temp/jr/maneuver_snap.log 2>/dev/null | grep -q MANEUVER_SNAP_DONE; do sleep 10; done
for S in 0 1; do
  bash tools/env/_py14.sh temp/gt2/snap_score.py e308_t_s$S $S 0.5 0.1 > temp/gt2/snap_e308_s$S.log 2>&1
done
echo SNAP_DONE >> temp/gt2/snap_e308_s1.log
