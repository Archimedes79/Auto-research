cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
for S in 0 1; do
  (nice -n 5 bash tools/env/_py14.sh jends_eval.py score --run e312_p_s$S --seed $S --snap --thr 0.1 > temp/gt2/redo_e312_snap_s$S.log 2>&1; echo "redo done s$S" >> temp/gt2/redo_e312_snap.log) &
done
wait
echo REDO_E312_DONE >> temp/gt2/redo_e312_snap.log
