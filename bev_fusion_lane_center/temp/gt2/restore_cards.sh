cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# my threshold sweep truncated runs/e308_t_s<S>/ends_card.json (the scorer opens the file before my shim redirects the dump): wait for the sweep, then score the run again at thr 0.5 (deterministic)
until grep -q THR_DONE temp/gt2/thr_e308_s1.log 2>/dev/null; do sleep 15; done
for S in 0 1; do
  bash tools/env/_py14.sh jends_eval.py score --run e308_t_s$S --seed $S > temp/gt2/restore_e308_s$S.log 2>&1
done
echo RESTORE_DONE >> temp/gt2/restore_e308_s1.log
