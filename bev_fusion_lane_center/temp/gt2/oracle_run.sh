cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
until grep -q GALLERY_DONE temp/jr/gallery_run.log 2>/dev/null; do sleep 10; done
for S in 0 1; do
  bash tools/env/_py14.sh temp/gt2/oracle_pairs.py e308_t_s$S $S > temp/gt2/oracle_e308_s$S.log 2>&1
done
echo ORACLE_DONE >> temp/gt2/oracle_e308_s1.log
