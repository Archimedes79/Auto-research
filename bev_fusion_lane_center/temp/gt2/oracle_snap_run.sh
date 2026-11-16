cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
until tail -n 1 temp/jr/jman_e305.log 2>/dev/null | grep -q JMAN2_DONE; do sleep 10; done
bash tools/env/_py14.sh temp/gt2/oracle_pairs.py e305_e_s0 0 snap > temp/gt2/oracle_snap_e305_s0.log 2>&1
echo ORACLE_SNAP_DONE >> temp/gt2/oracle_snap_e305_s0.log
