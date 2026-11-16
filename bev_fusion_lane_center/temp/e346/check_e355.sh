#!/bin/bash
# E355 step 0: the checks of the new arms of e346_arm.py in one go. CPU only, ONE process at a time, nice 19, 2 threads. Start from the problem folder in WSL, in the background:
#   bash temp/e346/check_e355.sh [selftest] [killrows] [smokes]        default: selftest killrows smokes (about 1 h on the shared box). The arms of start_arms_e355.txt start only AFTER this script ended green.
# killrows = two full 873-frame runs (about 10 min each): arm l2 against the stored card (1e-6) and arm s20 against the row of E276 P1; `e346_arm.py` ends such a run with its kill row itself, so a
# reproduction that fails is a FAILED line here. Log: temp/e346/out/check_e355.log (every command, its output; a failed step stops the script). Cards: temp/e346/out/ only, never runs/eval/.
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
OUT=$PWD/temp/e346/out                          # absolute: _py14.sh changes into the folder of the script
LOG=$OUT/check_e355.log
step() { echo "== $(date +%T) e346_arm.py $*" >> $LOG; nice -n 19 bash tools/env/_py14.sh e346_arm.py "$@" >> $LOG 2>&1 || { echo "FAILED: $*" >> $LOG; exit 1; }; }
stages="${*:-selftest killrows smokes}"
for s in $stages; do
  case $s in
    selftest)                                  # every grid, the toys, the merge rule by hand, every E355 arm on 9 real frames (planted degradations; the path of a job without override)
      step selftest ;;
    killrows)                                  # the kill rows of the card on the full 873 frames (run_arm ends each with `killrow`, exit 1 = no reproduction)
      step l2 --out $OUT/killrow_l2_full.json --skip-selftest
      step s20 --out $OUT/killrow_s20_full.json --skip-selftest ;;
    smokes)                                    # 31 frames (every 29th): the old dense arms again (must equal their stored smokes to 1e-6, same frame hash), then every E355 arm; then the table
      for a in l2 l25 l31; do
        step $a --max 31 --out $OUT/chk_${a}_n31.json --skip-selftest
        step killrow --ref $OUT/e346_${a}_n31.json --card $OUT/chk_${a}_n31.json
      done
      for a in s20 s25 s31 v20 v25 v31 m20 m25 c25; do step $a --max 31 --skip-selftest; done
      step report355 --max 31 ;;
    *) echo "unknown stage $s" >> $LOG; exit 2 ;;
  esac
done
echo "== $(date +%T) check_e355 done: $stages" >> $LOG
