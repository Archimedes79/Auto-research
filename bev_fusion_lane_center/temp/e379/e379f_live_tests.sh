# scratch: the live tests after the move (CPU, CUDA hidden, 2 threads, nice)
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
o=temp/e379/e379f_live_tests.out
: > "$o"
for t in "training/grid/test_jtoken.py" "training/grid/test_jpast_train.py selftests" "training/grid/test_jlive_train.py selftests" "training/grid/test_jmap_train.py selftests"; do
  echo "=== $t $(date +%H:%M:%S)" >> "$o"
  nice -n 10 bash tools/env/_py14.sh $t > temp/e379/e379f_one.out 2>&1
  echo "exit $? $(date +%H:%M:%S)" >> "$o"
  grep -v '^[0-9][0-9]:[0-9][0-9]:[0-9][0-9] ' temp/e379/e379f_one.out | cut -c1-200 | tail -6 >> "$o"
done
echo ALLDONE >> "$o"
