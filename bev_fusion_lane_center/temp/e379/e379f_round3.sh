# scratch: round 3 before the move: the HEAD comparison and the whole scratch suite (CPU, 2 threads, nice)
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
o=temp/e379/e379f_round3.out
: > "$o"
for t in temp/e379/e379f_headcmp.py "temp/e379/e379f_harness.py new"; do
  echo "=== $t $(date +%H:%M:%S)" >> "$o"
  nice -n 10 bash tools/env/_py14.sh $t > temp/e379/e379f_one.out 2>&1
  echo "exit $? $(date +%H:%M:%S)" >> "$o"
  grep -v '^[0-9][0-9]:[0-9][0-9]:[0-9][0-9] ' temp/e379/e379f_one.out | cut -c1-160 | tail -30 >> "$o"
done
echo ALLDONE >> "$o"
