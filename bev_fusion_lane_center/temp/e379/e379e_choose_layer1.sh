#!/bin/bash
# layer 1 of `jtoken_eval.py choose-cell` (the dev copy) against the scratch choosers of E377 / E378 on the same tables; jsons to temp/e379/e379e_choose/ (never runs/)
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
for rs in e376_mx_s0 e376_mx_s1 e377_n_s0,e377_n_s1 e367_b5_s0,e367_b5_s1 e378_x_s0; do
  new=$(nice -n 10 bash tools/env/_py14.sh temp/e379/e379e_session_scratch_dev2/evaluation/jtoken_eval.py choose-cell --runs $rs --out-dir ../../e379e_choose 2>/dev/null | tail -1)
  old=$(python3 temp/e378/choose378.py ${rs/,/ })
  old377=$( [ "${rs#*,}" != "$rs" ] && python3 temp/e377/choose377.py ${rs/,/ } | grep CHOSEN | cut -c1-40 )
  echo "$rs  new $new  choose378 $old  choose377 ${old377:--}"
done
