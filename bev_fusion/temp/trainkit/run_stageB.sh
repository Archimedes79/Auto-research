#!/usr/bin/env bash
# scratch: all regression runs of a stage: the train.py self-test in the background, the related tests and the slower ones in a row
cd "$(dirname "$0")/../.." || exit 1
tag=${1:-B}
bash temp/trainkit/run_one.sh training/train.py > temp/trainkit/train_selftest_$tag.log 2>&1 &
bash temp/trainkit/run_related.sh temp/trainkit/stage${tag}_related.log
bash temp/trainkit/run_extra.sh temp/trainkit/stage${tag}_extra.log
wait
echo ALLDONE > temp/trainkit/stage${tag}_done.flag
