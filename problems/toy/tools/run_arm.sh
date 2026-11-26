#!/usr/bin/env bash
# one arm and seed as ONE run under run.py: train, then evaluate and write the metric card. runs/<name>/.done only when both ended with 0.
#   run_arm.sh <composed config.json>          (the "train" command of the training block of project.json)
# The same command again continues: the trainer resumes from its newest checkpoint (a finished training returns at once), the evaluation reuses stored predictions.
cd "$(dirname "$0")/.." || exit 2
cfg=$1
name=$(basename "$cfg" .json)
mkdir -p "runs/$name"
echo "=== $(date +%F_%T) run ${AR_RUN_ID:-without run.py} commit ${AR_COMMIT:-none} config $cfg" >> "runs/$name/train.log"
bash tools/py.sh training/train.py "$cfg" >> "runs/$name/train.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  bash tools/py.sh evaluation/score.py "$cfg" >> "runs/$name/eval.log" 2>&1
  rc=$?
  tail -n 12 "runs/$name/eval.log"
fi
tail -n 8 "runs/$name/train.log"
[ $rc -eq 0 ] && touch "runs/$name/.done"
echo "$name exit $rc $(date +%T)"
exit $rc
