#!/usr/bin/env bash
# t11 vel cause: launcher of an instrumented run under autoresearch/tools/run.py (the pattern of temp/train/run_ar.sh, whose code snapshot
# it shares: a `git archive` of training/ at $AR_COMMIT), running temp/box_vel/train_diag.py instead of train.py; the driver and the config
# are copied into runs/<name>/code so the run folder holds what ran. Exit code = the driver's.
# Usage (from bev_fusion/, through run.py, see records/README.md): bash temp/box_vel/run_diag.sh temp/box_vel/configs/<name>.json
cd "$(dirname "$0")/../.." || exit 1
cfg=$1; name=$(basename "$cfg" .json)
rev=${AR_COMMIT:-${2:-HEAD}}
commit=$(git -C .. rev-parse --verify --quiet "$rev^{commit}") || { echo "run_diag: no commit $rev"; exit 2; }
mkdir -p runs/$name/code
git -C .. archive "$commit" bev_fusion/training | tar -x -C runs/$name/code --strip-components=2 || exit 1
cp temp/box_vel/train_diag.py "$cfg" runs/$name/code/ || exit 1
echo "commit ${commit:0:10} (git archive of bev_fusion/training; run ${AR_RUN_ID:-without run.py}, tag ${AR_TAG:-none}) + temp/box_vel/train_diag.py" > runs/$name/code/SNAPSHOT.txt
echo "=== $(date +%F_%T) run ${AR_RUN_ID:-} commit ${commit:0:10} config $cfg (diag driver)" >> runs/$name/train.log
~/miniconda3/envs/gcar/bin/python -u runs/$name/code/train_diag.py "runs/$name/code/$(basename "$cfg")" >> runs/$name/train.log 2>&1
rc=$?
tail -n 40 runs/$name/train.log
[ $rc -eq 0 ] && touch runs/$name/.done
echo "$name exit $rc $(date +%T)"
exit $rc
