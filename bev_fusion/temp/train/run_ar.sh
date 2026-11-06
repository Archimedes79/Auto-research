#!/usr/bin/env bash
# Train one config under autoresearch/tools/run.py (D23): the run.py variant of run_cfg.sh.
#   - code: a `git archive` snapshot of training/ and (D48) of ../autoresearch/trainkit at $AR_COMMIT, the commit run.py made for THIS attempt (a resumed id keeps its tag at
#     attempt 1, so the tag is not used); without run.py the second argument (default HEAD)
#   - config: the copy inside the snapshot when the config lies in training/ (same code, same config), else the given path
#   - exit code = train.py's exit code (run_cfg.sh always ends with 0): run.py writes its done flag only for 0, and the last 40 lines of
#     train.log are printed so run.py's CUDA "unknown error" retry and its last_lines see them
#   - not pinned: autogt/ is imported from the live tree when train.py starts (the run's tag records its state)
# Usage (from bev_fusion/, see records/README.md): bash temp/train/run_ar.sh <config.json> [<commit>]
cd "$(dirname "$0")/../.." || exit 1
cfg=$1; name=$(basename "$cfg" .json)
rev=${AR_COMMIT:-${2:-HEAD}}
commit=$(git -C .. rev-parse --verify --quiet "$rev^{commit}") || { echo "run_ar: no commit $rev"; exit 2; }
mkdir -p runs/$name/code
git -C .. archive "$commit" bev_fusion/training | tar -x -C runs/$name/code --strip-components=2 || exit 1
git -C .. archive "$commit" autoresearch/trainkit | tar -x -C runs/$name/code --strip-components=1 || exit 1   # D48: the engine is pinned next to train.py
echo "commit ${commit:0:10} (git archive of bev_fusion/training; run ${AR_RUN_ID:-without run.py}, tag ${AR_TAG:-none})" > runs/$name/code/SNAPSHOT.txt
run_cfg=$cfg
case "$cfg" in training/*) [ -f "runs/$name/code/${cfg#training/}" ] && run_cfg="runs/$name/code/${cfg#training/}" ;; esac
echo "=== $(date +%F_%T) run ${AR_RUN_ID:-} commit ${commit:0:10} config $run_cfg" >> runs/$name/train.log
~/miniconda3/envs/gcar/bin/python -u runs/$name/code/train.py "$run_cfg" >> runs/$name/train.log 2>&1
rc=$?
tail -n 40 runs/$name/train.log
[ $rc -eq 0 ] && touch runs/$name/.done
echo "$name exit $rc $(date +%T)"
exit $rc
