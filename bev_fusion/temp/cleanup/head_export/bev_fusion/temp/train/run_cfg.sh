#!/usr/bin/env bash
# One-off launcher of this loop: train one config from a SNAPSHOT of the training/ code at a PINNED commit (git archive:
# other chats edit and commit the shared code meanwhile; a round must differ from its control only by its own change),
# log to runs/<name>/train.log, record the commit, mark .done only on exit code 0.
# Usage: bash temp/train/run_cfg.sh <config.json> [<commit>, default HEAD]
cd "$(dirname "$0")/../.." || exit 1
cfg=$1; rev=${2:-HEAD}; name=$(basename "$cfg" .json)
mkdir -p runs/$name/code
git -C .. archive "$rev" bev_fusion/training | tar -x -C runs/$name/code --strip-components=2 || exit 1
echo "commit $(git rev-parse --short "$rev") (git archive of bev_fusion/training)" > runs/$name/code/SNAPSHOT.txt
~/miniconda3/envs/gcar/bin/python -u runs/$name/code/train.py "$cfg" > runs/$name/train.log 2>&1
rc=$?
[ $rc -eq 0 ] && touch runs/$name/.done
echo "$name exit $rc $(date +%T)"
