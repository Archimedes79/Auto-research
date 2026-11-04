#!/usr/bin/env bash
# one-off: commit "Doppler static gate + runner + closed bundle" through a temporary index: my files whole (each differs from HEAD only by my
# hunks, checked), the three READMEs = HEAD + my rows only (make_readmes.py). Run from the repository root (Git Bash), after the tests.
set -euo pipefail
T=bev_fusion/temp/arch_build
export GIT_INDEX_FILE=.git/index.gate
cp .git/index $GIT_INDEX_FILE
git read-tree HEAD
MINE="bev_fusion/runner/bev_runner.py bev_fusion/runner/README.md bev_fusion/tests/test_bundle.py bev_fusion/tests/test_runner.py
  bev_fusion/tests/test_slots.py bev_fusion/training/architecture.json bev_fusion/training/architecture.md bev_fusion/training/architecture.svg
  bev_fusion/training/augment.py bev_fusion/training/interface.py bev_fusion/training/model.py bev_fusion/training/precompute.py"
git add -- $MINE
for r in bev_fusion/training/README.md bev_fusion/tests/README.md bev_fusion/README.md; do
  out=$T/readme_gate_$(echo $r | tr '/' '_')
  wsl.exe python3 $T/make_readmes.py $r $out
  git update-index --cacheinfo 100644,$(git hash-object -w $out),$r
done
git diff --cached --stat HEAD
msg=$(cat $T/commit_gate_msg.txt)
c=$(printf '%s\n' "$msg" | git commit-tree $(git write-tree) -p HEAD)
git update-ref refs/heads/main $c HEAD
rm -f $GIT_INDEX_FILE; unset GIT_INDEX_FILE
git reset -q -- $MINE bev_fusion/training/README.md bev_fusion/tests/README.md bev_fusion/README.md
git log --oneline -1
