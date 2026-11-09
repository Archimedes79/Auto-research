#!/usr/bin/env bash
# one-off: commit the bug hunt through a temporary index: my files whole (each differs from HEAD only by my hunks), training/README.md
# = HEAD + my precompute row only. Run from the repository root (Git Bash).
set -euo pipefail
T=bev_fusion/temp/arch_build
export GIT_INDEX_FILE=.git/index.bughunt
cp .git/index $GIT_INDEX_FILE
git read-tree HEAD
MINE="bev_fusion/tests/test_slots.py bev_fusion/tests/equivalence.py bev_fusion/training/architecture.json bev_fusion/training/architecture.md
  bev_fusion/training/architecture.svg bev_fusion/training/augment.py bev_fusion/training/extract.py bev_fusion/training/interface.py
  bev_fusion/training/lift_pd.py bev_fusion/training/model.py bev_fusion/training/precompute.py bev_fusion/training/splat.py
  bev_fusion/training/train.py"
git add -- $MINE
r=bev_fusion/training/README.md
row=$(grep -F '| `precompute.py` |' $r)
git show HEAD:$r | awk -v row="$row" 'index($0, "| `precompute.py` |") == 1 { print row; next } { print }' > $T/readme_bughunt.md
[ "$(grep -c 'precompute.py check <config.json>' $T/readme_bughunt.md)" = 1 ]
git update-index --cacheinfo 100644,$(git hash-object -w $T/readme_bughunt.md),$r
git diff --cached --stat HEAD
msg='bev_fusion framework bug hunt before the first training: bf16 crash, cached bf16 geometry, unequal memory arms, contracts on real batches, precompute on its own

Two fresh ar-critic reviews (correctness, design) and a dry run on real data (hurdle chain, val clips, flip, fp32 and bf16, dense
outputs, evaluation path on a scratch run folder).
Fixed: splat.py computed cell indices in the feature dtype - under bf16 (exact integers only to 256) they ran past the grid, a
device-side assert, older than the build; positions now at least float32, integer indices, weights cast back. lift_pd.build_table
always float32 (bf16 tables were cached for later float32 calls); the visibility cache key holds the depth range. Every temporal
kind gets its write mask (the ego-GRU control got none). flip_batch mirrors precomputed maps. radar_hits optional. interface.py
names RADAR_FIELDS, requires v_hat at dt 0, finite-checks the dense boxes; train.forward_loss runs both contracts on the first real
batch of a process. precompute.py runs on its own (check / frames <config>). The drawing'"'"'s wiring corrected (backbone -> time
mover -> heads; planned mask edges dashed); architecture.md lists the four design gaps left.
Checks: tests/equivalence.py in clip mode (2 keyframes, memory carried, one sample mirrored, state compared) exact against the code
before the build for t10_dual_s0 (64 outputs) and the hurdle model (70) - run before the last splat tweak, a no-op in float32 (a cast to the same dtype, a branch for half precision only); the rerun died of a full machine; test_slots 28 kinds + 5 combinations (the delta_slots combinations are pending: the machine was full);
gates, train.py and model.py self-tests, the dry run and the evaluation path pass.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'
c=$(printf '%s\n' "$msg" | git commit-tree $(git write-tree) -p HEAD)
git update-ref refs/heads/main $c HEAD
rm -f $GIT_INDEX_FILE; unset GIT_INDEX_FILE
git reset -q -- $MINE $r
git log --oneline -1
