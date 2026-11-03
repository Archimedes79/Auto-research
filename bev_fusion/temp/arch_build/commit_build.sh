#!/usr/bin/env bash
# one-off: commit the five-layer build through a temporary index: my files whole, tests/README.md = HEAD + my two rows only,
# the superseded architecture_target.* removed. Run from the repository root (Git Bash).
set -euo pipefail
T=bev_fusion/temp/arch_build
export GIT_INDEX_FILE=.git/index.build
cp .git/index $GIT_INDEX_FILE
git read-tree HEAD
MINE="bev_fusion/training/model.py bev_fusion/training/extract.py bev_fusion/training/precompute.py bev_fusion/training/interface.py
  bev_fusion/training/architecture.json bev_fusion/training/architecture.md bev_fusion/training/architecture.svg bev_fusion/training/README.md
  bev_fusion/tests/test_slots.py bev_fusion/tests/equivalence.py"
git add -f -- $MINE
OLD="bev_fusion/training/architecture_target.json bev_fusion/training/architecture_target.md bev_fusion/training/architecture_target.svg"
git rm -q --cached -- $OLD
r=bev_fusion/tests/README.md
git show HEAD:$r > $T/tests_readme_head.md
grep -F '| `test_slots.py` |' $r > $T/rows.md
grep -F '| `equivalence.py` |' $r >> $T/rows.md
[ "$(wc -l < $T/rows.md)" = 2 ]
awk -v rows="$T/rows.md" 'index($0, "| `test_velocity_hurdle.py` |") == 1 { while ((getline l < rows) > 0) print l } { print }' $T/tests_readme_head.md > $T/tests_readme_new.md
[ "$(grep -c '^| `test_slots.py` |\|^| `equivalence.py` |' $T/tests_readme_new.md)" = 2 ]
git update-index --cacheinfo 100644,$(git hash-object -w $T/tests_readme_new.md),$r
git diff --cached --stat HEAD
msg='bev_fusion in five layers: precompute step, input interface, slot tables, output interface, tiny post-processing (Mirko 2026-11-01)

training/precompute.py: the visibility maps (camera now, radar fan placeholder (D36), fusion area (D37), radar hits) in the batch frame,
made only where a memory reads them. training/interface.py: the input / output contract with check_inputs / check_outputs.
model.py: one builder table per slot and check_config instead of the constructor if-chains (attribute names and module order kept),
the precompute step at the entry, the radar memory writes where the radar measured, dense outputs (boxes, soft lines) behind
dense_outputs (off by default: training and timing compute what they did), boxes() selects on double maps (= decode).
extract.py: the heading vector carries the direction gradient, clamped line map, float32 and negative-distance cases tested.
Checks: tests/equivalence.py on CPU, the old code against this tree: t10_dual_s0 (29 outputs) and the hurdle model t10_dual_long_s0
step 5030 (32 outputs) identical, state keys identical; tests/test_slots.py 28 kinds and switches (gradient, interfaces, state; the
untrainable parts named, incl. the radar memory finding); fast gates and model.py selftest pass; fresh ar-critic pass (10 findings,
fixed). training/architecture.* is the final drawing (0 rule steps between the interfaces); architecture_target.* superseded.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'
c=$(printf '%s\n' "$msg" | git commit-tree $(git write-tree) -p HEAD)
git update-ref refs/heads/main $c HEAD
rm -f $GIT_INDEX_FILE; unset GIT_INDEX_FILE
git reset -q -- $MINE $r $OLD
rm -f $OLD
git log --oneline -1
