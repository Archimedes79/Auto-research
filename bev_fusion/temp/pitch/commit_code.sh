#!/usr/bin/env bash
# commit the REVIEWED t11 velocity-hurdle code of the training folder through a temporary index (the shared index and the working tree of the other chats stay as they are):
# whole files for the reviewed paths, tests/README.md as HEAD + the hurdle row only; main moves by a checked update-ref; the shared index entries of those paths are refreshed.
set -e
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
HERE="$(cd "$(dirname "$0")" && pwd)"
MSG="$HERE/commit_code_msg.txt"
PATHS="bev_fusion/training/object_head.py bev_fusion/training/losses.py bev_fusion/training/model.py bev_fusion/training/train.py bev_fusion/training/README.md \
 bev_fusion/training/configs/t10_dual_s1.json bev_fusion/training/configs/t10_dual_long_s0.json bev_fusion/training/configs/t11_vel_ctl.json \
 bev_fusion/training/configs/t11_vel_hurdle.json bev_fusion/training/configs/t11_vel_hurdle_acc2.json bev_fusion/training/configs/README.md \
 bev_fusion/tests/test_velocity_hurdle.py bev_fusion/tests/run_gates.sh"
old=$(git rev-parse HEAD)
# guard: the reviewed blobs of object_head / losses / model are the ones of the run snapshot (nobody edited them since), train.py is the file that was tested
for f in object_head losses model; do
  [ "$(git hash-object bev_fusion/training/$f.py)" = "$(git rev-parse d14597e2e8:bev_fusion/training/$f.py)" ] || { echo "ABORT: bev_fusion/training/$f.py changed since the reviewed snapshot"; exit 3; }
done
[ -z "${EXPECT_TRAIN:-}" ] || [ "$(git hash-object bev_fusion/training/train.py)" = "$EXPECT_TRAIN" ] || { echo "ABORT: train.py is not the tested file"; exit 3; }
export GIT_INDEX_FILE="$(mktemp -u)"
git read-tree "$old"
git add -- $PATHS
tmpreadme="$(mktemp)"
python3 "$HERE/make_tests_readme.py" "$tmpreadme"
blob=$(git hash-object -w "$tmpreadme")
git update-index --add --cacheinfo "100644,$blob,bev_fusion/tests/README.md"
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p "$old" -F "$MSG")
unset GIT_INDEX_FILE
git update-ref -m "reviewed t11 velocity hurdle code" refs/heads/main "$commit" "$old"
git reset -q HEAD -- $PATHS bev_fusion/tests/README.md
git log --oneline -1
git show --stat --format= "$commit" | tail -20
