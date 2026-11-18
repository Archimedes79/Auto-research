#!/bin/bash
# THE LANDING (07:00 - 08:30, the lead's order of 2026-11-18 01:00 and 01:05): the archive tags, the S2 / S3 / jtok2 / jrecur removal with the edits of the files that stay, ONE commit. Run it from the project folder ONLY when (a) no job runs and no chain is armed to start (quiet chain finished or stopped),
# (b) agreed order 2026-11-18: MINE first, then Fable's units U1 - U5 right after (the two hooks out of jtoken.py, `jtoken_eval.py` switched from jrecur_time.timed to jtiming.timed), then the report chat's --jobs, (c) the E379 / S4 read-outs are done. After BOTH: the token tests, `check_structure.py`, one smoke (lane_live_tok selftest), then the full run_all_tests.sh.
# usage: bash temp/landing/land.sh [--yes]      (without --yes it only checks the preconditions and rehearses the patch)
cd "$(dirname "$0")/../.."
set -e
if pgrep -f "run.py .*--gpu|jtoken_train.py|jlive_train.py|quiet_chain.sh|e379/sup.sh|time_s2.py" > /dev/null; then echo "STOP: a job or a chain is alive"; pgrep -af "run.py .*--gpu|jtoken_train.py|jlive_train.py|quiet_chain.sh|e379/sup.sh" | cut -c1-120; exit 1; fi
grep -q "past_block" training/grid/jtoken.py && echo "NOTE: the hooks are still in jtoken.py (Fable lands his unit U5 right after this commit; test_jtoken.py's past / maps tests need jpast / jmap until then)"
grep -q "import jrecur_time" evaluation/jtoken_eval.py && echo "NOTE: jtoken_eval.py still imports jrecur_time inside timing() (Fable's switch to jtiming.timed lands right after this commit: only `jtoken_eval.py timing` is broken in between)"
python3 temp/landing/land_patch.py . > /tmp/land_check.out 2>&1 || { cat /tmp/land_check.out; exit 1; }
tail -2 /tmp/land_check.out
[ "$1" = "--yes" ] || { echo "preconditions hold, patch anchors found: run again with --yes"; exit 0; }
git tag archive/s2-s3-blocks-2026-11-18 HEAD
git tag archive/jtok2-jrecur-2026-11-18 HEAD
python3 temp/landing/land_patch.py . --apply
GONE="training/grid/jpast.py training/train/jpast_train.py evaluation/jpast_eval.py training/grid/jmap.py training/train/jmap_train.py evaluation/jmap_eval.py training/data/lane_map_data.py training/grid/test_jpast.py training/grid/test_jpast_train.py training/grid/test_jmap.py training/grid/test_jmap_train.py training/grid/test_lane_map.py evaluation/token_readout.py evaluation/tests/test_token_readout.py training/grid/jtok2.py training/train/jtok2_train.py evaluation/jtok2_eval.py training/grid/test_jtok2.py training/grid/jrecur.py training/train/jrecur_train.py evaluation/jrecur_eval.py evaluation/jrecur_time.py training/grid/test_jrecur.py tools/maintenance/keep_alive.sh tools/maintenance/test_keep_alive.sh tools/maintenance/run_token_tests.sh"
rm -f $GONE
cd ..
P=""; for f in $GONE; do P="$P bev_fusion_lane_center/$f"; done
for f in $(cat bev_fusion_lane_center/temp/landing/changed.txt); do P="$P bev_fusion_lane_center/$f"; done
python3 autoresearch/tools/check_git.py --paths $P | tail -2
python3 autoresearch/tools/commit_paths.py -F bev_fusion_lane_center/temp/landing/msg.txt $P
