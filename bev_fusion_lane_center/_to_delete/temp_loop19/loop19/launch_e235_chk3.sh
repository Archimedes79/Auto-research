#!/bin/bash
# D7 steps 5-6: the stronger reproduction check E235.chk3 (three b123 scenes per part in reversed order; exact-distance gate against a float64
# truth and the 0.5 m-point bound, see check_b456_gt_repro.py). CPU only. A separate file because launch_e235.sh is running E235.gt.
cd "$(dirname "$0")/../.."
bash tools/env/_py14.sh temp/loop19/check_b456_gt_repro.py setup || exit 2
python3 ../autoresearch/tools/run.py --id E235.chk3 --retries 1 --timeout 90 \
  --hyp "data extension b456 (D7 steps 5-6): the b123 ground-truth scripts with a tag argument reproduce b123 on three scenes per part in reversed order (review finding 4)" \
  --pred "every artefact identical to the stored b123 rows in token-list order; exact distance equal to the float64 truth within 0.01 px and never above the 0.5 m-point bound" \
  --rule "PASS -> the b456 files of E235.gt are accepted for step 7 (after their post-check); FAIL -> E235.gt is re-run after the fix" \
  -- bash -c "bash temp/loop19/run_b456_gt.sh gtchk && bash tools/env/_py14.sh temp/loop19/check_b456_gt_repro.py compare"
