#!/bin/bash
# D7 steps 5-6 through the launcher (ledger row, commit + tag, done flag). CPU only, new files only.
#   bash temp/loop19/launch_e235.sh chk     reproduction check on one b123 scene per part (~/exp_gcar_gtchk), then the comparison
#   bash temp/loop19/launch_e235.sh gt      the 255 new scenes (~/exp_gcar_b456)
cd "$(dirname "$0")/../.."
RUN="python3 ../autoresearch/tools/run.py"
HYP="data extension b456 (D7 steps 5-6): ground truth, route and SD map of the new frames with the b123 scripts (tag argument, CPU)"
case "$1" in
  chk) $RUN --id E235.chk2 --retries 1 --timeout 90 --hyp "$HYP; reproduction check on b123 train scene 3 and val scene 1" \
         --pred "every artefact identical to the stored b123 rows; exact distance (CPU k-d tree vs GPU cdist) within 0.05 px, >= 99 % identical pixels" \
         --rule "PASS -> run E235.gt on b456; FAIL -> find the difference before any b456 file is written" \
         -- bash -c "bash temp/loop19/run_b456_gt.sh gtchk && bash tools/env/_py14.sh temp/loop19/check_b456_gt_repro.py compare" ;;
  gt)  $RUN --id E235.gt --retries 1 --timeout 300 --hyp "$HYP; exact distance by the CPU k-d tree (E235.chk2: the stored b123 map has TF32 errors, everything else reproduces bit for bit)" \
         --pred "8,731 + 1,567 rows in every artefact; route found for about the b123 share of frames (train 89 %, val 88 %)" \
         --rule "b456 ground truth is used (step 7, loader) only if the post-check passes (row counts, value ranges, route share within 5 points of b123)" \
         -- bash temp/loop19/run_b456_gt.sh b456 ;;
  *) echo "usage: launch_e235.sh chk|gt"; exit 2 ;;
esac
