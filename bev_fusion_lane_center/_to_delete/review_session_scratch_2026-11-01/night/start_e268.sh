cd /mnt/c/Auto-research/bev_fusion_lane_center
echo 1000 > /proc/self/oom_score_adj
python3 ../autoresearch/tools/run.py --id E268.p --gpu --require-card --split test \
  --hyp "the loop-22 prototype recall_sp0_dp keeps its VAL coverage gain on TEST (spurs kept: true short geometry at range)" \
  --pred "TEST 833: lanes found about 0.735 - 0.74 (E230.rc 0.724 + the VAL gain), Q about 0.73, vertices within the budget; control lanes found 0.762" \
  --rule "report only (E268 card): nothing selected or tuned on TEST, no promotion; if lanes found is not above E230.rc's 0.724 the VAL gain did not transfer" \
  -- bash tools/env/_py14.sh evaluate.py recall_sp0_dp --split test --test-id E268.p --compare control > temp/loop22/E268p.launcher.log 2>&1
echo "E268.p: $(tail -n 1 temp/loop22/E268p.launcher.log) $(date +%T)"
