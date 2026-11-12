#!/bin/bash
# the head's tests that read the stored lane GT (GtGenerator) and the E273+ CPU tests, after the GtGenerator rename of 2026-11-10; output: one line per test
cd "$(dirname "$0")/../.."
P="bash tools/env/_py14.sh"
for t in grid/qtargets.py grid/qformat.py grid/map_sim.py grid/qgraph.py grid/qraster.py grid/test_qraster.py grid/test_qtargets.py test_qgraph_eval.py test_qgraph_readout.py test_qgraph_score_glue.py test_tier_metrics.py grid/test_tiers_sparse.py grid/test_tier_loss.py grid/test_soft_nms.py grid/test_zones.py test_zone_metrics.py viz/qgraph_view.py; do
  a=$t; [ "$t" = "viz/qgraph_view.py" ] && a="viz/qgraph_view.py --selftest"
  s=$(date +%s)
  out=$($P $a 2>&1); rc=$?
  e=$(( $(date +%s) - s ))
  echo "$([ $rc -eq 0 ] && echo PASS || echo FAIL) $t (${e}s) | $(echo "$out" | tail -1 | cut -c1-140)"
done
echo DONE
