#!/bin/bash
# scratch: the cheap checks after the documentation cleanup (D49)
cd /mnt/c/Auto-research || exit 1
echo "== install_agents --check"
python3 autoresearch/tools/install_agents.py --check 2>&1 | tail -n 3
echo "== check_structure lane (before the cleanup: 75 problems)"
(cd bev_fusion_lane_center && python3 ../autoresearch/tools/check_structure.py 2>&1 | tail -n 4)
echo "== check_structure bev (before: 2 problems)"
(cd bev_fusion && AR_PROJECT=../bev_fusion python3 ../autoresearch/tools/check_structure.py 2>&1 | tail -n 4)
echo "== guard_hook selftest"
(cd bev_fusion_lane_center && python3 ../autoresearch/tools/guard_hook.py --selftest 2>&1 | tail -n 1)
echo "== catalogue check (lane literature)"
(cd bev_fusion_lane_center && python3 tools/literature/catalogue.py check 2>&1 | tail -n 3)
echo "== test_frozen"
(cd bev_fusion && python3 tests/test_frozen.py 2>&1 | tail -n 2)
