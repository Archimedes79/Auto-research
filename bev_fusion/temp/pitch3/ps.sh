pgrep -af "real_chain|--golden|test_bev_free|test_evaluate_door|test_evaluation_api" | cut -c1-150
grep MemAvailable /proc/meminfo
ls $(git rev-parse --show-toplevel)/bev_fusion/temp/pitch3 | grep real
