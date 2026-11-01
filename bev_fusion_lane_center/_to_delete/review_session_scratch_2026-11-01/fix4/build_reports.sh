cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
nice bash tools/env/_py14.sh reports/build/reports/build_status21_report.py 2>&1 | tail -3
nice bash tools/env/_py14.sh reports/build/reports/build_loop18_report.py 2>&1 | tail -3
nice python3 tools/maintenance/make_structure.py 2>&1 | tail -2
ls -la reports/weekly/*.html temp/STRUCTURE.md
