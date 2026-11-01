cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
nice bash tools/env/_py14.sh temp/loop21/make_results_video.py recall 2>&1 | tail -8
ls -la reports/weekly/Lane_graph_recall_val*
