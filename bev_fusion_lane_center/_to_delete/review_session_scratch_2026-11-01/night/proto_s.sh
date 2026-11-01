cd /mnt/c/Auto-research/bev_fusion_lane_center
echo 1000 > /proc/self/oom_score_adj
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
bash tools/env/_py14.sh lane_prototype.py recall_sp0_dp > temp/loop22/proto_sp0_full.log 2>&1
echo "full VAL: exit $? $(tail -n 3 temp/loop22/proto_sp0_full.log | tr '\n' ' ')"
bash tools/env/_py14.sh lane_prototype.py recall_sp0_dp --scenes 23 7 --video --out runs/prototype/recall_sp0_dp_video > temp/loop22/proto_sp0_video.log 2>&1
echo "video: exit $? $(tail -n 3 temp/loop22/proto_sp0_video.log | tr '\n' ' ')"
