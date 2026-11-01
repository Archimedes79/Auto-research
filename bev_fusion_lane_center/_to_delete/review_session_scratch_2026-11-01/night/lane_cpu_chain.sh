# lane CPU chain (loop 22): E265.a now, then E265.v, E265.b, E265.eb one after another (the second CPU lane is E261.g / E261.cv)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
S=temp/loop22
mk() { echo "#!/usr/bin/env bash" > $S/run_$1.sh; echo 'cd "$(dirname "$0")/../.." || exit 1' >> $S/run_$1.sh; echo "$2" >> $S/run_$1.sh; }
E="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2"
mk E265v "$E bash tools/env/_py14.sh experiments/E265_far-coverage-loss-budget/cleanup_split.py recall_s1 e24_j1rc_s1 e24_j1rc_s2 e24_j1rc_s3 champion fast_recall_na"
mk E265b "for m in recall_s1 e24_j1rc_s1 e24_j1rc_s2 e24_j1rc_s3 recall; do $E bash tools/env/_py14.sh evaluate.py \$m --asm spur_len=2.5 --tag sp25 --compare \$m || exit 1; done"
mk E265eb "$E bash tools/env/_py14.sh experiments/E261_recall-weight-6-operating-point/frontier_scratch.py e123_D7_r3_rc1:288:e123_D7_r3_rc1,e18_D7_j1rcs1:192:e18_D7_j1rcs1 0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7"
for id in E265.a E265.v E265.b E265.eb; do
  f=$S/run_${id/./}.sh
  echo 1000 > /proc/self/oom_score_adj
  python3 ../autoresearch/tools/run.py --id $id --require-card --split val --basis E259 -- bash $f > $S/${id/./}.launcher.log 2>&1
  echo "$id: $(tail -n 1 $S/${id/./}.launcher.log)"
done
