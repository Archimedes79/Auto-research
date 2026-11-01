free -g | head -2
cd /mnt/c/Auto-research/bev_fusion || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
/usr/bin/time -v /home/ml/miniconda3/envs/gcar/bin/python tests/test_velocity_hurdle.py > /tmp/vh.log 2>&1
echo EXIT $? >> /tmp/vh.log
grep -v "fresh keys:" /tmp/vh.log | grep -i "ok\|PASS\|Error\|assert\|Maximum resident\|Elapsed\|EXIT\|Traceback\|line " ; rm -f /tmp/vh.log
