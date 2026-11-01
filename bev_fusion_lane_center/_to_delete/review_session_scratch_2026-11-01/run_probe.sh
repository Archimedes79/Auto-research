S=/mnt/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/6e9638e0-a42f-4660-b7d5-cac910d9084d/scratchpad
cd /mnt/c/Auto-research/bev_fusion || exit 1
mkdir -p /tmp/t11probe; export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
P=/home/ml/miniconda3/envs/gcar/bin/python
$P $S/probe_default.py $S/old/training /tmp/t11probe/old.pt > $S/probe.log 2>&1 || { echo EXIT old $? >> $S/probe.log; exit 1; }
$P $S/probe_default.py /mnt/c/Auto-research/bev_fusion/training /tmp/t11probe/new.pt >> $S/probe.log 2>&1
echo EXIT $? >> $S/probe.log
