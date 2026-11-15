#!/bin/bash
# baseline: the fast gates of bev_fusion on a clean export of HEAD (other chats edit the working tree)
cd /mnt/c/Auto-research || exit 1
D=bev_fusion/temp/cleanup/head_export
rm -rf $D && mkdir -p $D
git archive HEAD autoresearch autogt bev_fusion data/NuScenes | tar -x -C $D
echo "exported $(git rev-parse --short HEAD): $(find $D -type f | wc -l) files"
cd $D/bev_fusion || exit 1
# the cache / data links of the real tree, read-only
for l in data runs; do [ -e /mnt/c/Auto-research/bev_fusion/$l ] && [ ! -e $l ] && ln -s /mnt/c/Auto-research/bev_fusion/$l $l; done
date +%T
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 10 bash tests/run_gates.sh 2>&1 | tee ../../gates_head.log
date +%T
