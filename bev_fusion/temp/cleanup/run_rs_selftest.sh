#!/bin/bash
cd /mnt/c/Auto-research/autoresearch || exit 1
unset AR_PROJECT
P=~/miniconda3/envs/gcar/bin/python
$P -m runsuite.selftest 2>&1 | tail -30
echo "exit ${PIPESTATUS[0]}"
