#!/bin/bash
cd /mnt/c/Auto-research/autoresearch || exit 1
python3 -m jobrunner JobRunner.check /mnt/c/Auto-research/bev_fusion/temp/cleanup/job_demo.json --project /mnt/c/Auto-research/bev_fusion 2>&1 | cut -c1-260
echo "--- catalogue"
python3 -m jobrunner JobRunner.catalogue --project /mnt/c/Auto-research/bev_fusion 2>&1 | cut -c1-200 | head -30
