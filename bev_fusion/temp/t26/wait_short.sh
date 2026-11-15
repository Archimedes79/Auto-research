#!/bin/bash
# exits when the short GPU chain has written "all short jobs done" or died
cd /mnt/c/Auto-research/bev_fusion
until grep -q "all short jobs done" temp/t26/short_gpu.log || ! pgrep -f "[q]ueue_short_gpu.sh" > /dev/null; do sleep 60; done
tail -n 12 temp/t26/short_gpu.log | cut -c1-240
date
