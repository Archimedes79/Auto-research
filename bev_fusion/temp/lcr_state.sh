#!/bin/bash
cd /mnt/c/Auto-research || exit 1
pgrep -fa "lcr_delete|git push" | head -3 | cut -c1-120
timeout 120 git ls-remote --tags origin 2>&1 | awk '{print $2}' | sed 's#^refs/tags/##; s#\^{}$##' | sort -u > /tmp/remote_now.txt
echo "origin: lcr $(grep -c '^lcr/' /tmp/remote_now.txt), other $(grep -vc '^lcr/' /tmp/remote_now.txt) | local: lcr $(git tag | grep -c '^lcr/'), other $(git tag | grep -vc '^lcr/')"
ls -la bev_fusion_lane_center/_to_delete/old_tags_2026-11-10/ | tail -2
ls /tmp/lcr_batch_* 2>/dev/null | head
