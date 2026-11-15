#!/usr/bin/env bash
cd /mnt/c/Auto-research || exit 1
python3 bev_fusion/temp/cleanup/audit_b/readme_check.py bev_fusion/tests bev_fusion/tests/support bev_fusion/studies bev_fusion/studies/early bev_fusion/tools bev_fusion/temp bev_fusion/training 2>&1
