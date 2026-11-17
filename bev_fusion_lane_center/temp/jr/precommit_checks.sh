#!/usr/bin/env bash
cd /mnt/c/Auto-research || exit 1
python3 autoresearch/tools/check_git.py --paths reporter winners 2>&1 | tail -5
python3 - <<'EOF'
import zipfile
z = zipfile.ZipFile("bev_fusion_lane_center/results/E2E_Champion_Report/E2E_Champion_Paket.zip")
print(len(z.namelist()), "files in the package")
print([n for n in z.namelist()])
EOF
