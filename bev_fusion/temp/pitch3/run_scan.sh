cd "$(git rev-parse --show-toplevel)"
~/miniconda3/envs/gcar/bin/python bev_fusion/temp/pitch3/scan.py > bev_fusion/temp/pitch3/scan_out.txt 2>&1
tail -3 bev_fusion/temp/pitch3/scan_out.txt
