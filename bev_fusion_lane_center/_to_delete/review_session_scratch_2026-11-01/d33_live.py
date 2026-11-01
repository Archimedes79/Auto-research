"""live check of D33 spellings against the real project (project.json is protected and has no grant); run from bev_fusion_lane_center in WSL"""
import os
import sys
sys.path.insert(0, "../autoresearch/tools")
import guard_hook as g

cwd = "C:\\Auto-research"
for p in ["C:\\Auto-research\\bev_fusion_lane_center\\evaluation\\golden.json",
          "\\\\wsl$\\Ubuntu\\mnt\\c\\Auto-research\\bev_fusion_lane_center\\evaluation\\golden.json",
          "\\\\wsl.localhost\\Ubuntu\\mnt\\c\\Auto-research\\bev_fusion_lane_center\\evaluation\\golden.json",
          "\\\\localhost\\C$\\Auto-research\\bev_fusion_lane_center\\evaluation\\golden.json",
          "C:\\AUTO-R~1\\bev_fusion_lane_center\\evaluation\\golden.json",
          "\\Auto-research\\bev_fusion_lane_center\\evaluation\\golden.json",
          "//mnt/c/Auto-research/bev_fusion_lane_center/evaluation/golden.json",
          "bev_fusion_lane_center\\evaluation\\golden.json",
          "C:\\Auto-research\\bev_fusion_lane_center\\HANDOVER.md"]:
    for tool, inp in [("Write", dict(file_path=p, content="x")), ("mcp__dc__move_file", dict(source=p, destination="C:\\tmp\\x"))]:
        print("REFUSED" if g.verdict(tool, inp, cwd=cwd) else "allowed", tool[:18].ljust(18), p)

