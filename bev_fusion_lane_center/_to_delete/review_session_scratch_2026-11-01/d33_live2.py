import sys
sys.path.insert(0, "../autoresearch/tools")
import guard_hook as g
cwd = "C:\\Auto-research"
for p in ["/c/Auto-research/bev_fusion_lane_center/evaluation/golden.json", "C:bev_fusion_lane_center\\evaluation\\golden.json",
          "D:bev_fusion_lane_center\\evaluation\\golden.json", "C:\\AUTO-R~1\\BEV_FU~1\\evaluation\\golden.json",
          "/c/Auto-research/bev_fusion_lane_center/HANDOVER.md", "C:bev_fusion_lane_center\\HANDOVER.md", "C:\\Users\\x.md"]:
    print("REFUSED" if g.verdict("Write", dict(file_path=p, content="x"), cwd=cwd) else "allowed", p)
print("no cwd:", "REFUSED" if g.verdict("Write", dict(file_path="evaluation/golden.json", content="x")) else "allowed")
