import sys, os, time
sys.path.insert(0, "autoresearch/tools")
import guard_hook as G
B = chr(92)
w = lambda s: s.replace("|", B)
cwd = w("C:|Auto-research")
cases = [w(x) for x in [
 "C:|AUTO-R~1|BEV_FU~1|evaluation|golden.json",
 "C:|AUTO-R~1|BEV_FU~1|models|REGIST~1.JSO",
 "C:|auto-r~1|bev_fu~1|evaluation|golden.json",
 "C:|Auto-research|BEV_FU~1|project.json",
 "||wsl$|Ubuntu|mnt|c|AUTO-R~1|BEV_FU~1|evaluation|golden.json",
 "C:bev_fusion_lane_center|evaluation|golden.json",
 "/c/Auto-research/bev_fusion_lane_center/evaluation|golden.json",
 "/cygdrive/c/Auto-research/bev_fusion_lane_center/evaluation|golden.json",
 "||?|Volume{x}|Auto-research|bev_fusion_lane_center|evaluation|golden.json",
 "||localhost|C$|Auto-research|bev_fusion_lane_center|evaluation|golden.json",
 "||127.0.0.1|c$|Auto-research|bev_fusion_lane_center|evaluation|golden.json",
 "||wsl.localhost|Ubuntu|mnt|c|Auto-research|bev_fusion_lane_center|evaluation|golden.json",
 "||?|UNC|wsl$|Ubuntu|mnt|c|Auto-research|bev_fusion_lane_center|evaluation|golden.json",
 "|Auto-research|bev_fusion_lane_center|evaluation|golden.json",
 "C:|Auto-research|bev_fusion_lane_center|evaluation|golden.json ",
 "C:|Auto-research|bev_fusion_lane_center|evaluation|golden.json:$DATA",
 "C:|Auto-research|bev_fusion_lane_center|evaluation|golden.json::$DATA",
 "C:|Auto-research|bev_fusion_lane_center|evaluation|golden.json:x",
 "||wsl$|Ubuntu|..|..|mnt|c|Auto-research|bev_fusion_lane_center|evaluation|golden.json",
]]
for c in cases:
    t = time.time()
    v = G.verdict("Write", {"file_path": c, "content": "x"}, cwd=cwd)
    print("REFUSED" if v else "PASSES ", f"{time.time()-t:5.2f}s", repr(c), "->", G.norm(c, cwd))
print("--- no event cwd (hook cwd = repo root) ---")
for c in ["evaluation|golden.json", "bev_fusion_lane_center/evaluation|golden.json", w("|Auto-research|bev_fusion_lane_center|evaluation|golden.json")]:
    v = G.verdict("Write", {"file_path": c, "content": "x"}, cwd=None)
    print("REFUSED" if v else "PASSES ", repr(c), os.getcwd())
print("--- ordinary paths, cwd windows ---")
for c in [w("C:|Auto-research|bev_fusion_lane_center|HANDOVER.md"), "/home/u/x.md", "/tmp/x", w("C:|Users|mmeut|x.md"), w("C:|Auto-research|bev_fusion_lane_center|experiment_execution|grants.csv")]:
    print(G.verdict("Write", {"file_path": c, "content": "x"}, cwd=cwd), repr(c), G.norm(c, cwd))
