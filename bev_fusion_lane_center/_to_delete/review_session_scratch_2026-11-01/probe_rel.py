import os, sys, tempfile
sys.path.insert(0, "autoresearch/tools")
import guard_hook as G, project as P
print("hook cwd", os.getcwd(), "problem root", P.root())
for c in ["evaluation/golden.json", "bev_fusion_lane_center/evaluation/golden.json"]:
    print("no cwd:", c, "->", "REFUSED" if G.verdict("Write", {"file_path": c, "content": "x"}) else "PASSES")
dc = "mcp__desktop-commander__"
print("write_pdf outputPath:", G.verdict(dc + "write_pdf", {"path": "C:/x/in.pdf", "outputPath": "C:/Auto-research/bev_fusion_lane_center/evaluation/golden.json", "content": []}, cwd="C:\Auto-research"))
os.environ["AR_MACHINE_LOCK"] = os.path.join(tempfile.mkdtemp(), "nostate", "machine_lock.json")
import machine_lock as m
try:
    m.release("me")
except Exception as e:
    print("release without state dir -> CRASH", type(e).__name__, e)
