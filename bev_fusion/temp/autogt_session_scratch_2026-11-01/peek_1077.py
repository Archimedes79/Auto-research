"""scratch: render scene-1077 keyframe 20 from the stored GT and list the boxes ahead of the ego car"""
import os
import sys
import numpy as np
sys.path.insert(0, ".")
import autogt
from autogt import view

OUT = "/mnt/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/5472e7ce-e594-4b06-a345-258377292ebc/scratchpad"
tok = autogt.scene_tokens("scene-1077", "v1.0-trainval")[20]
g = autogt.load(tok)
print("token", tok)
p = view.stored(tok, OUT)
print(p)
print("boxes (ego frame), sorted by distance ahead:")
rows = []
for i, b in enumerate(g["meta"]["boxes"]):
    cx, cy, cz = b["center"]
    sp = float(np.hypot(*b["vel"]))
    rows.append((cx, i, b["group"], b["observed"], b["stationary"], b["vel_known"], b["vel_flag"], b["vel_source"], round(cy, 1), round(sp * 3.6, 1), b["visibility"], round(b["size"][0], 1), round(b["size"][1], 1)))
for r in sorted(rows)[:40]:
    print("x=%5.1f i=%2d %-11s observed=%-5s stationary=%-5s vel_known=%-5s flag=%-9s src=%-8s y=%5.1f v=%5.1f km/h vis=%s size=%s x %s" % r)
inst = g["instance"]
print("cells with velocity weight > 0:", int((g["flow_weight"] > 0).sum()), "| moving cells (speed>1):", int(((np.hypot(*g["velocity"]) > 1) & (g["flow_weight"] > 0)).sum()))
