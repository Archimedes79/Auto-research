import numpy as np
import qgraph_eval as QE
fr = QE.val_frames()[::29]
gaps = []; ns = 0
for f in fr[:12]:
    g = f.lane()
    idx = {p["id"]: i for i, p in enumerate(g["pieces"])}
    for p in g["pieces"]:
        if not len(p["pts"]): continue
        for q in p["succ"]:
            if q in idx and len(g["pieces"][idx[q]]["pts"]):
                gaps.append(np.linalg.norm(np.asarray(p["pts"])[-1] - np.asarray(g["pieces"][idx[q]]["pts"])[0]))
    f._lane = None
gaps = np.array(gaps)
print(len(gaps), "succ joins: gap quantiles", np.percentile(gaps, [0, 10, 50, 90, 100]))
# spacing of points
sp = []
g = fr[3].lane()
for p in g["pieces"][:10]:
    q = np.asarray(p["pts"])
    if len(q) > 1: sp.append(np.linalg.norm(np.diff(q, axis=0), axis=1))
sp = np.concatenate(sp); print("spacing", np.percentile(sp, [0, 50, 100]))
print("pieces", len(g["pieces"]), "connector segments", sum(s["kind"] == "connector" for p in g["pieces"] for s in p["segments"]), "pieces mixed", sum(len({s["kind"] for s in p["segments"]}) > 1 for p in g["pieces"]))
