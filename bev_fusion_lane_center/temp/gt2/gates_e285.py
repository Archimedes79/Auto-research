import json
import sys

import numpy as np
import torch

sys.path.insert(0, "../../tools/viz")
sys.path.insert(0, "../../training/grid")
import jgt as J
import jmaps as JM
import qends as QE
import qgraph_view as QV

toks = []
for k, (name, tk) in QV.val_scenes().items():
    toks += tk
R = dict(frames=len(toks), junction=0, maneuvers=0, multi=0, mono=0, ident=0, ent=0, ent_found=0, ext=0, ext_found=0, spread_gt3=0, arms=0, pairs_mismatch=0, errs=0)
for tok in toks:
    g = QV.load_gt(tok)
    try:
        jg = J.junction_gt(g)
    except Exception as e:                                                           # a frame that breaks the layer is a failed gate
        R["errs"] += 1
        print("error", tok, repr(e))
        continue
    if jg is None:
        continue
    R["junction"] += 1
    o = jg["order"]
    R["maneuvers"] += o["maneuvers"]; R["multi"] += o["multi"]; R["mono"] += o["monotone"]; R["ident"] += o["identity"]
    for a in jg["arms"]:
        R["arms"] += 1
        R["spread_gt3"] += int(max(a["spread_in"], a["spread_out"]) > 3.0)
    et = QE.end_table(g)
    nt = 0 if et is None else len(et["pairs"])
    if et is not None and not np.array_equal(np.array(sorted(map(tuple, et["pairs"].astype(int).tolist()))), np.array(sorted(map(tuple, jg["pairs"][:, :2].astype(int).tolist())))):
        R["pairs_mismatch"] += 1
    # decode gate: the heat maps give back the ends
    zone, heat, head = J.fields(jg)
    j1 = torch.full((1, 8, 100, 100), -6.0)
    j1[0, 2:4] = torch.from_numpy(np.log(np.clip(heat, 1e-4, 1 - 1e-4) / (1 - np.clip(heat, 1e-4, 1 - 1e-4))))
    kp = JM.keypoints(dict(j1=j1), thr=0.5)[0]
    for name, E, k in (("ent", jg["ent"], kp["ent"].numpy()), ("ext", jg["ext"], kp["ext"].numpy())):
        R[name] += len(E)
        for e in E:
            if len(k) and np.linalg.norm(k[:, :2] - e[:2], axis=1).min() <= 1.0:
                R[name + "_found"] += 1
S = dict(R, monotone_share=R["mono"] / max(R["multi"], 1), identity_share=R["ident"] / max(R["multi"], 1), arms_spread_gt3_share=R["spread_gt3"] / max(R["arms"], 1),
         decode_recall_ent=R["ent_found"] / max(R["ent"], 1), decode_recall_ext=R["ext_found"] / max(R["ext"], 1))
print(json.dumps(S, indent=1))
json.dump(S, open("gates_e285_val.json", "w"), indent=1)
