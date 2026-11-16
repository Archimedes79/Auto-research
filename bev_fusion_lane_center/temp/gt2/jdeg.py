import json
import sys

import numpy as np

sys.path.insert(0, "../../tools/viz")
sys.path.insert(0, "../../training/grid")
import qgraph_view as QV
import jgt as J

toks = []
for k, (name, tk) in QV.val_scenes().items():
    toks += tk[::3]
dout, din = [], []
for tok in toks:
    jg = J.junction_gt(QV.load_gt(tok))
    if jg is None or not len(jg["pairs"]):
        continue
    a = jg["pairs"][:, 0].astype(int)
    b = jg["pairs"][:, 1].astype(int)
    dout += list(np.bincount(a, minlength=len(jg["ent"]))[np.unique(a)])
    din += list(np.bincount(b, minlength=len(jg["ext"]))[np.unique(b)])
S = dict(entries=len(dout), out_degree_hist=np.bincount(dout).tolist(), out_degree_gt1=round(float(np.mean(np.array(dout) > 1)), 3), mean_out=round(float(np.mean(dout)), 2),
         exits=len(din), in_degree_hist=np.bincount(din).tolist(), in_degree_gt1=round(float(np.mean(np.array(din) > 1)), 3), mean_in=round(float(np.mean(din)), 2))
print(json.dumps(S))
json.dump(S, open("jdeg_val.json", "w"))
