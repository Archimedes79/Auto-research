"""one-off (E275): over the 873 VAL frames, the share of GT nodes that have another GT node within 1.5 m (the recall the close-node step cannot keep), the histogram of the nearest-neighbour distance, by event flag"""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE
fr = QE.val_frames()
n = tot = near = ev_tot = ev_near = 0
nn_all = []
per_frame = []
for k, f in enumerate(fr):
    nt = f.ref().nt
    G = nt["pos"]; ev = np.asarray(nt["flags"]).any(1)
    f._lane = f._ref = None
    if len(G) < 2:
        continue
    d = cKDTree(G).query(G, k=2)[0][:, 1]
    nn_all.append(d); per_frame.append(float((d < 1.5).mean()))
    n += 1; tot += len(G); near += int((d < 1.5).sum()); ev_tot += int(ev.sum()); ev_near += int(((d < 1.5) & ev).sum())
d = np.concatenate(nn_all)
out = dict(frames=n, gt_nodes=tot, share_within_1_5=near / tot, event_nodes=ev_tot, event_share_within_1_5=ev_near / max(ev_tot, 1),
           hist={f"{a}-{b}": int(((d >= a) & (d < b)).sum()) for a, b in ((0, 0.01), (0.01, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 99))},
           per_frame_share_quartiles=[float(x) for x in np.percentile(per_frame, [5, 25, 50, 75, 95])])
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(ROOT, "temp", "e275", "gt_pairs.json"), "w"), indent=1)
