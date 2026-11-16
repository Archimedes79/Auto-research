"""scratch (E282 critic pass, read-only): the E276 arm s checkpoint (runs/e276_s_s0_p3) on VAL frames on the CPU: is the existence logit at the GT cell of a free zone higher than at the other cells of the zone?
and what happens at the fixed zones (anchors)?
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 bash tools/env/_py14.sh s_ckpt_diag.py [step]
"""
import os
import sys
from collections import Counter

import numpy as np
import torch

torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for sub in ("training/train", "training/grid", "training/data", "training", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_gt_data as LD  # noqa: E402
import qgraph as QG  # noqa: E402
import qtargets as QT  # noqa: E402

step = int(sys.argv[1]) if len(sys.argv) > 1 else 24
LD.QData.DEFAULT_SPARSE = (0.3, 7.5)
ds = LD.QData("val_clean", pixels=False, map_mode="exact")
model = QG.QGraph(width=48)
sd = torch.load(os.path.join(ROOT, "runs", "e276_s_s0_p3", "model.pt"), map_location="cpu")["state"]
print("load", model.load_state_dict(sd, strict=False))
model.eval()
res = Counter()
rows = []
for i in range(0, len(ds), step):
    k = ds.ks[i]
    row = int(ds.bd.rows[k])
    token = ds.tokens[row]
    g = ds.lg.load(token)
    nd = QT.node_table(g)
    ns = QT.node_table(g, sparse=(0.3, 7.5))
    x = torch.from_numpy(np.asarray(ds[i]["x"], np.float32))[None]
    with torch.no_grad():
        P = model.pyr(x)
        no = model.node_stage(P)
    el = no["e"][0].numpy()           # (50, 50) logits
    gd, gs = nd["gid"], ns["gid"]
    pieces, off = nd["piece"], nd["off"]
    anchors = set()
    for p in np.unique(pieces):
        ids = np.where(pieces == p)[0]
        anchors.add(int(gd[ids[0]]))
        anchors.add(int(gd[ids[-1]]))
    evset = set(int(v) for v in gd[nd["ev"]])
    anchors |= evset
    sset = set(int(v) for v in gs)
    cell_of_gid = {int(a): tuple(c) for a, c in zip(gd, nd["cell"])}
    sparse_vis = {int(a): bool(v) for a, v in zip(gs, ns["vis"])}
    zone_cells = {}
    nozone_cells = []
    for p in np.unique(pieces):
        ids = [int(v) for v in gd[pieces == p]]
        sp = sorted(int(v) for v in gs if off[p] <= v < off[p + 1])
        for gid in ids:
            d = min((abs(gid - s), s) for s in sp)
            s = d[1]
            if gid == s:
                zone_cells.setdefault(s, []).append(gid)
            elif s in anchors:
                nozone_cells.append(gid)
            else:
                zone_cells.setdefault(s, []).append(gid)
    for s, ids in zone_cells.items():
        lg = np.array([el[cell_of_gid[v]] for v in ids])
        gt_pos = ids.index(s)
        res["zones"] += 1
        if s in anchors:
            kind = "event" if s in evset else "pieceend"
            res[f"fixed_{kind}_n"] += 1
            res[f"fixed_{kind}_p>=0.5"] += int(lg[gt_pos] > 0)
            res[f"fixed_{kind}_p>=0.3"] += int(lg[gt_pos] > np.log(0.3 / 0.7))
            continue
        res["free_n"] += 1
        n = len(ids)
        res["free_chance"] += 1.0 / n
        res["free_argmax_is_gt"] += int(np.argmax(lg) == gt_pos)
        order = (-lg).argsort()
        res["free_rank_mean"] += float(np.where(order == gt_pos)[0][0])
        res["free_maxp>=0.5"] += int(lg.max() > 0)
        res["free_maxp>=0.3"] += int(lg.max() > np.log(0.3 / 0.7))
        res["free_gt_p>=0.5"] += int(lg[gt_pos] > 0)
        res["free_ncells_p>=0.5"] += int((lg > 0).sum())
        res["free_ncells_p>=0.3"] += int((lg > np.log(0.3 / 0.7)).sum())
        res["free_cells"] += n
        w = lg + 0.5 * (np.arange(n) == gt_pos)
        res["free_winner_is_gt"] += int(np.argmax(w) == gt_pos)
        res["free_gt_minus_best_other_mean"] += float(lg[gt_pos] - np.delete(lg, gt_pos).max()) if n > 1 else 0.0
        res["free_n_multi"] += int(n > 1)
        # the spread inside the zone: max-min logit
        res["free_spread_mean"] += float(lg.max() - lg.min())
        res["free_vis" if sparse_vis[s] else "free_unvis"] += 1
        res["free_vis_argmax_gt" if sparse_vis[s] else "free_unvis_argmax_gt"] += int(np.argmax(lg) == gt_pos)
    if nozone_cells:
        lg = np.array([el[cell_of_gid[v]] for v in nozone_cells])
        res["nozone_n"] += len(lg)
        res["nozone_p>=0.5"] += int((lg > 0).sum())
        res["nozone_p>=0.3"] += int((lg > np.log(0.3 / 0.7)).sum())
    allp = 1 / (1 + np.exp(-el))
    res["frames"] += 1
    res["cells_p>=0.5"] += int((allp >= 0.5).sum())
    res["cells_p>=0.3"] += int((allp >= 0.3).sum())
    # off-lane: cells not holding any dense node
    mask = np.ones((50, 50), bool)
    for c in cell_of_gid.values():
        mask[c] = False
    res["offlane_p>=0.3"] += int((allp[mask] >= 0.3).sum())
    res["offlane_p>=0.5"] += int((allp[mask] >= 0.5).sum())
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
n = res["free_n"]
print("free zones", n, "mean cells", res["free_cells"] / n, "chance of argmax==GT (mean 1/n)", res["free_chance"] / n)
print("argmax logit cell is the GT cell:", res["free_argmax_is_gt"] / n, " rank of GT cell (0 = best) mean", res["free_rank_mean"] / n)
print("GT cell logit minus best other cell (mean, logits):", res["free_gt_minus_best_other_mean"] / res["free_n_multi"])
print("winner by (logit + 0.5 at GT) is the GT cell:", res["free_winner_is_gt"] / n)
print("zone max p>=0.5:", res["free_maxp>=0.5"] / n, " p>=0.3:", res["free_maxp>=0.3"] / n, " GT cell p>=0.5:", res["free_gt_p>=0.5"] / n)
print("cells per zone with p>=0.5:", res["free_ncells_p>=0.5"] / n, " p>=0.3:", res["free_ncells_p>=0.3"] / n)
print("spread (max-min logit) inside a free zone mean:", res["free_spread_mean"] / n)
print("demanded free zones argmax==GT:", res["free_vis_argmax_gt"] / max(res["free_vis"], 1), "n", res["free_vis"], "; unseen:", res["free_unvis_argmax_gt"] / max(res["free_unvis"], 1), "n", res["free_unvis"])
for kind in ("event", "pieceend"):
    nn_ = res[f"fixed_{kind}_n"]
    print(f"fixed zones {kind}: n {nn_}, p>=0.5 {res[f'fixed_{kind}_p>=0.5'] / max(nn_, 1):.3f}, p>=0.3 {res[f'fixed_{kind}_p>=0.3'] / max(nn_, 1):.3f}")
print("no-zone dense cells: n", res["nozone_n"], "p>=0.5", res["nozone_p>=0.5"] / max(res["nozone_n"], 1), "p>=0.3", res["nozone_p>=0.3"] / max(res["nozone_n"], 1))
print("per frame cells p>=0.5", res["cells_p>=0.5"] / res["frames"], "p>=0.3", res["cells_p>=0.3"] / res["frames"], "off-lane p>=0.3", res["offlane_p>=0.3"] / res["frames"])
