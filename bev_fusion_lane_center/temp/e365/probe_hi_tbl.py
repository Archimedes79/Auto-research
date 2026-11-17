import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation"))
import numpy as np, torch
import jtoken_eval as E
import lane_tok_data as TD
ds = E.open_cache("val", 0)
td = TD.TokData("val", 2.0, feats=False)
b = next(iter(E.batches(ds, range(3), bs=3)))
for i in range(3):
    t = b["tokens_np"][i]; gid = b["label"]["gt_id"][i].numpy(); y = b["y"][i].numpy()
    a = gid >= 0
    tab = td.table(i)
    ii, jj = np.nonzero(y.astype(bool) & a[:, None] & a[None, :])
    # table based: GT links contracted over unassigned GT nodes
    sub = E.JV.sub_table(tab, np.isin(np.arange(len(tab["pos"])), gid[a]))
    n_tab = len(sub["src"])
    # y pairs that are real GT successor of the source's node
    gt_pairs = set(zip(tab["src"].tolist(), tab["dst"].tolist()))
    good = sum((int(gid[p]), int(gid[q])) in gt_pairs for p, q in zip(ii, jj))
    print(f"frame {i}: valid {int(t['valid'].sum())}, assigned {int(a.sum())}, GT nodes {len(tab['pos'])}, GT links {len(tab['src'])}, contracted-over-unassigned {n_tab}, y pairs assigned-assigned {len(ii)}, of them a direct GT link {good}, y pairs total {int(y.sum())}")
