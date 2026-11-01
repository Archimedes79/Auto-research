"""read-only CPU diagnostic for the E263 / E264 cards: where do the far (>= 40 m) misses of the recall chains lie, and were they seen nearer in the last 1-3 frames?
also: how many of the top-64 nodes (the prior of the time recursion) are far, against all valid nodes"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation"))
import evaluate as EV
R = os.environ["BFLC_ROOT"]
reg = EV.registry()
rows = np.array(json.load(open(f"{R}/data/splits.json"))["val"])
P = dict(np.load(os.path.expanduser("~/exp_gcar_b123/poses_val.npz")))
row2k = {int(r): k for k, r in enumerate(rows)}

def dense(pl, step=0.25):
    out = []
    for p in pl:
        p = np.asarray(p, float)
        if len(p) == 1:
            out.append(p); continue
        for a, b in zip(p[:-1], p[1:]):
            n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
            out.append(a + (b - a) * np.linspace(0, 1, n + 1)[:, None])
    return np.concatenate(out) if out else np.zeros((0, 2))

for name in sys.argv[1:]:
    g, gts, info = EV.graphs_of(name, reg[name], "val", require_cache=True)
    raw, _, _ = EV.builder_raw(reg[name], "val", require_cache=True)
    cnt = dict(hit=0, miss=0, miss_front=0, miss_seen30=0, miss_seen40=0, hit_seen40=0, miss_rear_x=0)
    far_top64, far_valid, n64, nval = 0, 0, 0, 0
    for k, (pg, gt) in enumerate(zip(g, gts)):
        r = int(rows[k])
        gp = dense(gt["pts"])
        if len(gp) == 0:
            continue
        far = np.linalg.norm(gp, axis=1) >= 40
        if not far.any():
            continue
        q = dense(pg["pts"]) if len(pg["pts"]) else np.zeros((0, 2))
        fp = gp[far]
        if len(q):
            d = np.sqrt(((fp[:, None, :] - q[None, :, :]) ** 2).sum(-1)).min(1)
        else:
            d = np.full(len(fp), 1e9)
        hit = d <= 1.0
        # was the point within 30 / 40 m of the ego in one of the 3 previous frames of the scene?
        pw = fp @ P["R"][r].T + P["t"][r]
        seen30 = np.zeros(len(fp), bool); seen40 = np.zeros(len(fp), bool)
        pr = r
        for _ in range(3):
            pr = int(P["prev"][pr])
            if pr < 0:
                break
            pk = (pw - P["t"][pr]) @ P["R"][pr]
            dk = np.linalg.norm(pk, axis=1)
            seen30 |= dk < 30; seen40 |= dk < 40
        cnt["hit"] += int(hit.sum()); cnt["miss"] += int((~hit).sum())
        cnt["miss_front"] += int((~hit & (fp[:, 0] > 0)).sum())
        cnt["miss_seen30"] += int((~hit & seen30).sum()); cnt["miss_seen40"] += int((~hit & seen40).sum())
        cnt["hit_seen40"] += int((hit & seen40).sum())
        rw = raw[k]
        s = rw["pi"] * rw["valid"]
        top = np.argsort(-s)[:64]
        rad = np.linalg.norm(rw["pos"], axis=1)
        far_top64 += int((rad[top] >= 40).sum()); n64 += len(top)
        v = rw["valid"] > 0.5
        far_valid += int((rad[v] >= 40).sum()); nval += int(v.sum())
    m = cnt["miss"]
    print(f"== {name}: far points (0.25 m) hit {cnt['hit']} miss {m} -> R@1_far approx {cnt['hit']/(cnt['hit']+m):.3f}")
    print(f"   misses with x > 0 (front, if x is forward): {cnt['miss_front']/m:.3f}")
    print(f"   misses seen within 40 m in t-1..t-3: {cnt['miss_seen40']/m:.3f}; within 30 m: {cnt['miss_seen30']/m:.3f}; hits seen within 40 m: {cnt['hit_seen40']/max(cnt['hit'],1):.3f}")
    print(f"   far share of the top-64 prior nodes {far_top64/n64:.3f}; far share of all valid raw nodes {far_valid/max(nval,1):.3f} (valid per frame {nval/len(g):.0f})")
