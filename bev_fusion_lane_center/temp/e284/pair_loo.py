"""scratch (continuation of pair_geometry.py, Mirko's OK of 2026-11-08): the pair classifier reaches AUC 0.995 on VAL, but 669 of 825 VAL junction frames show a junction that TRAIN has shown too (same map location, entries within 5 m, same heading): the
classifier may only have learned the junction layouts of the four nuScenes maps. Test: LEAVE-JUNCTION-OUT. All TRAIN and VAL frames are linked when their junction centres (mean of the entry positions in the global frame) lie within 40 m at the same map location
(connected components = physical junctions). (1) out-of-fold AUC on TRAIN with the folds made of whole junctions; (2) the classifier trained on all TRAIN, tested on the VAL frames whose component has NO TRAIN frame ('new junction').
Both also for the complex junctions only (frames with >= 20 candidate pairs). Needs the functions of pair_geometry.py (its part before the TRAIN section is executed here).
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 bash tools/env/_py14.sh temp/e284/pair_loo.py"""
import json
import os

src = open(os.path.join(os.environ["BFLC_ROOT"], "temp", "e284", "pair_geometry.py")).read().split("# ---- TRAIN set")[0]
exec(compile(src, "pair_geometry_head", "exec"))          # imports, c / cells, hermite, cluster, allowed, sang, side, analyse

POSE = {part: np.load(os.path.expanduser(f"~/exp_gcar_b123/poses_{part}.npz")) for part in ("train", "val")}
ds = LD.QData("train", map_mode="exact", pixels=False)
ks_tr = list(ds.ks)[::3]
Xs, ys, grp_frame, cen, loc, part_of, ncand = [], [], [], [], [], [], []
rows_X = []                                                       # per frame: (X, y)
for i, k in enumerate(ks_tr):
    row = int(ds.bd.rows[k])
    r = analyse(ds.lg.load(ds.tokens[row]))
    if r is None:
        continue
    gp = np.array([e[0] for e in r["ent"]]) @ POSE["train"]["R"][row].T + POSE["train"]["t"][row]
    rows_X.append((r["X"], r["y"]))
    cen.append(gp.mean(0))
    loc.append(int(POSE["train"]["loc"][row]))
    part_of.append("train")
    if (i + 1) % 500 == 0:
        print(i + 1, "train frames", flush=True)
frames = QE.val_frames()
val_idx = []
for i, f in enumerate(frames):
    r = analyse(f.lane())
    f._lane = f._ref = None
    if r is None:
        continue
    gp = np.array([e[0] for e in r["ent"]]) @ POSE["val"]["R"][f.row].T + POSE["val"]["t"][f.row]
    rows_X.append((r["X"], r["y"]))
    cen.append(gp.mean(0))
    loc.append(int(POSE["val"]["loc"][f.row]))
    part_of.append("val")
    if (i + 1) % 300 == 0:
        print(i + 1, "val frames", flush=True)
cen, loc, part_of = np.array(cen), np.array(loc), np.array(part_of)
# connected components of frames within 40 m at the same map location
parent = list(range(len(cen)))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


for lc in np.unique(loc):
    idx = np.where(loc == lc)[0]
    for a, b in cKDTree(cen[idx]).query_pairs(40.0):
        ra, rb = find(idx[a]), find(idx[b])
        if ra != rb:
            parent[ra] = rb
comp = np.array([find(i) for i in range(len(cen))])
print("frames", len(cen), "components (physical junctions)", len(np.unique(comp)), "of which with TRAIN frames", len(np.unique(comp[part_of == "train"])), flush=True)
mk = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, random_state=0)  # noqa: E731
tr_idx = np.where(part_of == "train")[0]
va_idx = np.where(part_of == "val")[0]
res = dict(frames=int(len(cen)), junctions=int(len(np.unique(comp))), junctions_with_train=int(len(np.unique(comp[part_of == "train"]))))


def stack(ids):
    return np.concatenate([rows_X[i][0] for i in ids]), np.concatenate([rows_X[i][1] for i in ids]), np.concatenate([np.full(len(rows_X[i][1]), len(rows_X[i][1])) for i in ids])


# (1) out-of-fold on TRAIN, folds = whole junctions
ucomp = np.unique(comp[tr_idx])
rng = np.random.RandomState(0)
fold_of = dict(zip(ucomp, rng.randint(0, 5, len(ucomp))))
Xt, yt, nct = stack(tr_idx)
fo = np.concatenate([np.full(len(rows_X[i][1]), fold_of[comp[i]]) for i in tr_idx])
oof = np.zeros(len(yt))
for f_ in range(5):
    m = mk().fit(Xt[fo != f_], yt[fo != f_])
    oof[fo == f_] = m.predict_proba(Xt[fo == f_])[:, 1]


def report(name, y, p, nc, tau=0.42):
    out = {}
    for lab, sel in (("all", np.ones(len(y), bool)), ("complex (>= 20 candidates in the frame)", nc >= 20)):
        if sel.sum() < 50 or y[sel].sum() in (0, sel.sum()):
            continue
        pr = (p >= tau) & sel
        out[lab] = dict(candidates=int(sel.sum()), true=int(y[sel].sum()), precision_all=float(y[sel].mean()), auc=float(roc_auc_score(y[sel], p[sel])),
                        precision_at_tau=float((pr & (y == 1)).sum() / max(pr.sum(), 1)), recall_at_tau=float((pr & (y == 1)).sum() / max(y[sel].sum(), 1)))
        print(f"  {name:38s} {lab:40s}", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in out[lab].items()}), flush=True)
    return out


res["train_leave_junction_out"] = report("TRAIN, folds of whole junctions", yt, oof, nct)
# (2) trained on all TRAIN, tested on VAL, split by 'new junction' (component without a TRAIN frame)
model = mk().fit(Xt, yt)
train_comps = set(comp[tr_idx])
for name, ids in (("VAL, junction new (no TRAIN frame within 40 m)", [i for i in va_idx if comp[i] not in train_comps]), ("VAL, junction also in TRAIN", [i for i in va_idx if comp[i] in train_comps])):
    if not ids:
        print("  ", name, "no frames")
        continue
    Xv, yv, ncv = stack(ids)
    res[name] = dict(frames=len(ids), **report(name, yv, model.predict_proba(Xv)[:, 1], ncv))
    print("   frames:", len(ids))
json.dump(res, open(os.path.join(os.environ["BFLC_ROOT"], "temp", "e284", "pair_loo.json"), "w"), indent=1)
