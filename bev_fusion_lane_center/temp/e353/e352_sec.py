"""why do the section starts of 24 frames differ between the CPU dump and the stored GPU dump? the margin between the two largest entries of the membership row M (the section start is its argmax) of the nodes concerned"""
import os, sys, pickle
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train"):
    sys.path.insert(0, os.path.join(ROOT, s))
import jprune_b as PB
import lane_gt_data as LD
import qpredict as QP
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
n = 24
m = QP.load_model(os.path.join(ROOT, "runs", f"e273_p_s{seed}"), "cpu")
ref = pickle.load(open(os.path.join(ROOT, "temp/e276", f"predfull_e273_p_s{seed}.pkl"), "rb"))[:n]
ds, dl = LD.loader("val_clean", bs=4, workers=0, shuffle=False, map_mode="exact", pixels=False, strict=True)
gaps_all, gaps_diff, nd, nm = [], [], 0, 0
i = 0
with torch.no_grad(), PB.cpu_solve_fix("cpu"):
    for b in dl:
        out = m(b, upto="assign", frozen=("pyr", "nodes", "links", "sections", "assign"), gt=False)
        preds = QP.predict_from_out(out, b)
        import qgraph_readout as RO
        rows = [int(ds.bd.rows[k]) for k in ds.ks][i:i + 4]
        RO.dump_full(preds, rows, "/tmp/e352_sec.pkl")
        new = pickle.load(open("/tmp/e352_sec.pkl", "rb"))
        M = out["M"]
        for j in range(len(new)):
            x, y = ref[i + j], new[j]
            ia, ib = PB.match_nodes(x, y)
            where_a = -np.ones(int(max(x["idx"].max(), np.asarray(x["section"]).max())) + 1, int); where_a[x["idx"]] = np.arange(len(x["idx"]))
            where_b = -np.ones(int(max(y["idx"].max(), np.asarray(y["section"]).max())) + 1, int); where_b[y["idx"]] = np.arange(len(y["idx"]))
            mapa = -np.ones(len(x["e"]), int); mapa[ia] = ib
            sa, sb = where_a[np.asarray(x["section"])[ia]], where_b[np.asarray(y["section"])[ib]]
            for t, (u, v) in enumerate(zip(sa, sb)):
                row = M[j][int(y["idx"][ib[t]])]
                top = torch.topk(row, 2).values
                gap = float(top[0] - top[1])
                differs = (u >= 0) != (v >= 0) or (u >= 0 and v >= 0 and mapa[u] != v)
                gaps_all.append(gap); nm += 1
                if differs:
                    gaps_diff.append(gap); nd += 1
        i += 4
        if i >= n:
            break
ga, gd = np.array(gaps_all), np.array(gaps_diff)
print(f"seed {seed}: {nm} matched nodes in {n} frames, {nd} with another section start than the stored GPU dump")
print("margin between the two largest M entries (this CPU run): all nodes: share < 1e-3 =", float((ga < 1e-3).mean()), ", share < 1e-2 =", float((ga < 1e-2).mean()))
print("  nodes with another section start: n =", len(gd), "margins", np.round(np.sort(gd), 5).tolist()[:20], "max", float(gd.max()) if len(gd) else None)
