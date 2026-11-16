import os, sys, time, pickle
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train"):
    sys.path.insert(0, os.path.join(ROOT, s))
import lane_gt_data as LD
import qpredict as QP
import qgraph_readout as RO
import jends_eval as JEV
S = 0
t0 = time.time()
m = QP.load_model(os.path.join(ROOT, "runs", f"e273_p_s{S}"), "cpu")
jm, _ = JEV.load_model(f"e324_s_s{S}", "cpu")
print("loaded", time.time() - t0)
# pyramid equality
sd_b = m.state_dict(); sd_j = jm.state_dict()
ks = [k for k in sd_b if k.startswith("pyr.")]
print("pyr tensors in B", len(ks), "equal:", all(torch.equal(sd_b[k], sd_j["jmodel." + k]) for k in ks))
nk = [k for k in sd_b if k.startswith("nodes.")]
print("node head tensors", len(nk), "equal to the one in E324 s:", all(torch.equal(sd_b[k], sd_j[k]) for k in nk))
ds, dl = LD.loader("val_clean", bs=4, workers=0, shuffle=False, map_mode="exact", pixels=False, strict=True)
rows = [int(ds.bd.rows[k]) for k in ds.ks]
preds, tb = [], []
it = iter(dl)
for i in range(6):
    b = next(it)
    t1 = time.time()
    preds += QP.predict(m, b)
    tb.append(time.time() - t1)
print("B forward bs4 per batch (s):", [round(t, 1) for t in tb])
RO.dump_full(preds, rows[:len(preds)], "/tmp/e352_eq.pkl")
ref = pickle.load(open(os.path.join(ROOT, "temp/e276/predfull_e273_p_s0.pkl"), "rb"))[:len(preds)]
new = pickle.load(open("/tmp/e352_eq.pkl", "rb"))
mx = {}
for a, b in zip(ref, new):
    assert a["row"] == b["row"]
    for k in a:
        if isinstance(a[k], np.ndarray):
            if a[k].shape != b[k].shape:
                mx.setdefault(k, []).append("SHAPE %s %s" % (a[k].shape, b[k].shape)); continue
            mx.setdefault(k, []).append(float(np.abs(a[k].astype(float) - b[k].astype(float)).max()) if a[k].size else 0.0)
print({k: (max(v) if not isinstance(v[0], str) else v[:3]) for k, v in mx.items()})
print([(len(a["e"]), len(b["e"]), len(a["link_p"]), len(b["link_p"])) for a, b in zip(ref, new)])
