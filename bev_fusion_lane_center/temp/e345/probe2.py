import os, sys, torch, time
R = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import jends_eval as JEV
import lane_gt_data as LD
torch.set_num_threads(2)
LD.QData.DEFAULT_JUNCTION = True
t = time.time()
ds, dl = LD.loader("val_clean", bs=8, workers=0, shuffle=False, map_mode="error", pixels=False)
print("ds", len(ds), ds.missing, time.time() - t)
it = iter(dl)
t = time.time(); b = next(it); print("batch", time.time() - t, sorted(k for k in b.keys())[:60])
print(b["x"].dtype, b["x"].shape, b["ent"].shape, b["pairs"].shape, b["row"][:3])
m, a = JEV.load_model("e312_p_s0", "cpu")
t = time.time()
with torch.no_grad():
    _, _, out = m(b["x"].float(), gt=None)
print("fwd", time.time() - t, {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)})
print(out["valid"].sum(), out["allowed"].sum(), out["kp_p"][0][:5])
dst, dtr = LD.loader("train", bs=8, workers=0, shuffle=False, map_mode="error", pixels=False)
print("train", len(dst), dst.missing)
import qgraph_eval as QE
print(QE.LS.B123)
