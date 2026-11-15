import os, sys, time, pickle
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(__file__))
import heat_replay as H
import numpy as np, torch
import lane_gt_data as LD
import jends_eval as JEV
t0 = time.time()
m, _ = JEV.load_model("e312_p_s0", "cpu")
LD.QData.DEFAULT_JUNCTION = True
ds, dl = LD.loader("val_clean", bs=8, workers=0, shuffle=False, map_mode="error", pixels=False)
ends = {f["token"]: f for f in pickle.load(open(os.path.join(H.ROOT, "runs", "e312_p_s0", "ends_val.pkl"), "rb"))}
print("load", time.time() - t0, flush=True)
b = next(iter(dl))
t1 = time.time()
with torch.no_grad():
    _, out = m.jmodel(b["x"].float(), joint=False)
    full = m(b["x"].float(), gt=None)[2]
print("forward bs8", time.time() - t1, flush=True)
j1 = out["j1"].float()
p = j1[:, 2:4].sigmoid()
dv = torch.stack([H.JE._unit(j1[:, 4:6]), H.JE._unit(j1[:, 6:8])], 1)
for i in range(2):
    pos, D, val = H.extract(p[i], dv[i])
    e = ends[b["token"][i]]
    print("frame", i, "valid", val.sum(), np.asarray(e["valid"]).sum())
    print(" mine ", pos[val][:4].round(2).tolist())
    print(" dump ", np.asarray(e["kp_pos"])[np.asarray(e["valid"]).reshape(-1)][:4].round(2).tolist())
    print(" fwd  ", full["kp_pos"][i][full["valid"][i].reshape(-1)][:4].numpy().round(2).tolist())
    print(" valid shapes", np.asarray(e["valid"]).shape, val.shape, "dump kp shape", np.asarray(e["kp_pos"]).shape)
