import os, sys, time
t0 = time.time()
ROOT = os.environ["BFLC_ROOT"]
for sub_ in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub_))
import torch
torch.set_num_threads(2)
import lane_tok_data as LT
print("import LT", time.time() - t0)
t0 = time.time()
import lane_past_data as LPD, lane_tokcache as LC, lane_ptok_data as LP, qpredict as QP, qgraph as QG, jtok as JT
print("import rest", time.time() - t0)
t0 = time.time()
td = LT.TokData("val", res=2.0, feats=True)
print("TokData val", time.time() - t0, len(td))
t0 = time.time()
o = td[0]
print("item0", time.time() - t0, o["x"].shape, o["x"].dtype, {k: (v.shape, v.dtype) for k, v in o.items() if hasattr(v, "shape")})
t0 = time.time()
q = td.qd[0]
print("qdata item", time.time() - t0, [k for k in q])
t0 = time.time()
m = QP.load_model(os.path.join(ROOT, "runs/e292_a_s0"), "cpu")
print("load_model", time.time() - t0, m.mnms, m.soft_nms)
