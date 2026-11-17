import os, sys, time, copy
ROOT = os.environ["BFLC_ROOT"]
import numpy as np, torch
sys.path.insert(0, os.path.join(ROOT, "training/data"))
import lane_live_tok as LL
import lane_tokcache as LC
torch.set_num_threads(2)
B = 4
ds = LL.LiveData("val", 0)
b = LL.collate([ds[i] for i in range(B)])
m32 = LL.QP.load_model(os.path.join(ROOT, "runs/e292_a_s0"), "cpu")
m64 = copy.deepcopy(m32).double()
with torch.no_grad():
    n32 = m32.node_stage(m32.pyr(b["x"].float()))
    n64 = m64.node_stage(m64.pyr(b["x"].double()))
print("CPU float32 vs float64 node-stage outputs, max |diff| over the whole 50 x 50 maps:")
for k in n32:
    print("  ", k, "%.2e" % float((n32[k].double() - n64[k]).abs().max()), " (value range %.1f)" % float(n64[k].abs().max()))
# the top-512 cell lists
t32, t64 = LC.select_tokens(n32, None, 512), LC.select_tokens({k: v.float() for k, v in n64.items()}, None, 512)
print("same top-512 cell list on", sum(bool((t32["cell"][i] == t64["cell"][i]).all()) for i in range(B)), "of", B, "frames")
# the whole-label step on the CPU, batch 8 (second call: no import cost)
ds8 = [ds[i] for i in range(8)]
b8 = LL.collate(ds8)
with torch.no_grad():
    no8 = m32.node_stage(m32.pyr(b8["x"].float()))
tk8 = LL.live_tokens(no8, 512)
import jtok as JT
LL.label_batch(tk8, b8["table"], b8["token"], b8["row"], 512)
t0 = time.perf_counter(); LL.label_batch(tk8, b8["table"], b8["token"], b8["row"], 512); t1 = time.perf_counter() - t0
t0 = time.perf_counter(); JT.pair_geometry(tk8["pos"], tk8["dirv"], tk8["valid"]); t2 = time.perf_counter() - t0
_, _, secs = LL.label_batch(tk8, b8["table"], b8["token"], b8["row"], 512)
print("label_batch bs 8 on the CPU, whole (incl. pair_geometry on the CPU): %.3f s; pair_geometry alone on the CPU %.3f s; the per-frame loop %.3f s" % (t1, t2, secs))
import torch as T
print("threads", T.get_num_threads(), "load", os.getloadavg())
