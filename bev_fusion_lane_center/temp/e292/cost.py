"""E292 scratch: ms of the Matrix-NMS layer (torch, CPU, 1 thread, 50 x 50 grid) against the hard close-node step of the read-out (numpy, per frame, on the stored dump of B), and the flops of the layer by hand."""
import os, pickle, sys, time
import numpy as np
import torch
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("evaluation", "training", "training/grid", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
torch.set_num_threads(1)
import qmnms as QM
import qgraph_readout as QR
cell = torch.stack(torch.meshgrid(-50 + 2 * (torch.arange(50).float() + 0.5), -50 + 2 * (torch.arange(50).float() + 0.5), indexing="ij"), 0)[None]
e = torch.randn(1, 50, 50) * 2.5
pos = cell + (torch.rand(1, 2, 50, 50) * 2 - 1)
for _ in range(5):
    QM.matrix_nms_logit(e, pos)
t0 = time.time()
for _ in range(100):
    QM.matrix_nms_logit(e, pos)
print(f"matrix_nms_logit torch CPU 1 thread, 1 x 50 x 50: {(time.time() - t0) * 10:.2f} ms")
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", "predfull_e273_p_s0.pkl"), "rb"))
spec = QR.parse("close:1.5@0.5")
ts = []
for d in dump[:200]:
    pred = dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")), links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"]))
    t0 = time.time(); QR.apply(pred, spec); ts.append(time.time() - t0)
print(f"hard close step QR.apply(close:1.5@0.5) per frame (numpy, python loop, 200 frames of the dump): mean {1000 * np.mean(ts):.2f} ms, p90 {1000 * np.percentile(ts, 90):.2f} ms")
print("nodes per frame with e >= 0.5 (mean):", np.mean([(np.asarray(d["e"]).reshape(-1) >= 0.5).sum() for d in dump[:200]]))
