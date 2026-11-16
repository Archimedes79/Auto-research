"""E292 COST (card section 0, verdict cells): the delivered path WITH the Matrix-NMS layer (read out plain) against the path with the hard close-node step, per frame, 95 % interval of (a - c).
GPU part: the layer on the 50 x 50 node grid (`qmnms.matrix_nms_logit`) timed with CUDA events in blocks A B | B A (A = layer, B = nothing), batch 1; host part: the hard step `qgraph_readout.apply` with the
spec close:1.5@0.5 against plain@0.5 on real frames of B's full dump (perf_counter, paired per frame). The two parts are independent: the interval of the sum is made by resampling both. Refuses a busy GPU
(another process on it) unless --force-busy (then the result is labelled 'busy GPU: indicative').
usage: wsl bash tools/env/_py14.sh ../temp/e292/cost_ab.py [--device cuda|cpu] [--frames 300] [--rounds 20] [--reps 200] [--force-busy] [--out file.json]"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("evaluation", "training", "training/grid"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_readout as QR  # noqa: E402
import qmnms as QMN  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--frames", type=int, default=300)
ap.add_argument("--rounds", type=int, default=20)
ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--force-busy", action="store_true")
ap.add_argument("--out", default=None)
a = ap.parse_args()
busy = False
if a.device == "cuda":
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    busy = bool(q)
    assert not busy or a.force_busy, f"the GPU is not idle ({q.splitlines()[0]}...): wait for an idle GPU or give --force-busy (the result is then labelled indicative)"
rng = np.random.default_rng(0)

# ---- GPU part: the layer on the node grid, blocks A B | B A (A = layer call, B = the same loop without it), paired per round
dev = a.device
e = (torch.randn(1, 50, 50, device=dev) * 2.0)
cell = torch.stack(torch.meshgrid(-50 + 2 * (torch.arange(50).float() + 0.5), -50 + 2 * (torch.arange(50).float() + 0.5), indexing="ij"), 0)[None].to(dev)
pos = cell + (torch.rand(1, 2, 50, 50, device=dev) * 2 - 1)


def block(layer):
    if dev == "cuda":
        torch.cuda.synchronize()
        s, t = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
    else:
        t0 = time.perf_counter()
    for _ in range(a.reps):
        out = QMN.matrix_nms_logit(e, pos, 1.75, 0.1) if layer else e + 0.0
    if dev == "cuda":
        t.record()
        torch.cuda.synchronize()
        return s.elapsed_time(t) / a.reps
    return (time.perf_counter() - t0) * 1000.0 / a.reps


for _ in range(3):
    block(True)
    block(False)                                                                                  # priming, not counted
diff = []
for r in range(a.rounds):
    order = (True, False) if r % 2 == 0 else (False, True)
    res = {}
    for lay in order:
        res[lay] = block(lay)
    diff.append(res[True] - res[False])
diff = np.array(diff)
layer_ms = float(diff.mean())

# ---- host part: the hard step against plain on real frames of B's full dump, paired per frame
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", "predfull_e273_p_s0.pkl"), "rb"))[:a.frames]
hard, plain = QR.parse("close:1.5@0.5"), QR.parse("plain@0.5")
d_host = []
for d in dump:
    pred = dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"]))
    t = {}
    for nm, sp in (("hard", hard), ("plain", plain)):
        t0 = time.perf_counter()
        for _ in range(3):
            QR.apply(pred, sp)
        t[nm] = (time.perf_counter() - t0) * 1000.0 / 3
    d_host.append(t["hard"] - t["plain"])
d_host = np.array(d_host)

# ---- the difference a - c per frame = the layer's ms - the hard step's ms; the interval by resampling both parts
B = 4000
sm = np.array([rng.choice(diff, len(diff)).mean() - rng.choice(d_host, len(d_host)).mean() for _ in range(B)])
lo, hi = np.percentile(sm, [2.5, 97.5])
point = layer_ms - float(d_host.mean())
cost = "LOWER" if hi < 0 else ("EQUAL" if (lo <= 0 <= hi and hi <= 1.0) else "HIGHER")
out = dict(device=dev, busy_gpu=busy, rounds=a.rounds, reps=a.reps, frames=len(dump), layer_ms_per_frame=layer_ms, layer_round_diff_ms=[float(x) for x in diff], hard_step_ms_per_frame=float(d_host.mean()),
           a_minus_c_ms=point, interval95=[float(lo), float(hi)], cost=cost, label=("CPU smoke (not the deployed path)" if dev == "cpu" else "busy GPU: indicative" if busy else "idle GPU, batch 1, paired blocks A B | B A (layer), paired per frame (host step)"))
print(json.dumps({k: v for k, v in out.items() if k != "layer_round_diff_ms"}, indent=1))
if a.out:
    json.dump(out, open(a.out, "w"), indent=1)
