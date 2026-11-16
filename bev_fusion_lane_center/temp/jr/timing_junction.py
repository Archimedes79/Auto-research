"""scratch (2026-11-12): ms per frame of the junction thread's blocks on an IDLE GPU (batch 1, bf16 autocast like training, CUDA events after warm-up) and of the CPU read-out (B's read-out, connectors, snap) per frame.
usage: wsl bash tools/env/_py14.sh jr/timing_junction.py [run] [--n 60]   -> temp/jr/timing_junction.json   (run it only when no training job runs: check nvidia-smi)"""
import json, os, pickle, sys, time
import numpy as np
import torch
ROOT = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jends_eval as JE
import lane_gt_data as LD
run = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "e312_p_s0"
n = 60
if "--n" in sys.argv:
    n = int(sys.argv[sys.argv.index("--n") + 1])
dev = "cuda"
m, a = JE.load_model(run, dev)
LD.QData.DEFAULT_JUNCTION = True
ds = LD.QData("val_clean", map_mode="error", pixels=False)
xs = [LD.collate([ds[i]])["x"].to(dev).float() for i in np.linspace(0, len(ds) - 1, 12).astype(int)]
ev = lambda: torch.cuda.Event(enable_timing=True)
t = {"pyramid": [], "junction_maps": [], "pair_head": [], "total": []}
with torch.no_grad():
    for it in range(n + 10):
        x = xs[it % len(xs)]
        mk = [ev() for _ in range(4)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mk[0].record()
            P = m.jmodel.pyr(x)
            mk[1].record()
            jo = m.jmodel.jm(P)
            mk[2].record()
            # the pair head part of the full forward (keypoints + tokens + attention + pair logits): total minus the two stages above
            _, _, oe = m(x, gt=None)
            mk[3].record()
        torch.cuda.synchronize()
        if it >= 10:
            t["pyramid"].append(mk[0].elapsed_time(mk[1]))
            t["junction_maps"].append(mk[1].elapsed_time(mk[2]))
            t["total"].append(mk[2].elapsed_time(mk[3]))
res = {k: float(np.median(v)) for k, v in t.items() if v}
# the full forward includes pyramid + maps again; the pair head = full forward - (pyramid + maps) measured separately
res["full_forward_ms"] = res["total"]                                   # pyramid + maps + pair head in one call
res["pair_head_ms"] = max(res["total"] - res["pyramid"] - res["junction_maps"], 0.0)
res.pop("total")
res["note"] = "batch 1, bf16, RTX 3060, median of %d; the B head (nodes, links, sections, assignment) is timed in runs/e273/e273_p_s0/timing.json (24.2 ms total, pyramid 4.3 ms of it, shared)" % n
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(ROOT, "temp", "jr", "timing_junction.json"), "w"), indent=1)
