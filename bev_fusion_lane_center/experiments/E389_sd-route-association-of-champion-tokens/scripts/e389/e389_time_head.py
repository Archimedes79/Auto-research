"""E389 scratch (2026-11-19, D128; clause 6 of the card): the time of the RouteHead on the GPU, batch 1, K = 512 tokens, M = 66 (the VAL mean route length) and 136 (the VAL maximum) route keys, bf16 autocast and float32, random weights (the time does not depend on them).
CUDA events, 50 warm-up forwards, 300 timed forwards per setting (median, p10, p90 ms). The GPU is read before and after: the 'after' sample waits 2.5 s (a timing tool sampling at once reads its own kernels); QUIET = no other compute process, utilisation < 10 % before, load average < 2.
usage (project folder): python3 temp/e389/e389_time_head.py [out.json]      (run it only when nothing else uses the GPU / CPU: the card's clause 6 is UNVERIFIED otherwise)"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

R = os.environ.get("BFLC_ROOT", ".")
for sub in ("training/grid", "training/train", "training"):
    sys.path.insert(0, os.path.join(R, sub))
import jroute as JR  # noqa: E402


def gpu_state():
    o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().split(",")
    return dict(util=float(o[0]), mem_mb=float(o[1]))


def load1():
    return float(open("/proc/loadavg").read().split()[0])


assert torch.cuda.is_available(), "no GPU"
before, l0 = gpu_state(), load1()
dev = torch.device("cuda")
res = {}
torch.manual_seed(0)
for inputs in ("full",):
    for M in (66, 136):
        head = JR.RouteHead(inputs=inputs).to(dev).eval()
        batch = JR.synthetic_batch(1, 512, M, seed=1)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items() if k not in ("route",)}
        b["route"] = {k: v.to(dev) for k, v in batch["route"].items()}
        for mode in ("fp32", "bf16"):
            ts = []
            with torch.no_grad():
                for i in range(350):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    if mode == "bf16":
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            head(b)
                    else:
                        head(b)
                    e.record()
                    torch.cuda.synchronize()
                    if i >= 50:
                        ts.append(s.elapsed_time(e))
            ts = np.array(ts)
            res[f"{inputs}_M{M}_{mode}"] = dict(median_ms=float(np.median(ts)), p10_ms=float(np.percentile(ts, 10)), p90_ms=float(np.percentile(ts, 90)), n=len(ts))
time.sleep(2.5)
after, l1 = gpu_state(), load1()
quiet = before["util"] <= 5 and after["util"] <= 5 and max(l0, l1) < 3.0 and after["mem_mb"] - before["mem_mb"] < 2000      # the jtiming protocol of the lead (D128 follow-up): GPU idle <= 5 % before and after, load < 3
out = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), before=before, after=after, load1_before=l0, load1_after=l1, quiet=bool(quiet), params=sum(p.numel() for p in JR.RouteHead().parameters()), results=res,
           note="head alone, batch 1, K 512, random weights, CUDA events; clause 6 of the card: at most 1.0 ms on the champion path")
print(json.dumps(out, indent=1))
if len(sys.argv) > 1:
    open(os.path.join(R, sys.argv[1]), "x").write(json.dumps(out, indent=1))
