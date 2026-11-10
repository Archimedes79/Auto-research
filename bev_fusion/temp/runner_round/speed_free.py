"""Scratch (diagnostic, no verdict): ms per frame of the ground-truth-free run, by stage, against the live subject, and the cost of storing a frame in two formats.
    run through run.py --no-card on an idle box:  python3 temp/cleanup/speed_free.py runs/t20_base_s0 scene-0003 14 cuda"""
import io
import json
import os
import pickle
import sys
import time
import zlib

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from evaluation import bev_adapter as A  # noqa: E402
from evaluation import bev_free as F  # noqa: E402

RUN, SCENE, N, DEV = (sys.argv[1:] + [None] * 4)[:4]
RUN, SCENE, N, DEV = RUN or "runs/t20_base_s0", SCENE or "scene-0003", int(N or 14), DEV or ("cuda" if torch.cuda.is_available() else "cpu")
WARM = 2


def sync():
    if DEV.startswith("cuda"):
        torch.cuda.synchronize()


def med(x):
    x = x[WARM:]
    return round(float(np.median(x)), 1) if x else None


def ms(f):
    sync()
    t = time.perf_counter()
    r = f()
    sync()
    return r, (time.perf_counter() - t) * 1000.0


def timed(obj, name, acc):
    f = getattr(obj, name)

    def g(*a, **k):
        r, t = ms(lambda: f(*a, **k))
        acc.setdefault(name, []).append(t)
        return r
    setattr(obj, name, g)


t0 = time.time()
src, subj, _ = A.build(RUN, device=DEV, split="val", workers=0)
build_s = round(time.time() - t0, 1)
refs = src.frames(SCENE)[:N]
out = {"run": RUN, "scene": SCENE, "frames": len(refs), "device": DEV, "build_s": build_s, "load_avg": os.getloadavg(), "warmup_frames_skipped": WARM}

subj.reset()
live_load, live_pred = [], []
for ref in refs:
    s, a = ms(lambda: src.load(ref))
    _, b = ms(lambda: subj.predict(s, taps=()))
    live_load.append(a), live_pred.append(b)
out["live_ms"] = {"load_with_gt": med(live_load), "predict_with_loss": med(live_pred)}

fsrc, fsub = F.free_source(src), F.FreeSubject(subj)
stages = {}
timed(fsub, "_forward", stages), timed(fsub, "_outputs", stages)


def free_pass(taps):
    fsub.reset()
    stages.clear()
    load, pred, preds = [], [], []
    for ref in refs:
        s, a = ms(lambda: fsrc.load(ref))
        p, b = ms(lambda: fsub.predict(s, taps=taps))
        load.append(a), pred.append(b), preds.append(p)
    return load, pred, preds


load, pred, preds = free_pass(())
out["free_ms"] = {"load_no_gt": med(load), "predict_no_loss": med(pred), "forward": med(stages["_forward"]), "outputs": med(stages["_outputs"])}
load, pred, preds_t = free_pass(("fused",))
out["free_ms_with_fused_tap"] = {"predict": med(pred)}

# storing one frame: pickle + zlib level 1 (the first format of the evaluation suite) against .npz (uncompressed, compressed)
p = preds[-1]
flat = {}
for dt, o in p.outputs["dt"].items():
    for g in ("prob", "dist", "dir"):
        for n, a in (o.get(g) or {}).items():
            flat[f"{dt:+g}::{g}::{n}"] = a
    for n in ("mass", "vhat", "objvel"):
        if o.get(n) is not None:
            flat[f"{dt:+g}::{n}"] = o[n]
raw = sum(a.nbytes for a in flat.values())


def store(kind):
    t = time.perf_counter()
    if kind == "pickle+zlib1":
        data = zlib.compress(pickle.dumps({"outputs": p.outputs, "info": p.info}, protocol=4), 1)
    else:
        buf = io.BytesIO()
        (np.savez_compressed if kind == "npz compressed" else np.savez)(buf, **flat)
        data = buf.getvalue()
    return len(data), (time.perf_counter() - t) * 1000.0


out["store"] = {"raw_array_mb": round(raw / 1e6, 2), "n_arrays": len(flat)}
for kind in ("pickle+zlib1", "npz uncompressed", "npz compressed"):
    r = [store(kind) for _ in range(5)]
    out["store"][kind] = {"mb": round(r[0][0] / 1e6, 2), "ms": round(float(np.median([x[1] for x in r])), 1)}
out["fused_tap_mb_raw"] = round(preds_t[-1].taps["fused"].nbytes / 1e6, 2)
text = json.dumps(out, indent=1)
print(text)
with open(os.path.join(ROOT, "temp", "cleanup", "speed_free.txt"), "w") as f:
    f.write(text + "\n")
