"""E368 (S2) inference cost, measured (scratch, GPU, the lead's request 2026-11-18: 'S2 inference cost: r-head vs z-head, ms per frame, parameters; the card predicted +1 - 2 ms, not measured'). The protocol of `jtoken_eval.py timing` / `jmap_eval.py time`: batch 1, bf16 autocast, CUDA events, no_grad, the head alone on cached VAL frames in STREAM order (the past of frame t = the head's own states of frame t - 1, moved by the stored pose: `jpast.make_past`, exactly what the stream read-out does,
without its kill-measure hooks (`UseMeter`: they would add a second block call)); the first `--frames` frames from the start of the VAL stream, `--repeats` passes after one untimed pass; the first frame of a scene runs cold (no past), reported by itself. Arms: `e368_r_s<S>` (the head WITH the PastBlock) and `e368_z_s<S>` (no block).
usage: python time_s2.py --runs e368_r_s0,e368_z_s0 [--frames 24] [--repeats 4] [--cpu]   -> prints a json, writes runs/<first run>/past_timing_<UTC>.json"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

R = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training/data", "training/grid", "training/train"):
    sys.path.insert(0, os.path.join(R, sub))
import jpast as JP  # noqa: E402
import jpast_train as JPT  # noqa: E402
import jrecur_time as JRT  # noqa: E402
import jtiming as JTG  # noqa: E402
import jtoken as JK  # noqa: E402
import jtoken_eval as JE  # noqa: E402
import jtoken_train as JKT  # noqa: E402
import lane_past_data as LPD  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--frames", type=int, default=24)
ap.add_argument("--repeats", type=int, default=4)
ap.add_argument("--cpu", action="store_true")
a = ap.parse_args()
ok, why = (False, "--cpu") if a.cpu else JE.gpu_free()
if not a.cpu and not ok:
    sys.exit(f"STOP: no GPU timing ({why})")
dev = "cpu" if a.cpu else "cuda"
cuda = dev == "cuda"
res = dict(device=dev, frames=a.frames, repeats=a.repeats, gpu_rule=why, arms={})
load0, gpu0 = JTG.load1(), (None if a.cpu else JTG.gpu_state())
for run in a.runs.split(","):
    m, args = JPT.load_model(run, dev, JE.RUNS)
    m.eval()
    kw = JE.cache_kw(args)
    ds, dl = LPD.loader("val", kw["seed"], 1, 0, ident=kw["ident"], K=kw["K"], shuffle=False, stream=True, **({"cache": kw["cache"]} if kw.get("cache") else {}))
    batches = []
    for b in dl:
        batches.append(b)
        if len(batches) == a.frames:
            break
    block = getattr(m, "past_block", None)
    frames_ms, cold_ms = [], []
    for rep in range(a.repeats + 1):
        out_prev = fr_prev = None
        row_warm, row_cold = [], []
        for b in batches:
            hp = bool(b["has_prev"][0, 0])
            frd = JKT.to_device(b["frames"][0], dev)
            R_, t_ = b["rel_R"][:, 0].to(dev), b["rel_t"][:, 0].to(dev)
            past = {}
            if block is not None and hp and out_prev is not None:
                past = JP.make_past(out_prev, fr_prev, frd, R_, t_, m._step_py >= JK.SWITCH)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=cuda):
                out, ms = JRT.timed(lambda: (m(frd, past) if (block is not None and past) else m(frd)), cuda)
            (row_warm if (past or block is None) else row_cold).append(ms)
            out_prev, fr_prev = out, frd
        if rep:
            frames_ms += row_warm
            cold_ms += row_cold
    res["arms"][run] = dict(params=JK.count_params(m), params_block=JK.count_params(block) if block is not None else 0, has_block=block is not None, n_timed=len(frames_ms), median_ms=float(np.median(frames_ms)), p90_ms=float(np.percentile(frames_ms, 90)),
                            cold_frames=len(cold_ms), cold_median_ms=float(np.median(cold_ms)) if cold_ms else None)
if not a.cpu:
    time.sleep(2.5)                                                                      # the aging wait of jtiming's own protocol (nvidia-smi averages: an at-once sample reads this process's own kernels)
load1, gpu1 = JTG.load1(), (None if a.cpu else JTG.gpu_state())
quiet, line = JTG.quiet_verdict(load0, load1, dict(util_max=0.0) if a.cpu else gpu0, dict(util_max=0.0) if a.cpu else gpu1)
res.update(host=line, quiet=bool(quiet), usable_for_gcost=bool(quiet and cuda))
arms = list(res["arms"].values())
if len(arms) == 2 and arms[0]["has_block"] and not arms[1]["has_block"]:
    res["block_cost_ms"] = arms[0]["median_ms"] - arms[1]["median_ms"]
print(json.dumps(res, indent=1))
first = a.runs.split(",")[0]
print("written", os.path.relpath(JE._write_stamped(os.path.join(JE.RUNS, first), "past_timing" + ("_cpu" if a.cpu else ""), res), R))
