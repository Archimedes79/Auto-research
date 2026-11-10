"""t21 (D51): ms per frame of the camera temporal paths (synthetic cached features, random weights: timing only, no quality).

  window  n_cam 3   one call per keyframe: the batch holds the three frames, all lifted (today's t20 path)
  pair    n_cam 2   batch of two frames / stream: ONE frame per call, the previous raw BEV from state["buf"]
  rec     n_cam 3   batch of three frames (the carried buffer) / stream: ONE frame per call, the previous fused output from state["buf"]
  rec light         a stream frame that needs no output: lift + the temporal block only (no motion block, radar, trunk, heads)
for the dts (0,) and (-0.5, 0, 0.5); median of --n calls after 3 warm-up calls, batch 1, state carried.

  python evaluation/stream_timing.py --size tiny            # CPU smoke (tiny model, 32 x 32 grid)
  python evaluation/stream_timing.py --size full --device cuda      # the t20 network at full size; refuses a busy GPU (then CPU smoke)
Run from the bev_fusion folder; the GPU is used only when nvidia-smi shows no compute process and < 10 % utilisation (the desktop holds about
1.3 GB with no job). Result: a table on stdout and temp/stream/timing_<size>_<device>.json."""
import argparse
import dataclasses
import json
import os
import statistics
import subprocess
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.environ.get("STREAM_OUT") or os.path.join(ROOT, "temp", "stream")      # where the json of a run is written (scratch)
for sub in ("tests", "training"):
    sys.path.insert(0, os.path.join(ROOT, sub))
from feature_cache import CachedBEVFusionModel  # noqa: E402
from model import ModelConfig  # noqa: E402
from test_slots import batch_for  # noqa: E402

TINY = dict(pretrained="", trunk_pretrained="", lane_pretrained="", grid_kwargs={"x_max": 16.0, "y_min": -8.0, "y_max": 8.0}, oversample=1,
            c_cam=16, c_rad=8, c_fuse=24, trunk_width=0.25, p_drop=0.0)


def gpu_idle():
    q = lambda *a: subprocess.run(["nvidia-smi", *a], capture_output=True, text=True, timeout=20).stdout.strip()   # noqa: E731
    try:
        apps = [x for x in q("--query-compute-apps=pid", "--format=csv,noheader").splitlines() if x.strip()]
        util, mem = (float(x) for x in q("--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits").split(","))
    except Exception:                                              # noqa: BLE001
        return False, "nvidia-smi unavailable"
    return (not apps and util < 10 and mem < 2500), f"{len(apps)} compute process(es), {util:.0f} % utilisation, {mem:.0f} MiB"


def model_cfg(size, buf):
    if size == "tiny":
        return ModelConfig(**dict(TINY, cam_buffer=buf, camera_features="dual", motion_block=True, box_vel_from_field=True, object_moving=True))
    m = json.load(open(os.path.join(ROOT, "training", "configs", "t20_base.json")))["model"]
    return ModelConfig(**dict(m, cam_buffer=buf, pretrained="", trunk_pretrained="", lane_pretrained="", centreline_1m=True, radar_stationary=False))


def world(cfg, dev, n_frames):
    b = batch_for(cfg, B=1, Fn=n_frames)
    b["lane_img"] = torch.randint(0, 256, (1, 3, 256, 704), dtype=torch.uint8)
    seg = [(torch.full((1,), 1 / 12), torch.tensor([10.0]), torch.tensor([0.1]))] * 2        # two sweeps = one stride-2 frame
    return b, seg


def to(x, dev):
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, dict):
        return {k: to(v, dev) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to(v, dev) for v in x)
    return x


def sub(b, order, seg):
    """frames `order` of the batch (newest first, as the collate stores them), with the steps of consecutive two-sweep frames"""
    out = dict(b)
    for k in ("K", "T_cam_car", "prob", "ctx", "prob2", "ctx2"):
        if k in b:
            out[k] = b[k][:, order]
    out["steps"] = [seg * k for k in range(len(order))]
    return out


def timed(fn, n, dev):
    ts = []
    for i in range(n + 3):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        if i >= 3:
            ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=("tiny", "full"), default="tiny")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    dev = torch.device(a.device)
    if dev.type == "cuda":
        idle, why = gpu_idle()
        print(f"GPU: {why} -> {'idle' if idle else 'NOT IDLE: CPU smoke with the tiny model only'}")
        if not idle:
            dev, a.size = torch.device("cpu"), "tiny"
    torch.set_num_threads(2)
    rows = {}
    for name, buf, F, mode in (("window n_cam 3", "window", 3, "batch"), ("pair batch n_cam 2", "pair", 2, "batch"), ("pair stream", "pair", 1, "stream"),
                               ("rec batch n_cam 3", "recurrent", 3, "batch"), ("rec stream", "recurrent", 1, "stream"), ("rec stream light", "recurrent", 1, "light")):
        cfg = model_cfg(a.size, buf)
        torch.manual_seed(0)
        m = CachedBEVFusionModel(cfg).to(dev).eval()
        with torch.no_grad():
            for q in m.parameters():
                if q.requires_grad and q.abs().max() == 0:
                    q.normal_(0, 0.02)
        b, seg = world(cfg, dev, 4)
        b = to(sub(b, [0] if F == 1 else list(range(F - 1, -1, -1)), seg), dev)
        seg = to(seg, dev)
        for dts in ((0.0,), (-0.5, 0.0, 0.5)):
            st = {"s": None}
            # warm state: one cold call, then every timed call continues from the last
            step_batch = (torch.full((1,), 0.5, device=dev), torch.tensor([10.0], device=dev), torch.tensor([0.1], device=dev))
            step = seg if mode != "batch" else step_batch

            def call():
                with torch.no_grad():
                    if mode == "light":
                        st["s"] = m.advance(b, st["s"], step if st["s"] else None)
                    else:
                        r = m(b, dts=dts, state=st["s"], step=step if st["s"] else None)
                        st["s"] = r["state"]
            rows.setdefault(name, {})[str(dts)] = round(timed(call, a.n, dev), 2) if mode != "light" or dts == (0.0,) else None
        print(f"{name:20s} dts (0,): {rows[name][str((0.0,))]} ms   dts (-0.5, 0, 0.5): {rows[name][str((-0.5, 0.0, 0.5))]} ms", flush=True)
        del m
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"timing_{a.size}_{dev.type}.json")
    json.dump({"size": a.size, "device": str(dev), "n": a.n, "ms_per_call_median": rows, "note": "synthetic cached features, random weights, batch 1; window / batch = one call per keyframe"},
              open(out, "w"), indent=1)
    print("written", out)


if __name__ == "__main__":
    main()
