"""scratch (t20, L4 on the dual features): ms per frame at batch 1 and the VRAM of one training step (batch 3, clips of 3 keyframes, BPTT = the config's
micro-batch), for t16_full, t20 without the lane branch and t20, on a synthetic cached batch of the real shapes. Output: temp/lane_dual/measure.json.
Run (GPU idle!): wsl bash -c 'cd bev_fusion && PYTHONPATH=/mnt/c/Auto-research $HOME/miniconda3/envs/gcar/bin/python temp/lane_dual/measure.py [--rounds 3]'"""
import copy
import json
import math
import os
import statistics
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "training"), os.path.join(ROOT, "tests"), os.path.join(ROOT, "tests", "support"), ROOT, os.path.join(ROOT, ".."),
                os.path.join(ROOT, "..", "autoresearch")]
os.chdir(ROOT)
import synthetic_world as sw  # noqa: E402
import train as T  # noqa: E402
from evaluation import bev_timing as BT  # noqa: E402
from losses import synthetic_gt  # noqa: E402
from test_slots import batch_for  # noqa: E402

DEV = "cuda"
RESOLVED_T20 = "temp/compose/t20_check.preview.json"            # `trainsuite start ... --id t20.check --no-card` (no --go) writes it
ROUNDS = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 3


def variants():
    t16 = T.read_config("training/configs/t16_full.json")
    t20 = T.read_config(RESOLVED_T20)
    nolane = copy.deepcopy(t20)
    nolane.model = dict(nolane.model, lane_branch="")
    return {"t16": t16, "t20_nolane": nolane, "t20": t20}


def synth_batch(cfg, mcfg, B, grid):
    b = batch_for(mcfg, B=B, Fn=cfg.n_cam)
    b["lane_img"] = torch.randint(0, 256, (B, 3, *b["hw"]), dtype=torch.uint8)
    scene = sw.Scene([sw.Box(6.0, -1.75, 4.5, 1.8, 1.5, 0.0, 8.0, 0.0, "car", 10.0), sw.Box(10.0, 4.0, 4.5, 1.8, 1.5, 0.0, 0.0, 0.0, "car", 10.0),
                      sw.Box(13.0, -4.0, 8.0, 2.5, 3.0, 0.05, 0.0, 0.0, "truck", 20.0),
                      sw.Box(5.0, 5.5, 0.6, 0.6, 1.8, math.pi / 2, 0.0, 1.2, "pedestrian", -5.0)], lanes=[-5.25, -1.75, 1.75, 5.25], v=8.0, omega=0.05)
    side = {"prev": synthetic_gt(scene.step(-0.5), grid), "cur": synthetic_gt(scene, grid), "next": synthetic_gt(scene.step(0.5), grid)}
    b["gt"] = [side] * B
    b["gap_steps"] = [{"prev": [(-0.5, 8.0, 0.05)], "next": [(0.5, 8.0, 0.05)]}] * B
    b["tokens"] = [f"t{i}" for i in range(B)]
    return b


def mb(x):
    return round(x / 1e9, 3)


def inference(name, cfg):
    model, loss, cached = T.build(cfg)
    model = model.to(DEV).eval()
    mcfg = model.cfg
    batch = T.to_device(synth_batch(cfg, mcfg, 1, model.grid), DEV)
    batch.pop("gt"), batch.pop("gap_steps")
    dts = (-0.5, 0.0, 0.5)

    def forward():
        with torch.no_grad():
            model(batch, dts=dts)
    torch.cuda.reset_peak_memory_stats()
    for attempt in range(8):                                    # a critic may run a short GPU test: wait for an idle GPU, keep the last try if it never comes
        state = BT.settled_state(DEV)
        idle = BT.is_idle(state, DEV, check_memory=False)
        if idle:
            break
        print("   GPU not idle:", BT.why_busy(state), flush=True)
        time.sleep(20)
    res = BT.time_modules(model, forward, DEV, 30, 5)
    res["idle"] = idle and BT.is_idle(BT.settled_state(DEV), DEV, check_memory=False)
    res["peak_gb"] = mb(torch.cuda.max_memory_allocated())
    res["params_lane"] = sum(p.numel() for p in model.lane.parameters() if p.requires_grad) if model.lane is not None else 0
    del model, loss
    torch.cuda.empty_cache()
    return res


def training(name, cfg):
    model, loss, cached = T.build(cfg)
    model, loss = model.to(DEV).train(), loss.to(DEV)
    params = [p for p in model.parameters() if p.requires_grad] + list(loss.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)
    clip = [synth_batch(cfg, model.cfg, cfg.batch_size, model.grid) for _ in range(cfg.seq_len)]
    peaks, secs = [], []
    for i in range(4):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        acc, ok, _, _ = T.run_clip(model, loss, clip, 1.0, DEV, "off", cfg, backward=True)
        opt.step()
        torch.cuda.synchronize()
        secs.append(time.perf_counter() - t0)
        peaks.append(mb(torch.cuda.max_memory_allocated()))
        assert ok and math.isfinite(acc["total"]), acc
    out = {"peak_gb": peaks, "s_per_microbatch": [round(s, 3) for s in secs], "total": float(acc["total"]), "reserved_gb": mb(torch.cuda.max_memory_reserved()),
           "n_samples_per_microbatch": cfg.batch_size * cfg.seq_len}
    del model, loss, opt, clip
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    res = {"inference": {}, "training": {}}
    cfgs = variants()
    for r in range(ROUNDS):
        for name, cfg in cfgs.items():
            res["inference"].setdefault(name, []).append(inference(name, cfg))
            x = res["inference"][name][-1]
            print(f"round {r} {name:11s} batch 1, dts (-0.5, 0, +0.5): {x['total_ms']:.1f} ms, lane {x['per_frame_ms'].get('lane', 0):.1f} ms, lift {x['per_frame_ms'].get('lift', 0):.1f}, "
                  f"lift2 {x['per_frame_ms'].get('lift2', 0):.1f}, idle {x['idle']}, peak {x['peak_gb']} GB", flush=True)
    for name, cfg in cfgs.items():
        res["training"][name] = training(name, cfg)
        x = res["training"][name]
        print(f"train {name:11s} {x['n_samples_per_microbatch']} samples/micro-batch: peak allocated {x['peak_gb']} GB, reserved {x['reserved_gb']} GB, {x['s_per_microbatch']} s", flush=True)
    res["median_total_ms"] = {n: statistics.median(x["total_ms"] for x in v) for n, v in res["inference"].items()}
    res["median_lane_ms"] = {n: statistics.median(x["per_frame_ms"].get("lane", 0.0) for x in v) for n, v in res["inference"].items()}
    print(json.dumps({k: res[k] for k in ("median_total_ms", "median_lane_ms")}))
    with open("temp/lane_dual/measure.json", "w") as f:
        json.dump(res, f, indent=1)
