"""One-off: milliseconds per frame of the S1 camera chain on an IDLE GPU (plan 10 level 6; experiments/s1a_lift_pd_vs_p section 4).
Batch 1, CUDA-synchronised median of 300 calls after 50 warm-ups (evaluation/metrics.ms_per_frame), fp32 and fp16 autocast.
Blocks: the frozen image heads (ResNet-50 + neck + DepthNet, random weights: timing does not depend on them), the lift of every
preset, and the whole module from cached features (lift + reduce + trunk + head) for the presets ours / dualbev / simplebev /
lss, plus our design with the stride-1 detail path and with trunk width 1.0. The result is INVALID (exit 1) if another process
uses the GPU before or after (checked with nvidia-smi: compute apps and utilisation over 10 s) unless --force is given.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/timing/time_s1_chain.py [--force]
"""
import datetime
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, "training")
sys.path.insert(0, ".")
from camera_encoder import FrontCameraEncoder  # noqa: E402
from evaluation.metrics import ms_per_frame  # noqa: E402
from s1_camera import CameraBEVSeg, S1Config, _rig, n_params  # noqa: E402


def smi(query):
    return subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def gpu_idle(seconds=10):
    apps = [l for l in smi("compute-apps=pid,used_memory").splitlines() if l.strip()]
    util = []
    for _ in range(seconds):
        util.append(float(smi("gpu=utilization.gpu").split()[0]))
        time.sleep(1)
    return not apps, apps, max(util)


def main():
    force = "--force" in sys.argv
    ok0, apps0, util0 = gpu_idle()
    lines = [f"S1 chain timing {datetime.datetime.now():%Y-%m-%d %H:%M}, {torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
             f"cuDNN {torch.backends.cudnn.version()}, clocks {smi('gpu=clocks.sm,clocks.max.sm')}",
             f"before: compute apps {apps0 or 'none'}, max utilisation {util0:.0f} %"]
    if not (ok0 and util0 < 5) and not force:
        print("\n".join(lines + ["GPU NOT IDLE: no timing (use --force for an INVALID indicative run)"]))
        sys.exit(1)
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    K, T = _rig()
    K, T = K.to(dev)[None], T.to(dev)[None]
    img = torch.randn(1, 3, 256, 704, device=dev)
    cam_vec = torch.randn(1, 27, device=dev)
    prob = torch.rand(1, 118, 16, 44, device=dev).softmax(1)
    ctx = torch.randn(1, 80, 16, 44, device=dev)
    batch = {"prob": prob, "ctx": ctx, "K": K, "T_cam_car": T, "hw": (256, 704), "ida": None}
    rows = []

    def time_it(name, fn, half=True):
        for mode in (("fp32", False), ("fp16", True)) if half else (("fp32", False),):
            def call():
                with torch.autocast("cuda", dtype=torch.float16, enabled=mode[1]):
                    return fn()
            r = ms_per_frame(call, n=300, warmup=50, batch1=False)
            rows.append(f"{name:52s} {mode[0]}  median {r['median']:7.2f} ms  p90 {r['p90']:7.2f} ms")
            print(rows[-1], flush=True)
    with torch.no_grad():
        heads = FrontCameraEncoder().to(dev).eval()
        time_it("image heads (ResNet-50 + neck + DepthNet)", lambda: heads.image_features(img, cam_vec))
        for name, cfg in (("ours", S1Config.preset("ours")), ("dualbev", S1Config.preset("dualbev")),
                          ("simplebev", S1Config.preset("simplebev")), ("lss", S1Config.preset("lss")),
                          ("ours + detail path 32", S1Config.preset("ours", detail=32)),
                          ("ours, trunk width 1.0", S1Config.preset("ours", trunk_width=1.0))):
            m = CameraBEVSeg(cfg).to(dev).eval()
            if name in ("ours", "dualbev", "simplebev", "lss"):
                time_it(f"lift only, {name}", lambda: m.lifted(prob, ctx, K, T, (256, 704)))
            time_it(f"module from cached features, {name} ({n_params(m) / 1e6:.1f} M)", lambda: m(batch))
    ok1, apps1, util1 = gpu_idle(5)
    lines.append(f"after: compute apps {apps1 or 'none'}, max utilisation {util1:.0f} %")
    valid = ok0 and ok1 and util0 < 5 and util1 < 5
    lines.append("VALID (idle GPU before and after)" if valid else "INVALID (the GPU was not idle)")
    out = "\n".join(lines[:2] + rows + lines[2:])
    os.makedirs("temp/timing", exist_ok=True)
    path = f"temp/timing/result_{datetime.date.today()}.txt"
    open(path, "w").write(out + "\n")
    print(out)
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
