"""scratch (t22 stage 0): milliseconds per frame of the image side of the dual chain against the students, on an IDLE GPU.

(a) DualBEV heads (ResNet-50 + neck + DepthNet, stride 16), (b) BEVFusion image side (Swin-T + FPN + depth conv, stride 8), (c) both, (d) the frozen ResNet-50 layer 1 of the `res4`
lane branch, (e) the EfficientViT-B1 student (both heads, the H8 warp, the softmax), (f) the ResNet-18 student, (g) the student plus (d) = what the stage-B path needs for the
current frame. Batch 1, 256 x 704, CUDA-synchronised median of 300 calls after 50 warm-ups (evaluation/metrics.ms_per_frame), fp32 and fp16 autocast, random weights (timing
does not depend on them). INVALID (exit 1) if another process uses the GPU before or after (compute apps, utilisation over 10 s) unless --force.
Run from the bev_fusion folder, only in a quiet slot: ~/miniconda3/envs/gcar/bin/python temp/t22/time_student.py [--force]
Built on temp/timing/time_image_sides.py (one-off scratch of the earlier timing); result file temp/t22/result_time_student_<date>.txt.
"""
import datetime
import subprocess
import sys
import time

import torch

sys.path.insert(0, "training")
sys.path.insert(0, ".")
from bevfusion_camera import BEVFusionImageSide  # noqa: E402
from camera_encoder import FrontCameraEncoder  # noqa: E402
from evaluation.metrics import ms_per_frame  # noqa: E402
from lane_branch import Res4Features  # noqa: E402
from student_encoder import StudentEncoder  # noqa: E402


def smi(query):
    return subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def gpu_idle(seconds=10):
    apps = [line for line in smi("compute-apps=pid,used_memory").splitlines() if line.strip()]
    util = []
    for _ in range(seconds):
        util.append(float(smi("gpu=utilization.gpu").split()[0]))
        time.sleep(1)
    return not apps, apps, max(util)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    force = "--force" in sys.argv
    ok0, apps0, util0 = gpu_idle()
    lines = [f"student timing {datetime.datetime.now():%Y-%m-%d %H:%M}, {torch.cuda.get_device_name(0)}, torch {torch.__version__}, clocks {smi('gpu=clocks.sm,clocks.max.sm')}",
             f"before: compute apps {apps0 or 'none'}, max utilisation {util0:.0f} %"]
    if not (ok0 and util0 < 5) and not force:
        print("\n".join(lines + ["GPU NOT IDLE: no timing (use --force for an INVALID indicative run)"]))
        sys.exit(1)
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    img = torch.randn(1, 3, 256, 704, device=dev)
    img255 = (img * 60 + 120).clamp(0, 255)
    cam_vec = torch.randn(1, 27, device=dev)
    rows = []

    def time_it(name, fn):
        for mode in (("fp32", False), ("fp16", True)):
            def call():
                with torch.autocast("cuda", dtype=torch.float16, enabled=mode[1]):
                    return fn()
            r = ms_per_frame(call, n=300, warmup=50, batch1=False)
            rows.append(f"{name:62s} {mode[0]}  median {r['median']:7.2f} ms  p90 {r['p90']:7.2f} ms")
            print(rows[-1], flush=True)

    with torch.no_grad():
        dual = FrontCameraEncoder().to(dev).eval()
        bevf = BEVFusionImageSide().to(dev).eval()
        res4 = Res4Features(path=None).to(dev).eval()
        evit = StudentEncoder("efficientvit_b1", pretrained=False).to(dev).eval()
        r18 = StudentEncoder("resnet18", pretrained=False).to(dev).eval()
        time_it(f"(a) DualBEV heads, stride 16 ({n_params(dual) / 1e6:.1f} M)", lambda: dual.image_features(img, cam_vec))
        time_it(f"(b) BEVFusion image side, stride 8 ({n_params(bevf) / 1e6:.1f} M)", lambda: bevf.image_features(img))
        time_it("(c) both one after the other (the dual chain, per camera frame)", lambda: (dual.image_features(img, cam_vec), bevf.image_features(img)))
        time_it(f"(d) res4 layer 1, frozen ResNet-50 stem + layer 1 ({n_params(res4) / 1e6:.2f} M)", lambda: res4(img))
        time_it(f"(e) student EfficientViT-B1, H16 + H8 + warp + softmax ({n_params(evit) / 1e6:.1f} M)", lambda: evit(img255))
        time_it(f"(f) student ResNet-18, H16 + H8 + warp + softmax ({n_params(r18) / 1e6:.1f} M)", lambda: r18(img255))
        time_it("(g) EfficientViT-B1 student + res4 layer 1 (stage-B current frame)", lambda: (evit(img255), res4(img)))
        time_it("(h) the teachers (c) + res4 (d): what (g) replaces", lambda: (dual.image_features(img, cam_vec), bevf.image_features(img), res4(img)))
    torch.cuda.synchronize()
    time.sleep(4)                       # nvidia-smi's utilisation averages the last second or more: sampling right after the last kernel would count this script's own tail as a foreign load (runs 1 and 2 of 2026-11-05: 59 % and 65 %, no compute apps)
    ok1, apps1, util1 = gpu_idle(5)
    lines.append(f"after: compute apps {apps1 or 'none'}, max utilisation {util1:.0f} %")
    valid = ok0 and ok1 and util0 < 5 and util1 < 5
    lines.append("VALID (idle GPU before and after)" if valid else "INVALID (the GPU was not idle)")
    out = "\n".join(lines[:2] + rows + lines[2:])
    path = f"temp/t22/result_time_student_{datetime.date.today()}.txt"
    open(path, "w").write(out + "\n")
    print(out)
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
