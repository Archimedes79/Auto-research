"""scratch: the evaluator's own loader (bev_probe.load_run -> bev_adapter.load_run) on the CPU for runs/t22_B_s0: is the student in the loaded model the checkpoint's EMA (fine-tuned), not the A2 file?"""
import os, sys
import torch
torch.set_num_threads(2)
os.chdir("/mnt/c/Auto-research/bev_fusion")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "evaluation")
from evaluation import bev_probe
cfg, model, loss, step, ckpath = bev_probe.load_run("runs/t22_B_s0", "cpu")
print("loaded step", step, "from", ckpath, "camera_student", cfg.model.get("camera_student") if isinstance(cfg.model, dict) else getattr(cfg.model, "camera_student", None))
live = {k: v for k, v in model.state_dict().items() if k.startswith("student.") or ".student." in k}
print("student tensors in the loaded model:", len(live), list(live)[:2])
ck = torch.load("runs/t22_B_s0/ckpt_00001500.pt", map_location="cpu", weights_only=False)
a2 = torch.load("runs/t22_A2_s0/ckpt_00006108.pt", map_location="cpu", weights_only=False)["ema"]
cke = {k: v for k, v in ck["ema"].items() if "student" in k}
def strip(k):
    return k.split("student.", 1)[1]
d_ck, d_a2, n = 0.0, 0.0, 0
for k, v in live.items():
    if not v.dtype.is_floating_point:
        continue
    s = strip(k)
    ek = [m for m in cke if m.endswith("student." + s)]
    if ek:
        d_ck = max(d_ck, float((v.float() - cke[ek[0]].float()).abs().max()))
    if s in a2:
        d_a2 = max(d_a2, float((v.float() - a2[s].float()).abs().max()))
    n += 1
print(f"floating student tensors compared: {n}; max abs diff loaded-vs-checkpoint EMA {d_ck:.3e} (should be 0); loaded-vs-A2 EMA {d_a2:.3e} (should be about 3e-3)")
