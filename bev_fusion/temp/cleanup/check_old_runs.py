"""scratch (Image Encoder, clean-base round): can the evaluation still load the stored t22 runs? One run per process (the chain's modules are imported once per process):
bev_adapter.setup_paths(run) puts the run's pinned code/ first (or the live training/ when it has none), load_run builds the model from the stored config and loads the checkpoint (CPU);
a stand-in run keeps its weights in another run (--ckpt-dir). Then the Evaluation chat's camera controls are applied to a synthetic item / batch that carries lane_img_old (zero and shuffle).
usage: python temp/cleanup/check_old_runs.py RUN [CKPT_DIR]   (from bev_fusion/)"""
import os
import sys
import traceback

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402
from evaluation import bev_adapter as A  # noqa: E402
from evaluation import bev_modules as BM  # noqa: E402
from evaluation import bev_probe as BP  # noqa: E402

run = sys.argv[1]
ckpt_dir = sys.argv[2] if len(sys.argv) > 2 else None
code = A.setup_paths(run)
try:
    cfg, model, loss, step, ck = A.load_run(run, "cpu", ckpt_dir)
    print(f"LOADS  {run}: code {os.path.relpath(code)}, checkpoint {os.path.relpath(ck)} step {step}, student {type(getattr(model, 'student', None)).__name__}, student_frames {getattr(model.cfg, 'student_frames', '-')}")
except Exception as e:                                                             # noqa: BLE001
    print(f"FAILS  {run}: code {os.path.relpath(code)}: {type(e).__name__}: {str(e)[:300]}")
    traceback.print_exc(limit=1)
    sys.exit(1)

# the camera controls of the evaluation on an item / batch with the older frames' images (the t3 cam_off class of bug: a control that leaves a camera input in)
def item(v):
    return {"frames": [{k: torch.full((2,), float(v)) for k in A.CAMERA_FRAME_KEYS} | {"token": "t"} for _ in range(2)],
            "lane_img": torch.full((3, 2), v, dtype=torch.uint8), "lane_img_old": torch.full((1, 3, 2), v, dtype=torch.uint8)}


it, other = item(5), item(9)
A.perturb_camera(it, other, "zero_cam")
ok_zero = int(it["lane_img"].sum()) == 0 and int(it["lane_img_old"].sum()) == 0
it = item(5)
A.perturb_camera(it, other, "shuffle_cam")
ok_shuf = int(it["lane_img_old"][0, 0, 0]) == 9 and int(it["lane_img"][0, 0]) == 9
b = {"lane_img": torch.ones(2, 3, 2, dtype=torch.uint8), "lane_img_old": torch.ones(2, 1, 3, 2, dtype=torch.uint8), "prob": torch.ones(2, 2)}
z = BM.zero_cam(b)
ok_bm = int(z["lane_img"].sum()) == 0 and int(z["lane_img_old"].sum()) == 0
pb = {"tokens": ["a", "b"], "lane_img": b["lane_img"].clone(), "lane_img_old": b["lane_img_old"].clone(), "prob": b["prob"].clone()}
BP.perturb(pb, "zero_cam")
ok_bp = int(pb["lane_img"].sum()) == 0 and int(pb["lane_img_old"].sum()) == 0
print(f"CONTROLS for {run}: perturb_camera zero {ok_zero}, shuffle {ok_shuf}; bev_modules.zero_cam {ok_bm}; bev_probe.perturb zero_cam {ok_bp}")
sys.exit(0 if all((ok_zero, ok_shuf, ok_bm, ok_bp)) else 2)
