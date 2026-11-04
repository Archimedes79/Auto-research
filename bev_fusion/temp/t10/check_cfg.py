"""one-off: the t10 config builds the dual model through train.build_model (CPU)"""
import json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "training")]
os.chdir(ROOT)
import train
cfg = train.TrainConfig(**json.load(open("training/configs/t10_dual_s0.json")))
fn = [n for n in dir(train) if n.startswith("build_model") or n == "build"]
print("builders:", fn)
model, loss, cached = getattr(train, fn[0])(cfg)
print("cached", cached, "lift2", model.lift2.frustum_cells, "params", sum(p.numel() for p in model.parameters()) / 1e6)
