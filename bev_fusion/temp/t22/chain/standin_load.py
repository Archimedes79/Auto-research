"""scratch: the stand-in / twin evaluation path on the CPU with the REAL t21 pair4 checkpoint: load_run (the evaluation's own loader), then one synthetic forward.
    python temp/t22/chain/standin_load.py RUN_FOLDER CHAIN_RUN [STUDENT_CKPT]"""
import os, sys, time
import torch
torch.set_num_threads(2)
sys.path[:0] = [os.getcwd()]
from evaluation import bev_adapter as A
run, chain = sys.argv[1], sys.argv[2]
code = A.setup_paths(run)
sys.path.insert(0, "tests")
ns = A._chain()
t = time.time()
cfg, model, loss, step, ck = A.load_run(run, "cpu", chain, ns)
print("load_run", round(time.time() - t, 1), "s; step", step, os.path.basename(ck), "code", os.path.relpath(code), "student", type(model.student).__name__ if model.student is not None else None)
cs = torch.load(os.path.join(chain, os.path.basename(ck)), map_location="cpu", weights_only=False)["ema"]
sd = model.state_dict()
keys = [k for k in sd if not k.startswith("student.") and k in cs and sd[k].dtype.is_floating_point]
print("chain tensors equal to the checkpoint's EMA:", sum(torch.equal(sd[k], cs[k]) for k in keys), "of", len(keys))
if model.student is not None:
    st = torch.load(sys.argv[3], map_location="cpu", weights_only=False)["ema"]
    print("student tensors equal to", os.path.basename(sys.argv[3]), "EMA:", sum(torch.equal(sd["student." + k], v) for k, v in st.items()), "of", len(st),
          "| trainable", sum(p.numel() for p in model.student.parameters() if p.requires_grad), "| BN training", any(m.training for m in model.student.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)))
from test_slots import batch_for
b = batch_for(model.cfg, B=1, Fn=cfg.n_cam)
b["lane_img"] = torch.randint(0, 256, (1, 3, 256, 704), dtype=torch.uint8)
t = time.time()
with torch.no_grad():
    res = model(b, dts=(0.0,))
print("forward", round(time.time() - t, 1), "s; sem", tuple(res["out"][0.0]["sem"].shape), "finite", bool(torch.isfinite(res["out"][0.0]["sem"]).all()), "student_side in result", "student_side" in res)
