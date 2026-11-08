"""tensor / group counts of the t20 optimiser, EMA and clip, and the environment (torch, triton) for the compile question"""
import common as C
import torch

cfg, model, loss, batch, _ = C.make(B=1, n_boxes=3, grid_kwargs={"x_max": 24.0, "y_min": -12.0, "y_max": 12.0})
from trainkit import optim  # noqa: E402

groups = optim.param_groups(model, loss, cfg.lr, cfg.weight_decay, cfg.lr_groups)
n_t = sum(len(g["params"]) for g in groups)
print("param groups", len(groups), "param tensors", n_t, "elements %.1f M" % (sum(p.numel() for g in groups for p in g["params"]) / 1e6))
print("trainable tensors", sum(1 for p in list(model.parameters()) + list(loss.parameters()) if p.requires_grad), "state_dict entries (EMA)", len(model.state_dict()) + len(loss.state_dict()))
split = optim.clip_groups(model, cfg.grad_clip_split)
print("clip split groups", [len(s) for s in split])
print("torch", torch.__version__, "cuda build", torch.version.cuda)
try:
    import triton
    print("triton", triton.__version__)
except Exception as e:                     # noqa: BLE001
    print("triton: not importable:", type(e).__name__)
import shutil
print("gcc", shutil.which("gcc"), "nvcc", shutil.which("nvcc"))
