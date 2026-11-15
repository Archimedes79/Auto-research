import os, sys, torch
ROOT = "/mnt/c/Auto-research/bev_fusion"
sys.path.insert(0, os.path.join(ROOT, "tests")); sys.path.insert(0, os.path.join(ROOT, "training"))
from feature_cache import CachedBEVFusionModel
from model import ModelConfig
from test_slots import batch_for
torch.set_num_threads(2)
GRID = {"x_max": 16.0, "y_min": -8.0, "y_max": 8.0}
BASE = dict(pretrained="", trunk_pretrained="", lane_pretrained="", grid_kwargs=GRID, c_cam=16, c_rad=8, c_fuse=24, trunk_width=0.25, p_drop=0.0, camera_features="dual", lane_branch="res4")
img = torch.randint(0, 256, (2, 3, 256, 704), dtype=torch.uint8, generator=torch.Generator().manual_seed(1))

def build(**kw):
    torch.manual_seed(0)
    m = CachedBEVFusionModel(ModelConfig(**dict(BASE, **kw)))
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        m.lane.mix.weight.copy_(torch.randn(m.lane.mix.weight.shape, generator=g) * 0.05)
        m.lane.mix.bias.copy_(torch.randn(m.lane.mix.bias.shape, generator=g) * 0.05)
    return m

def lane_out(m, b, grad=False):
    o = []
    h = m.lane.register_forward_hook(lambda mod, a, out: o.append(out))
    try:
        with torch.set_grad_enabled(grad):
            m(b, dts=(0.0,))
    finally:
        h.remove()
    return o[0]

for os_ in (1, 4):
    pd_, pull = build(oversample=os_, lane_use_depth=True), build(oversample=os_, lane_use_depth=False)
    pull.load_state_dict(pd_.state_dict())
    for mode in ("eval", "train"):
        getattr(pd_, mode)(); getattr(pull, mode)()
        b = batch_for(pull.cfg); b["lane_img"] = img
        ones = dict(b, prob=torch.ones_like(b["prob"]))
        a = lane_out(pull, b); a2 = lane_out(pull, b)
        c = lane_out(pd_, ones)
        rnd = dict(b, prob=torch.rand_like(b["prob"]).softmax(2))
        d = lane_out(pull, rnd)
        print(f"os={os_} {mode}: pull repeat equal {torch.equal(a, a2)} | pull == PD(prob=1): max|diff| {float((a - c).abs().max()):.3g} (max|a| {float(a.abs().max()):.3g}) allclose(1e-6) {torch.allclose(a, c, atol=1e-6, rtol=1e-5)} | pull indep of prob {torch.equal(a, d)}")
    # gradient w.r.t. the depth probabilities, train mode
    for name, m in (("PD", pd_), ("P", pull)):
        m.train()
        b = batch_for(m.cfg); b["lane_img"] = img
        pr = b["prob"].clone().requires_grad_(True); b["prob"] = pr
        o = lane_out(m, b, grad=True)
        g, = torch.autograd.grad(o.sum(), pr, allow_unused=True)
        print(f"os={os_} grad of the lane output wrt batch prob, {name}: {'None' if g is None else float(g.abs().sum())}")

# student path
try:
    for flag in (True, False):
        m = build(oversample=1, lane_branch="rgb", camera_student="resnet18", student_trainable=True, lane_use_depth=flag).train()
        b = batch_for(m.cfg); b["lane_img"] = img
        o = lane_out(m, b, grad=True)
        ps = [p for p in m.student.parameters() if p.requires_grad]
        gs = torch.autograd.grad(o.sum(), ps, allow_unused=True)
        nz = sum(1 for g in gs if g is not None and float(g.abs().sum()) > 0)
        print(f"student path, lane_use_depth={flag}: lane output reaches {nz} of {len(ps)} trainable student tensors; |lane out| max {float(o.abs().max()):.3g}")
except Exception as e:
    import traceback; traceback.print_exc()
