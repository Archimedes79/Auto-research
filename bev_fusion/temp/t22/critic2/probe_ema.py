import copy, os, sys, tempfile
sys.path[:0] = ["training", "tests", "tests/support", ".", "../autoresearch"]
import torch
torch.set_num_threads(2)
import model as M, student_encoder as S, train as T
from feature_cache import CachedBEVFusionModel
from losses import BEVLoss
from test_slots import BASE
from trainkit import checkpoint as tk
from trainkit.optim import param_groups
tmp = tempfile.mkdtemp(prefix="critic2_")
DUAL = dict(BASE, camera_features="dual")
def build(seed=0, **kw):
    torch.manual_seed(seed); return CachedBEVFusionModel(M.ModelConfig(**dict(DUAL, **kw)))
st = S.StudentEncoder("resnet18", pretrained=False)
ck_a = os.path.join(tmp, "a", "ckpt_00000001.pt"); os.makedirs(os.path.dirname(ck_a))
torch.manual_seed(1); st_a = S.StudentEncoder("resnet18", pretrained=False)
torch.save({"state": {"model." + k: v for k, v in st_a.state_dict().items()}, "ema": {k: v for k, v in st_a.state_dict().items() if v.dtype.is_floating_point}, "step": 1, "cfg": {"backbone": "resnet18"}}, ck_a)
# 1 EMA drift of a frozen student
fro = build(camera_student="resnet18", student_init=ck_a)
ema = T.EMA(fro, 0.99)
for step in range(1500):
    ema.update(fro, step)
worst = max(float((ema.shadow[k] - fro.state_dict()[k]).abs().max()) for k in ema.shadow if k.startswith("student."))
rel = max(float(((ema.shadow[k] - fro.state_dict()[k]).abs() / fro.state_dict()[k].abs().clamp_min(1e-8)).max()) for k in ema.shadow if k.startswith("student.") and fro.state_dict()[k].numel() > 1)
nz = sum(int((ema.shadow[k] != fro.state_dict()[k]).sum()) for k in ema.shadow if k.startswith("student."))
print(f"1 frozen student EMA after 1500 updates: max abs drift {worst:.2e}, max rel {rel:.2e}, elements differing {nz}")
# 2 resume round trip of a trainable student
loss = BEVLoss(grid=fro.grid)
tr = build(camera_student="resnet18", student_init=ck_a, student_trainable=True)
cfg = T.TrainConfig(device="cpu", lr=1e-4, steps=10, lr_groups={"student": {"lr": 1e-5}})
opt = torch.optim.AdamW(param_groups(tr, loss, cfg.lr, cfg.weight_decay, cfg.lr_groups), lr=cfg.lr)
with torch.no_grad():
    for p in tr.student.parameters(): p.add_(0.01 * torch.randn_like(p))      # "trained" student, differs from student_init
e = T.EMA(tr, 0.99)
path = os.path.join(tmp, "r.pt")
T.save_checkpoint(path, tr, loss, opt, e, 5, cfg)
tr2 = build(seed=3, camera_student="resnet18", student_init=ck_a, student_trainable=True)   # a resume builds with student_init, then loads the checkpoint
loss2 = copy.deepcopy(loss)
opt2 = torch.optim.AdamW(param_groups(tr2, loss2, cfg.lr, cfg.weight_decay, cfg.lr_groups), lr=cfg.lr)
e2 = T.EMA(tr2, 0.99)
step, _ = T.load_checkpoint(path, tr2, loss2, opt2, e2)
same = all(torch.equal(tr.state_dict()[k], tr2.state_dict()[k]) for k in tr.state_dict())
print(f"2 trainable student resume: step {step}, every tensor (student included) equals the saved one: {same}; differs from student_init: {not torch.equal(tr2.student.state_dict()['head8.out.bias'], st_a.state_dict()['head8.out.bias'])}")
# 3 warm start of a frozen-student chain checkpoint into a model with another student_init
ck_b = os.path.join(tmp, "b", "ckpt_00000001.pt"); os.makedirs(os.path.dirname(ck_b))
torch.manual_seed(2); st_b = S.StudentEncoder("resnet18", pretrained=False)
torch.save({"state": {"model." + k: v for k, v in st_b.state_dict().items()}, "ema": {k: v for k, v in st_b.state_dict().items() if v.dtype.is_floating_point}, "step": 1, "cfg": {"backbone": "resnet18"}}, ck_b)
fa = build(camera_student="resnet18", student_init=ck_a); ea = T.EMA(fa, 0.99)
optf = torch.optim.AdamW(param_groups(fa, loss, 1e-4, 0.01), lr=1e-4)
pa = os.path.join(tmp, "fa.pt"); T.save_checkpoint(pa, fa, loss, optf, ea, 5, cfg)
ck = torch.load(pa, weights_only=False)
print("3 frozen chain checkpoint: student keys in state:", sum(k.startswith("model.student.") for k in ck["state"]), "in ema:", sum(k.startswith("student.") for k in ck["ema"]))
fb = build(camera_student="resnet18", student_init=ck_b)
fresh = tk.warm_start(pa, fb, copy.deepcopy(loss))
print("   warm start of that checkpoint into a model with student_init=B: student now equals A's: ", all(torch.equal(fb.student.state_dict()[k], fa.student.state_dict()[k]) for k in fa.student.state_dict() if fa.student.state_dict()[k].dtype.is_floating_point), "| fresh student keys:", sum(k.startswith("student.") for k in fresh))
