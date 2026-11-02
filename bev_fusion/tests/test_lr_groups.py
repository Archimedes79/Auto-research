"""TrainConfig.lr_groups: a module's own learning rate and warm-up (training/train.py: lr_group_params, param_groups, scheduled_lr, check_resume; experiments/t17_motion_lr section 3);
CPU, seconds, toy network, and the REAL train() on a toy.

  1. without lr_groups (None or {}) param_groups is exactly the old function: two groups, the same parameters in the same order, the same weight decay
  2. with lr_groups: the rest in today's two groups (a module that only STARTS like the prefix, `motion_blockx`, and the loss parameters stay there), then a decay / no-decay pair per prefix,
     tagged base_lr / warmup / lr_group; groups are never empty
  3. the schedule: an untagged group gets cfg.lr * lr_factor exactly as before; a tagged one base_lr * the factor of its own warm-up (lr 1e-4, warm-up 1000: 3.34e-5 at step 0, 1e-4 at 999 and 1000);
     only lr or only warmup given: the other is the chain's
  4. errors: a prefix that names nothing, overlapping prefixes, a list instead of a dict, an unknown key, lr <= 0, a negative / float / bool warmup
  5. train(): a typo prefix fails at the start, before anything is trained; the optimiser has the groups, the rows carry lr.<prefix>; a group with a tiny lr hardly moves its weights; a resume with
     other lr_groups (or lr_groups on a checkpoint from before them) is refused, with the same ones it continues
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_lr_groups.py
"""
import math
import sys
import tempfile

import torch
from torch import nn

sys.path.insert(0, "training")
sys.path.insert(0, "tests")
import toy_train as TT  # noqa: E402
import train as T  # noqa: E402

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.trunk = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        self.motion_block = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        self.motion_blockx = nn.Linear(2, 2)                       # starts like the prefix, is not part of it
        self.head = nn.Linear(2, 2)


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(2))


def old_param_groups(model, loss, lr, wd):
    """the function before lr_groups; the reference"""
    decay, no_decay = [], []
    for name, p in list(model.named_parameters()) + [("loss." + n, p) for n, p in loss.named_parameters()]:
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def ids(g):
    return [id(p) for p in g["params"]]


m, l = Net(), Loss()
ref = old_param_groups(m, l, 3e-4, 0.01)
for label, kw in (("not given", {}), ("None", {"lr_groups": None}), ("{}", {"lr_groups": {}})):
    got = T.param_groups(m, l, 3e-4, 0.01, **kw)
    check(len(got) == 2 and all(ids(a) == ids(b) and a["weight_decay"] == b["weight_decay"] and set(a) == {"params", "weight_decay"} for a, b in zip(got, ref)),
          f"lr_groups {label}: today's two groups, the same parameters in the same order, no extra keys")

# 2 the layout with a prefix
spec = {"motion_block": {"lr": 1e-4, "warmup": 1000}}
g4 = T.param_groups(m, l, 3e-4, 0.01, spec)
mb = [p for n, p in m.named_parameters() if n.startswith("motion_block.")]
mb_ids = {id(p) for p in mb}
check(len(g4) == 4 and not (set(ids(g4[0])) | set(ids(g4[1]))) & mb_ids, "four groups; the first two are the rest and hold none of the prefix")
check(ids(g4[0]) == [i for i in ids(ref[0]) if i not in mb_ids] and ids(g4[1]) == [i for i in ids(ref[1]) if i not in mb_ids], "the rest keeps today's order (decay, no decay)")
check(id(m.motion_blockx.weight) in ids(g4[0]) and id(l.s) in ids(g4[1]), "`motion_blockx` (starts like the prefix) and the loss parameters stay in the rest")
check(set(ids(g4[2]) + ids(g4[3])) == mb_ids and all(p.ndim >= 2 for p in g4[2]["params"]) and all(p.ndim < 2 for p in g4[3]["params"]) and g4[2]["weight_decay"] == 0.01 and g4[3]["weight_decay"] == 0.0,
      "the prefix's parameters in their own decay / no-decay pair")
check(all(g["base_lr"] == 1e-4 and g["warmup"] == 1000 and g["lr_group"] == "motion_block" for g in g4[2:]) and all("base_lr" not in g for g in g4[:2]), "the prefix groups are tagged, the rest is not")
g6 = T.param_groups(m, l, 3e-4, 0.01, {"motion_block": {"lr": 1e-4}, "head": {"warmup": 50}})
check(len(g6) == 6 and [g.get("lr_group") for g in g6] == [None, None, "head", "head", "motion_block", "motion_block"] and g6[2]["base_lr"] == 3e-4 and "warmup" not in g6[4] and g6[2]["warmup"] == 50 and g6[4]["base_lr"] == 1e-4,
      "two prefixes: six groups in SORTED prefix order whatever the order of the keys; a missing lr is the chain's, a missing warmup stays untagged")
g6r = T.param_groups(m, l, 3e-4, 0.01, {"head": {"warmup": 50}, "motion_block": {"lr": 1e-4}})
check([ids(a) for a in g6] == [ids(b) for b in g6r], "reversing the order of the keys changes nothing: the same groups with the same parameters (a resume pairs the optimiser state by position)")
frozen = Net()
for p in frozen.motion_block.parameters():
    p.requires_grad_(False)
try:
    T.param_groups(frozen, l, 3e-4, 0.01, spec)
    frozen_err = False
except ValueError:
    frozen_err = True
check(frozen_err, "a prefix whose parameters are all frozen names no trainable parameter: an error")

# 3 the schedule
cfg = T.TrainConfig(lr=3e-4, warmup=300, steps=6000, lr_min_frac=0.01)
untagged, tagged = {"lr": 0.0}, {"lr": 0.0, **{k: v for k, v in g4[2].items() if k in ("base_lr", "warmup")}}
ok_chain = ok_mb = True
for step in (0, 150, 299, 300, 999, 1000, 3000, 5999):
    f = T.lr_factor(step, cfg.warmup, cfg.steps, cfg.lr_min_frac)
    ok_chain &= T.scheduled_lr(untagged, step, cfg, f) == cfg.lr * f
    hand = 1e-4 * (1 / 3 + 2 / 3 * (step + 1) / 1000) if step < 1000 else 1e-4 * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - 1000) / 5000)))
    ok_mb &= abs(T.scheduled_lr(tagged, step, cfg, f) - hand) < 1e-15
check(ok_chain, "an untagged group: cfg.lr * lr_factor, exactly the old single schedule (8 steps)")
check(ok_mb, "a tagged group: lr 1e-4 with its own warm-up 1000, against the formula by hand (8 steps)")
f0 = T.lr_factor(0, 300, 6000, 0.01)
check(abs(T.scheduled_lr(tagged, 0, cfg, f0) - 3.3400e-5) < 1e-9 and abs(T.scheduled_lr(tagged, 999, cfg, T.lr_factor(999, 300, 6000, 0.01)) - 1e-4) < 1e-12
      and abs(T.scheduled_lr(tagged, 1000, cfg, T.lr_factor(1000, 300, 6000, 0.01)) - 1e-4) < 1e-12, "the card's numbers: 3.34e-5 at step 0, 1e-4 at steps 999 and 1000")
only_lr = {"base_lr": 1e-4}
only_wu = {"base_lr": 3e-4, "warmup": 50}
check(abs(T.scheduled_lr(only_lr, 10, cfg, T.lr_factor(10, 300, 6000, 0.01)) - 1e-4 * T.lr_factor(10, 300, 6000, 0.01)) < 1e-18
      and abs(T.scheduled_lr(only_wu, 10, cfg, T.lr_factor(10, 300, 6000, 0.01)) - 3e-4 * T.lr_factor(10, 50, 6000, 0.01)) < 1e-18, "only lr: the chain's warm-up; only warmup: the chain's lr")

# 4 errors
bad = {"a prefix that names nothing": {"motoin_block": {"lr": 1e-4}}, "overlapping prefixes": {"motion_block": {"lr": 1e-4}, "motion_block.0": {"lr": 2e-4}},
       "a list": ["motion_block"], "an unknown key": {"motion_block": {"lr": 1e-4, "wd": 0.1}}, "an empty spec": {"motion_block": {}}, "lr 0": {"motion_block": {"lr": 0}},
       "negative lr": {"motion_block": {"lr": -1e-4}}, "a bool lr": {"motion_block": {"lr": True}}, "negative warmup": {"motion_block": {"warmup": -1}},
       "float warmup": {"motion_block": {"warmup": 10.5}}, "a string spec": {"motion_block": "1e-4"}}
errs = []
for label, spec_ in bad.items():
    try:
        T.lr_group_params(m, spec_)
        errs.append(label)
    except ValueError:
        pass
check(not errs, f"every bad lr_groups is a ValueError (not caught: {errs})")
check(len(T.lr_group_params(m, {"motion_block": {"warmup": 0}})["motion_block"]) == len(mb), "warmup 0 is allowed")

# 5 the real train() on a toy
root = tempfile.mkdtemp(prefix="lrg_")
base = TT.run_train(T, root, "base")
check(base["exc"] is None and len(base["ck"]["opt"]["param_groups"]) == 2 and not any(k.startswith("lr.") for r in base["rows"] for k in r), "train() without lr_groups: two optimiser groups, no lr.* column")
spec5 = {"net.3": {"lr": 2e-3, "warmup": 6}}
run = TT.run_train(T, root, "own", lr_groups=spec5)
pg = run["ck"]["opt"]["param_groups"]
check(run["exc"] is None and len(pg) == 4 and [g.get("lr_group") for g in pg] == [None, None, "net.3", "net.3"], "train() with lr_groups: four optimiser groups, the last two for net.3")
cfg5 = T.TrainConfig(lr=1e-2, warmup=2, steps=12, lr_min_frac=T.TrainConfig().lr_min_frac)
want = 2e-3 * T.lr_factor(0, 6, 12, cfg5.lr_min_frac)
first = run["rows"][0]
check(abs(first["lr.net.3"] - want) < 1e-12 and abs(first["lr"] - 1e-2 * T.lr_factor(0, 2, 12, cfg5.lr_min_frac)) < 1e-12, f"the rows: the chain's lr and lr.net.3 = {first['lr.net.3']:.3e} (by hand {want:.3e})")
init = TT.ToyNet()
tiny = TT.run_train(T, root, "tiny", lr_groups={"net.3": {"lr": 1e-9, "warmup": 0}})
w_init, w_base, w_tiny = init.net[3].weight.detach(), base["ck"]["state"]["model.net.3.weight"], tiny["ck"]["state"]["model.net.3.weight"]
moved_base, moved_tiny = (w_base - w_init).abs().max().item(), (w_tiny - w_init).abs().max().item()
check(moved_base > 1e-2 and moved_tiny < 1e-6, f"a group with lr 1e-9 hardly moves its weights ({moved_tiny:.1e}) while the chain's lr moves them ({moved_base:.1e})")
def _boom(*a, **k):
    raise RuntimeError("the data loaders were built")


typo = TT.run_train(T, root, "typo", loader=_boom, lr_groups={"net.33": {"lr": 1e-3}})                    # if the check came after the loaders, this would be the error
check(typo["exc"] and typo["exc"].startswith("ValueError") and "names no trainable parameter" in typo["exc"] and not typo["rows"] and typo["ck"] is None, "a typo prefix: a ValueError at the start, nothing trained")
other = TT.run_train(T, root, "own", lr_groups={"net.3": {"lr": 5e-3, "warmup": 6}})
check(other["exc"] and other["exc"].startswith("SystemExit") and "lr_groups" in other["exc"], "a resume with other lr_groups is refused")
again = TT.run_train(T, root, "own", lr_groups=spec5)
check(again["exc"] is None and "resumed from" in again["out"], "a resume with the same lr_groups continues")
late = TT.run_train(T, root, "base", lr_groups=spec5)
check(late["exc"] and late["exc"].startswith("SystemExit") and "lr_groups" in late["exc"], "lr_groups on a checkpoint of this code without them is refused (the changed value)")
import glob  # noqa: E402
import os  # noqa: E402
for path in glob.glob(os.path.join(root, "base", "ckpt_*.pt")):               # a checkpoint written BEFORE the field existed: its config has no `lr_groups` key at all
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck["cfg"].pop("lr_groups")
    torch.save(ck, path)
old_style = TT.run_train(T, root, "base", lr_groups=spec5)
check(old_style["exc"] and old_style["exc"].startswith("SystemExit") and "lr_groups" in old_style["exc"], "lr_groups on a checkpoint from before the field (no such key in its config) is refused")
old_plain = TT.run_train(T, root, "base")
check(old_plain["exc"] is None and "resumed from" in old_plain["out"], "a checkpoint from before the field resumes with no lr_groups, as before")

# lr_groups together with amp "fp16" (GradScaler): the scale is taken off before the optimiser sees the gradients, so a power-of-two scale changes nothing: the same weights, EMA and optimiser
# state as amp off, bit for bit, also through an overflow (here an overflow is not injected: see test_grad_scaler.py; the group tags and the per-group lr are what is new)
off_g = TT.run_train(T, root, "g_off", lr_groups=spec5)
fp_g = TT.run_train(T, root, "g_fp16", lr_groups=spec5, amp="fp16")
check(off_g["exc"] is None and fp_g["exc"] is None and TT.same_tree(off_g["ck"]["state"], fp_g["ck"]["state"]) and TT.same_tree(off_g["ck"]["ema"], fp_g["ck"]["ema"])
      and TT.same_tree(off_g["ck"]["opt"]["state"], fp_g["ck"]["opt"]["state"]) and [g["lr"] for g in off_g["ck"]["opt"]["param_groups"]] == [g["lr"] for g in fp_g["ck"]["opt"]["param_groups"]],
      "lr_groups with amp fp16 (scale 1024) = lr_groups with amp off: the same weights, EMA, optimiser state and group learning rates, bit for bit")
check(all(r["loss_scale"] == 1024.0 for r in fp_g["rows"]) and all(f"lr.net.3" in r for r in fp_g["rows"]), "and the rows carry both the loss scale and lr.net.3")

# evaluation/bev_adapter.load_run builds the optimiser "as train()" to load a checkpoint: with lr_groups it must pass them (the optimiser state has a slot per group), without them an older
# snapshot's param_groups (four arguments) must still work
sys.path.insert(0, "evaluation")
from types import SimpleNamespace  # noqa: E402

import bev_adapter as A  # noqa: E402
loaded = {}
for name in ("own", "tiny"):
    saved_build = T.build
    T.build = lambda cfg: (TT.ToyNet(), TT.ToyLoss(), True)
    try:
        cfg_l, model_l, loss_l, step_l, path_l = A.load_run(os.path.join(root, name), "cpu", ns=SimpleNamespace(T=T))
        loaded[name] = (dict(cfg_l.lr_groups), step_l)
    except Exception as e:  # noqa: BLE001
        loaded[name] = f"{type(e).__name__}: {str(e)[:100]}"
    finally:
        T.build = saved_build
check(loaded.get("own") == (spec5, 12) and loaded.get("tiny") == ({"net.3": {"lr": 1e-9, "warmup": 0}}, 12), f"bev_adapter.load_run loads a checkpoint of a run with lr_groups ({loaded})")
saved_build = T.build
T.build = lambda cfg: (TT.ToyNet(), TT.ToyLoss(), True)
old_pg = T.param_groups
T.param_groups = lambda model, loss, lr, wd: old_pg(model, loss, lr, wd)                          # an older snapshot: four arguments only
try:
    A.load_run(os.path.join(root, "base"), "cpu", ns=SimpleNamespace(T=T))
    plain_ok = True
except Exception as e:  # noqa: BLE001
    plain_ok = f"{type(e).__name__}: {e}"
finally:
    T.param_groups = old_pg
    T.build = saved_build
check(plain_ok is True, f"load_run of a run without lr_groups still calls param_groups with four arguments ({plain_ok})")

# BF_RESUME_ANY=1 lets a changed lr_groups pass the config check: the NEW values must then be the ones used (the optimiser state restores the saved tags)
calls = {"n": 0}
keep_stop = T._stop_requested
T._stop_requested = lambda d: "test" if (calls.__setitem__("n", calls["n"] + 1) or calls["n"]) > 8 else None          # stop at the 9th step: a checkpoint at step 8
first_half = TT.run_train(T, root, "force", steps=16, lr_groups={"net.3": {"lr": 2e-3, "warmup": 6}}, save_every=8)
T._stop_requested = keep_stop
os.environ["BF_RESUME_ANY"] = "1"
try:
    second = TT.run_train(T, root, "force", steps=16, lr_groups={"net.3": {"lr": 5e-3, "warmup": 3}}, save_every=8)
finally:
    os.environ.pop("BF_RESUME_ANY", None)
mf = T.TrainConfig().lr_min_frac
f_new = mf + (1 - mf) * 0.5 * (1 + math.cos(math.pi * (8 - 3) / (16 - 3)))                  # step 8 with warm-up 3 and 16 steps, by hand
row9 = next(r for r in second["rows"] if r["step"] == 9)
check(first_half["exc"] and "SystemExit: 75" in first_half["exc"] and second["exc"] is None and abs(row9["lr.net.3"] - 5e-3 * f_new) < 1e-12,
      f"BF_RESUME_ANY=1 with other lr_groups: the resumed run uses the new lr and warm-up (lr.net.3 at step 9 = {row9['lr.net.3']:.4e}, by hand {5e-3 * f_new:.4e})")
check(second["ck"]["opt"]["param_groups"][2]["base_lr"] == 5e-3 and second["ck"]["opt"]["param_groups"][2]["warmup"] == 3, "and the checkpoint it writes carries the new tags")

print("LR GROUPS GATE: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
sys.exit(1 if fails else 0)
