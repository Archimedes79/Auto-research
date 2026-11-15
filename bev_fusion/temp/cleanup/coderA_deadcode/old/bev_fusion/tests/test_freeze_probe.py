"""train_only / freeze (training/freeze.py), the step probe (training/probe.py) and the debug tools (training/debug.py) on a toy network: CPU, no data, a few seconds.

  1. a prefix selects the module and everything below it; `fusion` = `fusion.`; train_only freezes the rest; freeze takes from train_only; a typo is an error with the
     nearest name; freezing everything is an error
  2. frozen means frozen: requires_grad off, the optimiser never sees the parameters, and BatchNorm statistics do not move even after the trainer calls model.train()
     (the control without the freeze moves them); `unchanged` finds a leak (a frozen BatchNorm that was left in train mode)
  3. the probe: grad.<module> and upd.<module> equal an independent computation around a real optimiser step, a frozen module has no keys, nothing is copied on
     steps that are not logged; the timer separates the data wait from the step; the incident dump is atomic, capped and never raises
  4. debug: one batch can be overfitted (and a broken loss cannot), the frozen part stays bit-equal over real steps while the trained part moves
  5. the keys are the ones the trainsuite reads: a toy training that logs them is read back by autoresearch/trainsuite (module table, no finding on a healthy run,
     dead_module when a module is detached)
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_freeze_probe.py
"""
import json
import os
import sys
import tempfile
import time

import torch
from torch import nn

sys.path.insert(0, "training")
sys.path.insert(0, os.path.join("..", "autoresearch"))
import debug  # noqa: E402
import freeze  # noqa: E402
import probe  # noqa: E402

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU())
        self.head = nn.ModuleDict({"a": nn.Linear(8, 2), "b": nn.Linear(8, 1)})
        self.lane = nn.Linear(8, 2)
        self.register_buffer("grid", torch.arange(3.0))                 # a constant, like the depth bins of the lift

    def forward(self, x):
        h = self.fusion(x)
        return torch.cat([self.head["a"](h), self.head["b"](h), self.lane(h)], 1)


def toy(seed=0):
    torch.manual_seed(seed)
    return Toy()


torch.manual_seed(1)
X, Y = torch.randn(16, 4), torch.randn(16, 5)
batch = (X, Y)
loss_fn = lambda m, b: ((m(b[0]) - b[1]) ** 2).mean()          # noqa: E731
trainable_names = lambda pl: sorted(n for n, t in pl.items() if t)                 # noqa: E731

# 1 prefixes
m = toy()
check(trainable_names(freeze.plan(m, train_only=["fusion."])) == sorted(n for n, _ in m.named_parameters() if n.startswith("fusion.")), "train_only ['fusion.'] trains the fusion and nothing else")
check(freeze.plan(m, train_only=["fusion"]) == freeze.plan(m, train_only=["fusion."]), "'fusion' and 'fusion.' are the same prefix")
pl = freeze.plan(m, freeze=["head.a."])
check(not pl["head.a.weight"] and not pl["head.a.bias"] and pl["head.b.weight"] and pl["lane.weight"], "freeze ['head.a.'] freezes that module and only that")
pl = freeze.plan(m, train_only=["head."], freeze=["head.b."])
check(trainable_names(pl) == ["head.a.bias", "head.a.weight"], "train_only head. + freeze head.b.: freeze takes from train_only")
check(freeze.plan(m) == {n: True for n, _ in m.named_parameters()}, "neither: everything trains")
for bad in (["fuson."], [""], ["."], ["head.c."]):
    try:
        freeze.plan(m, train_only=bad)
        check(False, f"prefix {bad} must be refused")
    except ValueError as e:
        if bad == ["fuson."]:
            check("did you mean fusion" in str(e), "a typo names the nearest module")
check(True, "unknown / empty prefixes are errors")
try:
    freeze.apply(toy(), freeze=["fusion.", "head.", "lane."])
    check(False, "freezing everything must be refused")
except ValueError:
    check(True, "freezing everything is an error (nothing would train)")
check(freeze._under("lane.weight", "lane") and freeze._under("lane", "lane.") and not freeze._under("lanehead.weight", "lane") and not freeze._under("lan.w", "lane"), "a prefix is a whole module name, not a string prefix (lane is not lanehead)")
check(freeze.signature() == "-" and freeze.signature(["b.", "a."], ["c."]) == "train_only=a.,b.;freeze=c.", "the signature is stable and sorted")

# 2 frozen means frozen
m = toy()
freeze.apply(m, train_only=["head."])
check(all(not p.requires_grad for n, p in m.named_parameters() if not n.startswith("head.")) and len(freeze.trainable(m)) == 4, "requires_grad off outside the trained part, the optimiser list holds only the head")
rep = freeze.report(m)
check("head" in rep and "frozen" in rep and rep.splitlines()[-1].lstrip().startswith("total"), "the report has a line per module and a total")
fk = freeze.frozen_keys(m)
check(fk <= set(m.state_dict()) and {"fusion.0.weight", "fusion.1.running_mean", "fusion.1.running_var", "fusion.1.num_batches_tracked", "lane.bias"} <= fk and not {k for k in fk if k.startswith("head.")},
      "frozen_keys: the frozen parameters and the BatchNorm buffers of the frozen modules, nothing of the trained head")
snap = freeze.snapshot(m)
check("fusion.1.running_mean" in snap and "head.a.weight" not in snap, "the snapshot holds the frozen parameters and all buffers, not the trained ones")
opt = torch.optim.SGD(freeze.trainable(m), lr=0.1)
for _ in range(5):
    m.train()                                                  # what a trainer does before every epoch / evaluation
    opt.zero_grad()
    loss_fn(m, batch).backward()
    opt.step()
check(freeze.unchanged(m, snap) == [] and not m.fusion[1].training, "5 steps with model.train() each time: frozen weights and BatchNorm statistics are bit-equal, the frozen BatchNorm is in eval mode")
m2 = toy()
for p in m2.fusion.parameters():
    p.requires_grad_(False)                                    # frozen by hand, without the eval hook: the leak the module exists to prevent
sn2 = freeze.snapshot(m2)
m2.train()
m2(X)
leak = freeze.unchanged(m2, sn2)
check("fusion.1.running_mean" in leak, "control: a frozen BatchNorm left in train mode DOES change its statistics, and `unchanged` finds it")
fp0 = freeze.fingerprint(m)
check(freeze.fingerprint(m) == fp0 and freeze.fingerprint(toy(1)) != freeze.fingerprint(toy(2)), "the fingerprint is stable and tells weights apart")
check(freeze.unchanged(m, {"nope": torch.zeros(1)}) == ["nope"], "a name that is gone counts as changed")

# 3 probe
m = toy()
freeze.apply(m, freeze=["lane."])
pr = probe.ModuleProbe(m, every=5)
check(pr.due(0) and pr.due(10) and not pr.due(3) and pr.begin(3) is None, "due on every 5th step; begin() gives None otherwise (no weight copy)")
opt = torch.optim.SGD(freeze.trainable(m), lr=0.1)
m.train()
p = pr.begin(10)
opt.zero_grad()
loss_fn(m, batch).backward()
p.grads()
w0 = {n: q.detach().clone() for n, q in m.named_parameters()}
g_by = {k: sum(float(q.grad.pow(2).sum()) for n, q in m.named_parameters() if n.startswith(k + ".") and q.grad is not None) ** 0.5 for k in ("fusion", "head")}
opt.step()
row = p.finish()
upd = {k: (sum(float((q.detach() - w0[n]).pow(2).sum()) for n, q in m.named_parameters() if n.startswith(k + ".")) / sum(float(w0[n].pow(2).sum()) for n in w0 if n.startswith(k + "."))) ** 0.5 for k in ("fusion", "head")}
check(sorted(row) == ["grad.fusion", "grad.head", "upd.fusion", "upd.head"], f"keys: grad. / upd. of the modules that train, none for the frozen lane: {sorted(row)}")
check(all(abs(row["grad." + k] - g_by[k]) < 1e-6 * max(1.0, g_by[k]) for k in g_by) and all(abs(row["upd." + k] - upd[k]) < 1e-6 for k in upd), "grad. and upd. equal an independent computation around the optimiser step")
check(p.before == {}, "the weight copy is given back after finish()")


class Zero(nn.Module):
    def __init__(self):
        super().__init__()
        self.zero = nn.Linear(2, 2)
        nn.init.zeros_(self.zero.weight)
        nn.init.zeros_(self.zero.bias)

    def forward(self, x):
        return self.zero(x)


zm = Zero()
zp = probe.ModuleProbe(zm, every=1).begin(1)
zo = torch.optim.SGD(zm.parameters(), lr=1.0)
((zm(torch.ones(4, 2)) - 1.0) ** 2).mean().backward()
zp.grads()
zo.step()
zrow = zp.finish()
dn = float(sum(q.detach().pow(2).sum() for q in zm.parameters())) ** 0.5
check(abs(zrow["upd.zero"] - dn / probe.UPD_FLOOR) < 1e-6 * dn / probe.UPD_FLOOR and zrow["upd.zero"] < 1e4, f"a module that starts at zero: the update is measured against {probe.UPD_FLOOR}, not 1e-30 ({zrow['upd.zero']:.3g})")
pr2 = probe.ModuleProbe(m, every=1, depth=2)
pp = pr2.begin(1)
opt.zero_grad()
loss_fn(m, batch).backward()
pp.grads()
opt.step()
check("grad.head.a" in pp.finish() and probe.module_of("head.a.weight", 2) == "head.a" and probe.module_of("lane.weight") == "lane", "depth 2 splits the head into head.a / head.b")
t = probe.StepTimer()
check(t.row() == {}, "the first step has no start: no t_step")
with t.data():
    time.sleep(0.05)
time.sleep(0.05)
r1 = t.row()
check(0.045 <= r1["t_data"] < 0.2 and r1["t_step"] >= r1["t_data"] + 0.04, f"data wait inside the step: {r1}")
check(t.row()["t_data"] == 0.0, "the data time restarts with every step")
d = tempfile.mkdtemp()
path = probe.dump_incident(d, 800, "nonfinite", batch_ids=["tok1", "tok2"], losses={"total": torch.tensor(float("nan")), "lines": 1.5}, extra={"grad_norm": 3.0})
rec = probe.read_incident(path)
check(os.path.basename(path) == "incident_800.json" and rec["kind"] == "nonfinite" and rec["batch"] == ["tok1", "tok2"] and rec["losses"]["lines"] == 1.5 and rec["grad_norm"] == 3.0
      and not [f for f in os.listdir(d) if f.endswith(".tmp")], "the dump has step, kind, batch ids and losses (a NaN tensor too) and leaves no .tmp")
old_cap = probe.MAX_INCIDENTS
probe.MAX_INCIDENTS = 3
for s in (1, 2, 3, 4):
    probe.dump_incident(d, s, "spike")
check(len([f for f in os.listdir(d) if f.startswith("incident_")]) == 3 and probe.dump_incident(d, 99, "spike") is None, "the cap keeps the first dumps and writes no more")
probe.MAX_INCIDENTS = old_cap
check(probe.dump_incident(os.path.join(d, "no", "such", "dir"), 1, "spike") is None, "a dump that fails returns None and never raises")

# 4 debug
m = toy()
losses = debug.overfit_one_batch(m, loss_fn, batch, steps=300, lr=1e-2)
check(debug.overfit_ok(losses) and losses[-1] < losses[0], f"one batch can be overfitted ({losses[0]:.3f} -> {losses[-1]:.3f})")
check(not debug.overfit_ok([1.0, 1.0, 1.0]) and not debug.overfit_ok([1.0, float("nan")]) and not debug.overfit_ok([]) and not debug.overfit_ok([1.0, float("inf"), 0.0]), "a flat, NaN, empty or infinite loss curve is not overfit")
floor = lambda mm, b: (mm(b[0]) ** 2).mean() + 5.0                  # noqa: E731  a loss with a floor it cannot go below: falls, but not towards 0
check(not debug.overfit_ok(debug.overfit_one_batch(toy(), floor, batch, steps=200, lr=1e-2)), "a loss that cannot reach 0 on one batch is reported as not overfit (a bug in the loss or the targets)")
orig = toy()
fp_orig = freeze.fingerprint(orig)
changed, moved = debug.frozen_stayed(orig, loss_fn, batch, train_only=["head."])
check(freeze.fingerprint(orig) == fp_orig and all(p.requires_grad for p in orig.parameters()), "frozen_stayed works on a copy: the model it was given is untouched")
check(changed == [] and "head.a.weight" in moved and not [n for n in moved if not n.startswith("head.")], f"frozen_stayed: nothing frozen changed, the head moved ({len(moved)} parameters)")
changed, moved = debug.frozen_stayed(toy(), loss_fn, batch, freeze_=["lane.", "fusion."])
check(changed == [] and sorted(moved) == ["head.a.bias", "head.a.weight", "head.b.bias", "head.b.weight"], "frozen_stayed with freeze_ names")
check(debug.replay_ids({"batch": ["a", "b"]}, lambda i: i.upper()) == ["A", "B"], "replay_ids rebuilds the batch from the ids")
try:
    debug.replay_ids({"batch": None}, str)
    check(False, "an incident without ids cannot be replayed")
except ValueError:
    check(True, "an incident without batch ids says so")
with debug.anomaly():
    check(torch.is_anomaly_enabled(), "anomaly mode on inside the context")
check(not torch.is_anomaly_enabled(), "and off after it")

# 4b findings of the critic pass (2026-11-01)
class Holder(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.mover = shared

    def forward(self, x):
        return self.mover(x)


class Shared(nn.Module):
    def __init__(self):
        super().__init__()
        self.mover = nn.Linear(2, 2)                                   # one module under three names, like the TimeMover of the heads
        self.a, self.b = Holder(self.mover), Holder(self.mover)
        self.own = nn.Linear(2, 2)


torch.manual_seed(0)
sh = Shared()
check(sorted(n for n, _ in sh.named_parameters()) == ["mover.bias", "mover.weight", "own.bias", "own.weight"], "premise: named_parameters lists the shared module once")
check(freeze.plan(sh, freeze=["a."]) == {"mover.weight": False, "mover.bias": False, "own.weight": True, "own.bias": True}, "freezing one of the names of a shared module freezes it (the 'frozen' head must not change)")
check(freeze.plan(sh, train_only=["b.mover."]) == {"mover.weight": True, "mover.bias": True, "own.weight": False, "own.bias": False}, "train_only through any of its names trains it, and the prefix b.mover. is accepted")
check(freeze.plan(sh, train_only=["a."], freeze=["b."])["mover.weight"] is False, "freeze wins over train_only through another name")
freeze.apply(sh, freeze=["a."])
check({"mover.weight", "a.mover.weight", "b.mover.bias"} <= freeze.frozen_keys(sh) and not {k for k in freeze.frozen_keys(sh) if k.startswith("own.")}, "frozen_keys lists a shared module under every name it has")
fk2 = freeze.frozen_keys((lambda mm: (freeze.apply(mm, train_only=["fusion."]), mm)[1])(toy()))
check("grid" in fk2 and "lane.weight" in fk2 and not ({"fusion.1.running_mean", "fusion.1.running_var", "fusion.0.weight"} & fk2),
      "frozen_keys: a constant buffer is fixed, the statistics of a TRAINED BatchNorm are not")
pre = toy()
for q in pre.fusion.parameters():
    q.requires_grad_(False)                                            # frozen by the model itself, like freeze_camera
pl = freeze.plan(pre, freeze=["head."])
check(not any(v for n, v in pl.items() if n.startswith("fusion.")) and pl["lane.weight"] and not pl["head.a.weight"], "what the model froze itself is never unfrozen by the plan")
try:
    freeze.plan(pre, train_only=["fusion."])
    check(False, "train_only of what the model keeps frozen must be refused")
except ValueError as e:
    check("keeps frozen itself" in str(e), "train_only of parameters the model keeps frozen is an error")
freeze.apply(pre, train_only=["head."])
pre.train()
pre(X)
check(all(not q.requires_grad for q in pre.fusion.parameters()) and not pre.fusion[1].training, "apply never sets requires_grad True; the frozen part is in eval mode after model.train() and a forward")
check(freeze.plan(toy(), train_only="fusion") == freeze.plan(toy(), train_only=["fusion"]) and freeze.signature("a.", "b.") == freeze.signature(["a."], ["b."]), "a bare string is one prefix, not its characters")
changed, moved = debug.frozen_stayed(toy(), loss_fn, batch, train_only=["fusion."])
check(changed == [] and "fusion.0.weight" in moved, "a trained BatchNorm is no false alarm of frozen_stayed (only the frozen part is watched)")
pq = probe.ModuleProbe(toy(), every=1).begin(1)
check(pq.before == {} and pq.finish() == {}, "begin() copies nothing; finish() without grads() has nothing to compare")
mm_ = toy()
pq = probe.ModuleProbe(mm_, every=1).begin(1)
loss_fn(mm_, batch).backward()
pq.grads()
check(len(pq.before) == len(list(mm_.parameters())), "the weight copy is taken in grads(), after the backward")
check(probe.batch_tokens({"tokens": ["a", "b"]}) == ["a", "b"] and probe.batch_tokens([{"tokens": ["a"]}, {"tokens": ["b"]}]) == ["a", "b"] and probe.batch_tokens(5) == []
      and probe.batch_tokens({"tokens": 7}) == [] and probe.batch_tokens(None) == [], "batch_tokens: a batch, a list of clips, and anything odd gives []")

# 5 the keys are the ones the trainsuite reads
from trainsuite import health, modules as ts_modules  # noqa: E402
from trainsuite.runs import Run  # noqa: E402


def toy_run(detach_lane):
    d = tempfile.mkdtemp()
    json.dump({"steps": 1000, "freeze": []}, open(os.path.join(d, "config.json"), "w"))
    mm = toy()
    pr_ = probe.ModuleProbe(mm, every=1)
    o = torch.optim.SGD(mm.parameters(), lr=0.05)
    tm = probe.StepTimer()
    t0 = time.time() - 100
    with open(os.path.join(d, "log.jsonl"), "w") as f:
        for s in range(1, 101):
            with tm.data():
                pass
            pp_ = pr_.begin(s)
            o.zero_grad()
            if detach_lane:                                                                     # the lane output is not in the loss: its parameters get no gradient at all
                h = mm.fusion(X)
                loss_ = ((torch.cat([mm.head["a"](h), mm.head["b"](h)], 1) - Y[:, :3]) ** 2).mean()
            else:
                loss_ = ((mm(X) - Y) ** 2).mean()
            loss_.backward()
            pp_.grads()
            o.step()
            row = {"step": s, "t": t0 + s, "total": float(loss_.detach()), "lr": 0.05, "grad_norm": 1.0, "s_per_step": 1.0, "gpu_gb": 1.0}
            row.update(pp_.finish())
            row.update(tm.row())
            f.write(json.dumps(row) + "\n")
    open(os.path.join(d, "train.log"), "w").write("step 100: total 1.0\n")
    return Run(d)


healthy = toy_run(False)
tab = {t["module"]: t for t in ts_modules.module_table(healthy)}
check(sorted(tab) == ["fusion", "head", "lane"] and all(t["grad"] > 0 and t["upd"] > 0 for t in tab.values()), "the trainsuite reads grad. / upd. of the toy training: a module table with three live modules")
check(not {f.code for f in health.check(healthy)} & {"dead_module", "grad_module", "update_ratio", "data_bound", "incident"}, "a healthy toy run raises no module finding")
dead = toy_run(True)
check("dead_module" in {f.code for f in health.check(dead)}, "a detached module is a dead_module for the trainsuite")

print("FREEZE / PROBE GATE: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
sys.exit(1 if fails else 0)
