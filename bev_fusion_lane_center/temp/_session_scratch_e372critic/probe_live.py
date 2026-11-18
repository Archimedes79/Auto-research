"""critic probe (E372 live_inputs, 2026-11-17): the HEAD jtoken (git show, in memory) against the working tree; scratch, no record"""
import contextlib
import os
import subprocess
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402

torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import jtok as JT  # noqa: E402
import jtoken as JK  # noqa: E402

SRC_HEAD = subprocess.run(["git", "show", "HEAD:bev_fusion_lane_center/training/grid/jtoken.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
SRC_WT = open(JK.__file__, encoding="utf-8").read()
assert "b = {k: _const(batch[k]) for k in INPUTS}" in SRC_HEAD and "live_inputs" not in SRC_HEAD


def load(src, name):
    mod, path = types.ModuleType(name), list(sys.path)
    mod.__file__ = JK.__file__
    try:
        exec(compile(src, JK.__file__, "exec"), mod.__dict__)
    finally:
        sys.path[:] = path
    return mod


H = load(SRC_HEAD, "jtoken_head")
W = JK


def run(mod, kw, step, train, leaves, loss_kw, batch, amp=False, live=None):
    torch.manual_seed(0)
    m = mod.JTokenHead(**kw)
    if live is not None:
        m.live_inputs = live
    m.set_step(step)
    m.train(train)
    x = {k: (v.clone().requires_grad_(True) if (leaves and v.is_floating_point()) else v.clone()) for k, v in batch.items()}
    ctx = torch.autocast("cpu", dtype=torch.bfloat16) if amp else contextlib.nullcontext()
    with ctx:
        out = m(x)
        loss, info = mod.jtoken_loss(out, x, **loss_kw)
    loss.backward()
    return dict(m=m, out=out, loss=loss, pg={n: p.grad for n, p in m.named_parameters()}, ig={k: x[k].grad for k in mod.INPUTS if x[k].is_floating_point()})


def same(a, b):
    o = all(torch.equal(a["out"][k], b["out"][k]) for k in a["out"]) and set(a["out"]) == set(b["out"])
    g = list(a["pg"]) == list(b["pg"]) and all((a["pg"][n] is None and b["pg"][n] is None) or torch.equal(a["pg"][n], b["pg"][n]) for n in a["pg"])
    return o and g and torch.equal(a["loss"], b["loss"])


def fin(v):
    return v is not None and bool(torch.isfinite(v).all()) and float(v.abs().sum()) > 0


def all_fin(r):
    return all(fin(v) for v in r["ig"].values())


b_small = JK.synthetic_batch(3, 64, seed=1)
b_big = JK.synthetic_batch(2, 384, seed=2)
n_cases, bad = 0, []
for prior in (None, "rule"):
    kw = {} if prior is None else dict(prior=prior)
    for step in (0, JK.SWITCH - 1, JK.SWITCH, JK.SWITCH + 1):
        for train in (True, False):
            for leaves in (False, True):
                for loss_kw in ({}, dict(none_weight=0.25), dict(none_weight=0.25, pos_loss="max"), dict(none_weight=0.25, none_weight_near=1.0)):
                    h, w = run(H, kw, step, train, leaves, loss_kw, b_small), run(W, kw, step, train, leaves, loss_kw, b_small)
                    n_cases += 1
                    if not (same(h, w) and all(v is None for v in w["ig"].values()) and all(v is None for v in h["ig"].values())):
                        bad.append((prior, step, train, leaves, loss_kw))
                    if leaves and loss_kw == dict(none_weight=0.25):
                        lv = run(W, kw, step, train, leaves, loss_kw, b_small, live=True)
                        if not (same(h, lv) and all_fin(lv)):
                            bad.append(("live", prior, step, train))
print(f"HEAD vs working tree (default), synthetic 3 x 64: {n_cases} cases (prior none / rule, steps 0, 999, 1000, 1001, train / eval, leaves or not, 4 losses; live = HEAD in 16 of them): bad {bad}")
print("parameter tensors with a gradient (default head, step 1001):", sum(g is not None for g in run(W, {}, JK.SWITCH + 1, True, False, dict(none_weight=0.25), b_small)["pg"].values()))
for amp in (False, True):
    h, w = run(H, {}, JK.SWITCH + 1, True, True, dict(none_weight=0.25), b_big, amp=amp), run(W, {}, JK.SWITCH + 1, True, True, dict(none_weight=0.25), b_big, amp=amp)
    lv = run(W, {}, JK.SWITCH + 1, True, True, dict(none_weight=0.25), b_big, amp=amp, live=True)
    print(f"K 384 amp={amp}: HEAD == default {same(h, w)}, default == live (outputs, param grads) {same(w, lv)}, live input grads finite+nonzero {all_fin(lv)}")


def train_steps(mod, n=5, start=JK.SWITCH - 2, live=None):
    torch.manual_seed(0)
    m = mod.JTokenHead()
    if live is not None:
        m.live_inputs = live
    m.set_step(start)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for i in range(n):
        b = JK.synthetic_batch(2, 48, seed=10 + i)
        loss = mod.jtoken_loss(m(b), b, none_weight=0.25)[0]
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.state_dict()


sh, sw, sl = train_steps(H), train_steps(W), train_steps(W, live=True)
print("5 AdamW steps across the switch (998 -> 1003): HEAD == default", all(torch.equal(sh[k], sw[k]) for k in sh) and list(sh) == list(sw), "; default == live (cached inputs, no graph)", all(torch.equal(sw[k], sl[k]) for k in sw))

b = JK.synthetic_batch(2, 48, seed=3)
leaf = {k: v.clone().requires_grad_(True) for k, v in b.items() if k in JK.INPUTS and v.is_floating_point()}
x = dict(b, **{k: v * 1.0 for k, v in leaf.items()})
m = JK.JTokenHead(live_inputs=True)
m.set_step(JK.SWITCH + 1)
JK.jtoken_loss(m(x), x, none_weight=0.25)[0].backward()
print("live, non-leaf inputs: every leaf has a finite nonzero gradient", all(fin(v.grad) for v in leaf.values()))
for live in (False, True):
    x = dict(b, **{k: v * 1.0 for k, v in leaf.items()})
    m = JK.JTokenHead(live_inputs=live)
    m.set_step(JK.SWITCH + 1)
    out = m(x)
    loss = JK.jtoken_loss(out, x, none_weight=0.25)[0]
    x["pos"].add_(0.0)
    try:
        loss.backward()
        print(f"live={live}: in-place edit of batch[pos] after the forward -> backward ok")
    except RuntimeError as e:
        print(f"live={live}: in-place edit of batch[pos] after the forward -> RuntimeError ({str(e)[:90]}...)")

D64 = load(SRC_WT.replace(".float()", ".double()"), "jtoken_d64")
PG = JT.pair_geometry


def input_grads(mod, batch, dtype, pg=None, step=JK.SWITCH + 5):
    torch.manual_seed(0)
    m = mod.JTokenHead(live_inputs=True).to(dtype)
    m.set_step(step)
    m.eval()
    x = {k: (v.to(dtype).clone().requires_grad_(True) if v.is_floating_point() else v) for k, v in batch.items()}
    if pg is not None:
        JT.pair_geometry = pg
    try:
        loss = mod.jtoken_loss(m(x), x, none_weight=0.25)[0]
    finally:
        JT.pair_geometry = PG
    loss.backward()
    return {k: x[k].grad.double() for k in JK.INPUTS if x[k].is_floating_point()}, float(loss)


def pg_nogeo(pos, dirv, valid=None, e=None, **k):
    return PG(pos.detach(), dirv.detach(), valid, e=None if e is None else e.detach(), **k)


def pg_safe(pos, dirv, valid=None, e=None, **k):
    K = pos.shape[1]
    g, al = PG(pos, dirv, valid, e=e, **k)
    gd, _ = PG(pos.detach(), dirv, valid, e=e, **k)
    eye = torch.eye(K, dtype=torch.bool)[None, :, :, None]
    return torch.where(eye, gd, g), al


def rel(a, r):
    return float((a - r).norm() / r.norm())


for name, bb in (("3 x 64", JK.synthetic_batch(3, 64, seed=1)), ("2 x 384", JK.synthetic_batch(2, 384, seed=2))):
    g64, _ = input_grads(D64, bb, torch.float64)
    g64s, _ = input_grads(D64, bb, torch.float64, pg=pg_safe)
    g32, _ = input_grads(W, bb, torch.float32)
    g32s, _ = input_grads(W, bb, torch.float32, pg=pg_safe)
    gng, _ = input_grads(W, bb, torch.float32, pg=pg_nogeo)
    for k in ("pos", "dirv", "e"):
        print(f"{name} {k}: g64 vs g64safe {rel(g64[k], g64s[k]):.2e}; float32 vs float64: plain {rel(g32[k], g64[k]):.2e}, diag-safe {rel(g32s[k], g64s[k]):.2e}; geometry path cut changes g by {rel(gng[k], g32[k]):.3f}")
    others = ", ".join(f"{k} {rel(g32[k], g64[k]):.1e}" for k in g32 if k not in ("pos", "dirv", "e"))
    print(f"{name} other inputs float32 vs float64: {others}")
