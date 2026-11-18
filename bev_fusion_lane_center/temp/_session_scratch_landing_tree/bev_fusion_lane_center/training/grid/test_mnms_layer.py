"""E292 tests (CPU, torch): the one-pass Matrix-NMS layer as a switch of the model (`QGraph(soft_nms=('mnms', R, sigma))`, `qmnms.matrix_nms_logit`), the fine-tune phase 9 (pyramid frozen), the
checkpoint round trip, and the mutations the card names (each one must be caught by a check below).
Run: python test_mnms_layer.py [--real]    (`_py14.sh test_mnms_layer.py`)
--real: also the kill criterion of the card of E292: the layer reproduces `qgraph_readout.matrix_nms` on the scores of 20 frames of B's full dump (`temp/e276/predfull_e273_p_s0.pkl`, written by
`qgraph_run_eval.py score --dump-full`; needs the stored dump, so it is in run_all_tests.sh, not in the fast gates)."""
import json
import os
import pickle
import sys
import tempfile

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in (os.path.join(ROOT, "evaluation"), os.path.dirname(os.path.abspath(__file__))):
    sys.path.insert(0, sub)
import qgraph as QG  # noqa: E402
import qgraph_readout as QR  # noqa: E402
import qgates as QGt  # noqa: E402
import qmnms as QMN  # noqa: E402
import qnodes as QN  # noqa: E402
import qpredict as QP  # noqa: E402

n_ok = 0


def check(cond, msg=""):
    global n_ok
    assert cond, msg
    n_ok += 1


def refused(fn, what):
    try:
        fn()
    except AssertionError:
        check(True)
        return
    check(False, f"accepted {what}")


# ---- 1. the switch in the model: default = no change; on: the damped logit is `no['e']`, the head's own is `no['e_raw']`, same weights, no new parameter
torch.manual_seed(0)
plain = QG.QGraph(width=16, f05=16, d=32)
mm = QG.QGraph(width=16, f05=16, d=32, soft_nms=["mnms", 1.75, 0.1])                           # the list form, as it comes back from a checkpoint (json)
mm.load_state_dict(plain.state_dict())
check(set(plain.state_dict()) == set(mm.state_dict()), "the layer has no parameters")
check(plain.mnms is None and plain.soft_nms is None and mm.mnms == (1.75, 0.1) and mm.soft_nms is None, (plain.mnms, mm.mnms, mm.soft_nms))
x = torch.randn(1, 39, 200, 200)
with torch.no_grad():
    o_p = plain({"x": x}, upto="nodes", gt=False)["no"]
    o_m = mm({"x": x}, upto="nodes", gt=False)["no"]
check("e_raw" not in o_p, "the default model must be the E273 model")
check(torch.equal(o_m["e_raw"], o_p["e"]), "the raw logit of the model with the layer is the plain model's logit")
check(torch.allclose(o_m["e"], QMN.matrix_nms_logit(o_p["e"], o_p["pos"], 1.75, 0.1)), "no['e'] must be the damped logit")
check(float(o_m["e"].sigmoid().sum()) <= float(o_p["e"].sigmoid().sum()) + 1e-3, "the layer only damps")
check(bool((o_m["e"] <= o_p["e"] + 1e-4).all()), "e' above e somewhere")

# ---- 2. arguments: refused in the model, the E277 triple unchanged
for bad in (["mnms", 1.75], ["mnms", 2.5, 0.1], ["mnms", 1.75, 0.0], ["mnms", 0.0, 0.1], ["mnms", 1.75, 0.1, 3]):
    refused(lambda bad=bad: QG.QGraph(width=16, f05=16, d=32, soft_nms=bad), bad)
refused(lambda: QMN.matrix_nms_logit(o_p["e"], o_p["pos"], 2.5, 0.1), "a radius above 2 m in the layer")
check(QG.QGraph(width=16, f05=16, d=32, soft_nms=(1.5, 0.1, 3)).soft_nms == (1.5, 0.1, 3.0), "the E277 triple must keep working")
refused(lambda: QG.QGraph(width=16, f05=16, d=32, soft_nms=(2.5, 0.1, 3)), "the E277 triple with a radius above 2 m")

# ---- 3. the checkpoint carries the switch (list form 'mnms'), the plain and the E277 forms are unchanged
with tempfile.TemporaryDirectory() as d:
    m0 = QG.QGraph(width=48)
    for args, mnms, soft in ((dict(width=48, soft_nms=["mnms", 1.75, 0.1]), (1.75, 0.1), None), (dict(width=48, soft_nms=[1.5, 0.1, 3.0]), None, (1.5, 0.1, 3.0)), (dict(width=48), None, None)):
        torch.save(dict(state=m0.state_dict(), args=args), os.path.join(d, "model.pt"))
        lm = QP.load_model(d, "cpu")
        check(lm.mnms == mnms and lm.soft_nms == soft, (args, lm.mnms, lm.soft_nms))
    os.remove(os.path.join(d, "model.pt"))                                                       # a RUNNING arm: ckpt.pt + config.json
    torch.save(dict(model=m0.state_dict()), os.path.join(d, "ckpt.pt"))
    json.dump(dict(args=dict(width=48, soft_nms=["mnms", 1.75, 0.1])), open(os.path.join(d, "config.json"), "w"))
    check(QP.load_model(d, "cpu").mnms == (1.75, 0.1), "load_model of a running arm must restore the layer from config.json")

# ---- 4. phase 9: nodes, links, map encoder and assigner train, the pyramid does not
ids = lambda mod: {id(p) for p in mod.parameters()}  # noqa: E731
t9 = {id(p) for p in QG.trainable_params(mm, 9)}
check(t9 == ids(mm.nodes) | ids(mm.links) | ids(mm.menc) | ids(mm.asg), "phase 9 must train exactly nodes / links / menc / asg")
check(not (t9 & ids(mm.pyr)), "the pyramid must not be trainable in phase 9")
check(t9 == {id(p) for p in QG.trainable_params(mm, 7)} - ids(mm.pyr), "phase 9 = the joint phase 7 without the pyramid")
check(QG.PHASE_UPTO[9] == "assign" and QG.PHASE_FROZEN[9] == ("pyr",), (QG.PHASE_UPTO[9], QG.PHASE_FROZEN[9]))
b = QG._toy_batch(2)


def grad_sums(model, phase):
    """summed |grad| of the pyramid and of the node head after one backward pass of the loss of `phase`"""
    model.zero_grad()
    torch.manual_seed(7)                                                                      # the loss has a random part (sampling): the same draw for every call
    loss, _ = QG.qgraph_loss(model, b, phase)
    loss.backward()
    s = lambda mod: sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)  # noqa: E731
    return s(model.pyr), s(model.nodes), float(loss)


g_pyr, g_nodes, _ = grad_sums(mm, 9)
check(g_pyr == 0.0 and g_nodes > 0.0, (g_pyr, g_nodes))
# the phase 9 gate: numbers only, never a stop (like phase 7), and no pixel reading (the pyramid is frozen); the readings themselves are stubbed (they need the VAL store)
called = []
saved = {k: getattr(QGt, k) for k in ("gate_pixels", "gate_nodes", "gate_links", "gate_sections", "gate_assign")}
for k in saved:
    setattr(QGt, k, lambda model, run, n, k=k: (called.append(k), {"pass": False, "why": "stub", "x": 1.0})[1])
try:
    res9 = QGt.gate(mm, 9, type("R", (), {"log": staticmethod(lambda *a: None)})(), n=1)
finally:
    for k, v in saved.items():
        setattr(QGt, k, v)
check(res9["pass"] is True and "numbers only" in res9["why"] and "p2" not in res9 and set(res9) >= {"p3", "p4", "p5", "p6"} and "gate_pixels" not in called, (res9, called))

# ---- 5. the loss sees the damped existence e' (not the head's own e_raw): with the layer on, the node loss differs from the plain model's on the same batch. At a fresh init the offsets are about 0 and
# cells 2 m apart do not overlap (the layer does nothing), so the node head is wrapped: its odd columns move 1.5 m toward the left neighbour (every pair of neighbours overlaps)
import copy  # noqa: E402


class Squeeze(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, *a):
        o = dict(self.head(*a))
        pos = o["pos"].clone()
        pos[:, 1, :, 1::2] = pos[:, 1, :, 1::2] - 1.5
        o["pos"] = pos
        return o


def squeezed(model):
    model = copy.deepcopy(model)
    model.nodes = Squeeze(model.nodes)
    return model


plain_s, mm_s = squeezed(plain), squeezed(mm)
l_plain = grad_sums(plain_s, 3)[2]
l_mm = grad_sums(mm_s, 3)[2]
check(abs(l_plain - l_mm) > 1e-4, (l_plain, l_mm))

# ---- 6. mutations (each must change what the checks above / below measure)
# 6a. the node loss reading e_raw: the loss of the model with the layer becomes the plain model's loss
orig_node_loss = QG.node_loss


def reads_raw(no, batch, *a, **k):
    no = dict(no)
    no["e"] = no.get("e_raw", no["e"])
    return orig_node_loss(no, batch, *a, **k)


QG.node_loss = reads_raw
try:
    l_mut = grad_sums(mm_s, 3)[2]
finally:
    QG.node_loss = orig_node_loss
check(abs(l_mut - l_plain) < 1e-5 and abs(l_mut - l_mm) > 1e-4, "mutation 'node loss reads e_raw' must be visible in the loss of the model with the layer")
# 6b. the pyramid not frozen in phase 9: the pyramid gets a gradient (the frozen-pyramid check above sees it)
QG.PHASE_FROZEN[9] = ()
try:
    g_pyr_mut = grad_sums(mm, 9)[0]
finally:
    QG.PHASE_FROZEN[9] = ("pyr",)
check(g_pyr_mut > 0.0, "mutation 'pyramid not frozen' must give the pyramid a gradient")
# 6c. the layer itself against the numpy function: random frames, three settings; the order sign and the compensation term mutated in the source must be caught


def numpy_gap(fn):
    torch.manual_seed(5)
    worst = 0.0
    cell = torch.stack(torch.meshgrid(-50 + 2 * (torch.arange(50).float() + 0.5), -50 + 2 * (torch.arange(50).float() + 0.5), indexing="ij"), 0)[None]
    for radius, sigma in ((1.75, 0.1), (1.5, 0.5), (2.0, 0.2)):
        e_logit = torch.randn(2, 50, 50) * 2.5
        pos = cell + (torch.rand(2, 2, 50, 50) * 2 - 1)
        out = fn(e_logit, pos, radius, sigma).sigmoid()
        for k in range(2):
            e = e_logit[k].sigmoid().flatten().numpy().astype(float)
            ref = QR.matrix_nms(pos[k].flatten(1).t().numpy().astype(float), e, radius, thr=0.5, sigma=sigma)
            mine = out[k].flatten().numpy().astype(float)
            sel = e >= 0.5
            worst = max(worst, float(np.abs(np.where(ref > 0, ref, 0.0) - np.where(sel & (mine >= 0.5), mine, 0.0))[sel].max()))
    return worst


src = open(QMN.__file__, encoding="utf-8").read()
check(numpy_gap(QMN.matrix_nms_logit) < 2e-5, numpy_gap(QMN.matrix_nms_logit))
for name, a, bb in (("order sign", "(ej.detach() > ed)", "(ej.detach() < ed)"), ("compensation term", "(o.square() - cj.square()) / sigma", "o.square() / sigma"), ("radius in the overlap", "1.0 - d / radius", "1.0 - d / (radius * 0.5)")):
    check(src.count(a) == 1, (name, "mutation anchor not found: the layer's source changed, update the mutation"))
    ns = {}
    exec(compile(src.replace(a, bb), "<mutant " + name + ">", "exec"), ns)
    gap = numpy_gap(ns["matrix_nms_logit"])
    check(gap > 1e-3, f"mutation '{name}' is not caught by the comparison with the numpy function (gap {gap:.2e})")

# ---- 7. batch / dtype: a batch is a stack of independent frames; bfloat16 in, float32 out, no NaN at the border
torch.manual_seed(1)
e3, p3 = torch.randn(3, 50, 50) * 2.0, torch.rand(3, 2, 50, 50) * 100 - 50
o3 = QMN.matrix_nms_logit(e3, p3)
for k in range(3):
    check(torch.equal(o3[k], QMN.matrix_nms_logit(e3[k:k + 1], p3[k:k + 1])[0]), "a batch must equal its frames one by one")
ob = QMN.matrix_nms_logit(e3.bfloat16(), p3.bfloat16())
check(ob.dtype == torch.float32 and torch.isfinite(ob).all(), (ob.dtype,))

# ---- 8. what the layer does where the hard step does something else (documentation of the E292 finding, no claim): an isolated node is not changed, two nodes 1 m apart: the weaker one is damped below 0.5
lg = torch.logit(torch.tensor([[[0.9, 0.7]]]))
pp = torch.tensor([[[[0.0, 0.0]], [[0.0, 1.0]]]]).reshape(1, 2, 1, 2)
oo = QMN.matrix_nms_logit(lg, pp).sigmoid()[0, 0]
check(abs(float(oo[0]) - 0.9) < 1e-4 and float(oo[1]) < 0.5, oo)
lg_far = QMN.matrix_nms_logit(lg, torch.tensor([[[[0.0, 0.0]], [[0.0, 3.0]]]]).reshape(1, 2, 1, 2)).sigmoid()[0, 0]
check(abs(float(lg_far[0]) - 0.9) < 1e-4 and abs(float(lg_far[1]) - 0.7) < 1e-4, lg_far)
# the E277 form and this one agree on what they must: both keep an isolated node and drop a node on top of a stronger one
so = QN.soft_suppress_logit(lg, pp, 1.5, 0.1, 3).sigmoid()[0, 0]
check(float(so[0]) > 0.5 and float(so[1]) < 0.5 and float(oo[0]) > 0.5 and float(oo[1]) < 0.5, (so, oo))

# ---- 9. the layer is applied with the model's OWN (R, sigma) (at a fresh init nothing is damped, so section 1 alone would pass a model that skipped the layer): on the squeezed model no['e'] is the damped logit of
# that setting, it differs from e_raw in many cells, and a model with another sigma delivers other values
def node_out(model):
    with torch.no_grad():
        return model({"x": x}, upto="nodes", gt=False)["no"]


o_sq = node_out(mm_s)
check(torch.allclose(o_sq["e"], QMN.matrix_nms_logit(o_sq["e_raw"], o_sq["pos"], 1.75, 0.1), atol=1e-5), "no['e'] of the squeezed model must be the damped logit of the model's own (R, sigma)")
n_damped = int((o_sq["e"] < o_sq["e_raw"] - 1e-4).sum())
check(n_damped > 500, f"the squeezed model damps only {n_damped} cells: the layer is not applied")
mm_other = QG.QGraph(width=16, f05=16, d=32, soft_nms=["mnms", 1.75, 1.0])
mm_other.load_state_dict(plain.state_dict())
check(not torch.allclose(node_out(squeezed(mm_other))["e"], o_sq["e"], atol=1e-3), "another sigma must deliver other values (a hard-wired sigma would not)")

# ---- 10. phase 9 = phase 7 without the pixel loss, term by term (one draw of the random part); a phase set that loses a loss term fails here
def seeded(model, phase):
    torch.manual_seed(7)
    model.zero_grad()
    loss, info = QG.qgraph_loss(model, b, phase)
    return float(loss), info


l7, i7 = seeded(mm, 7)
l2, i2 = seeded(mm, 2)
l9, i9 = seeded(mm, 9)
check(abs(l7 - l2 - l9) < 1e-3 * max(1.0, abs(l9)), (l7, l2, l9))
check(set(i9) == set(i7) - set(i2) and all(abs(i9[k] - i7[k]) < 1e-3 * max(1.0, abs(i7[k])) for k in i9), (sorted(i9), sorted(set(i7) - set(i2))))
check(any(k.startswith(p) for k in i9 for p in ("n_", "l_", "s_", "a_")) and not any(k.startswith("px_") for k in i9) and {k[:2] for k in i9} >= {"n_", "l_", "s_", "a_"}, sorted(i9))

# ---- 11. an optimiser step as qgraph_train.py sets it up (requires_grad toggled, params=trainable_params): the pyramid stays torch.equal, the node head moves
mm2 = copy.deepcopy(mm)
pyr0 = {k: v.clone() for k, v in mm2.pyr.state_dict().items()}
nodes0 = {k: v.clone() for k, v in mm2.nodes.state_dict().items()}
for p in mm2.parameters():
    p.requires_grad_(False)
ps9 = QG.trainable_params(mm2, 9)
for p in ps9:
    p.requires_grad_(True)
opt = torch.optim.Adam(ps9, lr=1e-3)
for _ in range(2):
    torch.manual_seed(7)
    loss9, _ = QG.qgraph_loss(mm2, b, 9)
    opt.zero_grad()
    loss9.backward()
    opt.step()
check(all(torch.equal(pyr0[k], v) for k, v in mm2.pyr.state_dict().items()), "the pyramid changed in a phase 9 step")
check(any(not torch.equal(nodes0[k], v) for k, v in mm2.nodes.state_dict().items()), "the node head did not move in a phase 9 step")

# ---- 12. the switch end to end: --soft-nms-mnms -> args['soft_nms'] -> model.pt -> load_model; the two forms together are refused
sys.path.insert(0, os.path.join(ROOT, "training", "train"))
import qgraph_train as QT  # noqa: E402

sn = QT.resolve_soft_nms(None, [1.75, 0.1])
check(sn == ["mnms", 1.75, 0.1] and QT.resolve_soft_nms([1.5, 0.1, 3.0], None) == [1.5, 0.1, 3.0] and QT.resolve_soft_nms(None, None) is None, sn)
refused(lambda: QT.resolve_soft_nms([1.5, 0.1, 3.0], [1.75, 0.1]), "both forms of the switch")
with tempfile.TemporaryDirectory() as d:
    torch.save(dict(state=QG.QGraph(width=48, soft_nms=sn).state_dict(), args=dict(width=48, soft_nms=sn)), os.path.join(d, "model.pt"))
    check(QP.load_model(d, "cpu").mnms == (1.75, 0.1), "the chain flag -> args -> model.pt -> load_model must restore the layer")

# ---- 13. the gradient of the log-space layer against the analytic one: two nodes 1 m apart, the stronger A (e 0.9) over the weaker B (e 0.7); p = e_B x exp(-o^2 / sigma), o = 1 - d / R
zz = torch.logit(torch.tensor([[[0.9, 0.7]]])).requires_grad_(True)
pz = pp.clone().requires_grad_(True)
QMN.matrix_nms_logit(zz, pz, 1.75, 0.1)[0, 0, 1].backward()
dist = float(((pp[0, :, 0, 0] - pp[0, :, 0, 1]) ** 2).sum().sqrt())
o_ = 1.0 - dist / 1.75
d_ = float(np.exp(-o_ ** 2 / 0.1))
p_ = 0.7 * d_
check(abs(float(zz.grad[0, 0, 1]) - (1 - 0.7) / (1 - p_)) < 1e-4 and float(zz.grad[0, 0, 0]) == 0.0, (float(zz.grad[0, 0, 1]), (1 - 0.7) / (1 - p_)))        # d logit / d z_B = (1 - e) / (1 - e d); the order is a comparison: nothing for A
dpos = (2 * o_ / 0.1) / (1 - p_) / 1.75
check(abs(float(pz.grad[0, 1, 0, 1]) - dpos) < 1e-3 * dpos and abs(float(pz.grad[0, 1, 0, 0]) + dpos) < 1e-3 * dpos, (float(pz.grad[0, 1, 0, 1]), dpos))        # moving B away along x lifts e'; moving A away does the same

# ---- 14. (--real) the kill criterion of the card: the layer reproduces qgraph_readout.matrix_nms on 20 frames of B's full dump
if "--real" in sys.argv:
    dump_path = os.path.join(ROOT, "temp", "e276", "predfull_e273_p_s0.pkl")
    assert os.path.exists(dump_path), f"--real needs B's full dump {dump_path} (qgraph_run_eval.py score --run e273_p_s0 --seed 0 --exp e276 --readouts plain@0.5 --dump-full)"
    dump = pickle.load(open(dump_path, "rb"))
    H = W = 50
    cell = torch.stack(torch.meshgrid(-50 + 2 * (torch.arange(H).double() + 0.5), -50 + 2 * (torch.arange(W).double() + 0.5), indexing="ij"), 0)       # (2, H, W) centres in metres
    worst, kept_diff, n_kept, n_pairs = 0.0, 0, 0, 0
    for fr in dump[:: max(1, len(dump) // 20)][:20]:
        e = np.asarray(fr["e"], float)
        pos = np.asarray(fr["pos"], float)
        # the dump keeps the 512 best candidates (`idx` is only their rank), not their cells: a node lies within 1 m of its cell centre, so its cell follows from its position
        iy = np.clip(np.floor((pos[:, 0] + 50.0) / 2.0).astype(int), 0, H - 1)
        ix = np.clip(np.floor((pos[:, 1] + 50.0) / 2.0).astype(int), 0, W - 1)
        check(len(set(zip(iy.tolist(), ix.tolist()))) == len(iy), "two candidates fall into one cell: the grid cannot be rebuilt from the dump")
        lg_grid = torch.full((1, H, W), -20.0, dtype=torch.float64)
        ps_grid = cell.clone()[None]
        lg_grid[0, iy, ix] = torch.tensor(np.log(e / (1.0 - e)))
        ps_grid[0, 0, iy, ix], ps_grid[0, 1, iy, ix] = torch.tensor(pos[:, 0]), torch.tensor(pos[:, 1])
        mine = QMN.matrix_nms_logit(lg_grid.float(), ps_grid.float(), 1.75, 0.1).sigmoid()[0, iy, ix].numpy().astype(float)
        ref = QR.matrix_nms(pos, e, 1.75, thr=0.5, sigma=0.1)
        sel = e >= 0.5
        worst = max(worst, float(np.abs(np.where(ref > 0, ref, 0.0) - np.where(sel & (mine >= 0.5), mine, 0.0))[sel].max()))
        kept_diff += int(((ref >= 0.5) != (sel & (mine >= 0.5))).sum())
        n_kept += int((ref >= 0.5).sum())
        n_pairs += int((sel & (mine < e - 1e-6)).sum())
    check(worst < 2e-5 and kept_diff == 0 and n_pairs > 0, f"the layer does not reproduce matrix_nms on the real dump: max gap {worst:.2e}, kept sets differ at {kept_diff} nodes ({n_kept} kept, {n_pairs} damped)")
    print(f"real dump: 20 frames, {n_kept} kept nodes, {n_pairs} damped, max |layer - matrix_nms| {worst:.2e}, kept sets identical")
print(f"mnms layer tests ok ({n_ok} checks)")
