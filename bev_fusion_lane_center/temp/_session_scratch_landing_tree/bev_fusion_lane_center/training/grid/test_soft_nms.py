"""E277 tests (CPU, torch): the close-node suppression as a SOFT step of the forward path (`qnodes.soft_suppress_logit`, switched on by `QGraph(soft_nms=(radius, tau, passes))`).
The step must reproduce the hard rule of E275 (`evaluation/qgraph_readout.suppress_close_nodes`) in its limit, be differentiable in the intended places, and change nothing when it is off.
Run: python test_soft_nms.py   (`_py14.sh test_soft_nms.py`)"""
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in (HERE, os.path.join(HERE, "..", "..", "evaluation")):
    sys.path.insert(0, sub)
import qgraph as QG  # noqa: E402
import qnodes as QN  # noqa: E402
import qpredict as QP  # noqa: E402
import qgraph_readout as RO  # noqa: E402

R = 1.5
n_ok = 0


def check(cond, msg):
    global n_ok
    assert cond, msg
    n_ok += 1


def grid_frame(rng, H=12, W=12, p_active=0.35):
    """a random frame as the head produces it: cell centres on a 2 m grid, offsets in (-1, 1) m, a few active cells (e in 0.85 - 0.99, all different), the rest at 0.01"""
    cx = (np.arange(H) + 0.5) * 2.0
    cy = (np.arange(W) + 0.5) * 2.0
    pos = np.stack([np.repeat(cx[:, None], W, 1), np.repeat(cy[None, :], H, 0)], 0) + rng.uniform(-0.999, 0.999, (2, H, W))
    act = rng.random((H, W)) < p_active
    e = np.where(act, rng.permutation(np.linspace(0.85, 0.99, H * W)).reshape(H, W), 0.01)             # saturated probabilities: the damping of a node by a stronger one is then complete (a node of e 0.6 - 0.7 is damped to 0.2 only, so a chain of such nodes differs from the greedy rule: the step is soft)
    return pos, e, act


def hard_keep(pos, e):
    flat_pos = pos.reshape(2, -1).T
    return RO.suppress_close_nodes(flat_pos, e.reshape(-1), R, 0.5).reshape(e.shape)


def soft_keep(pos, e, tau=0.002, iters=8):
    lg = torch.logit(torch.tensor(e, dtype=torch.float32)).unsqueeze(0)
    out = QN.soft_suppress_logit(lg, torch.tensor(pos, dtype=torch.float32).unsqueeze(0), R, tau, iters)[0]
    return (out.sigmoid() >= 0.5).numpy(), out


def margin(pos, act):
    """the smallest | d - R | over the pairs of active cells that could interact (closer than R + 0.1): the soft radius needs a clear margin to be exact"""
    m = 9.9
    H, W = act.shape
    for i in range(H):
        for j in range(W):
            if not act[i, j]:
                continue
            for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
                a, b = i + di, j + dj
                if 0 <= a < H and 0 <= b < W and act[a, b]:
                    m = min(m, abs(float(np.linalg.norm(pos[:, i, j] - pos[:, a, b])) - R))
    return m


# 1. the hard limit: the same kept set as the greedy rule on random frames (frames with a pair at the radius within 0.02 m are skipped: the smooth radius is not exact there)
rng = np.random.default_rng(0)
used = bad = pairs = 0
for _ in range(300):
    pos, e, act = grid_frame(rng)
    if margin(pos, act) < 0.02:
        continue
    used += 1
    h = hard_keep(pos, e)
    s, _ = soft_keep(pos, e)
    pairs += int(h.sum() < act.sum())
    bad += int((h != s).any())
check(used > 150 and pairs > 100, ("the random frames must contain suppressions", used, pairs))
check(bad == 0, f"{bad} of {used} random frames: the soft step does not keep the set of the hard rule")

# 2. a chain A > B > C (B within R of A, C within R of B but not of A): the hard rule keeps A and C; one pass drops C, two passes keep it (the pass count is exact for chains up to its length)
pos_c = torch.tensor([[[[0.0, 0.0, 0.0]], [[0.0, 1.0, 2.0]]]]).reshape(1, 2, 1, 3)       # (1, 2, 1, 3): x = 0, y = 0 / 1 / 2 m, cells 0 / 1 / 2 of a 1 x 3 grid
lg_c = torch.logit(torch.tensor([[[0.9, 0.8, 0.7]]]))
one = QN.soft_suppress_logit(lg_c, pos_c, R, 0.002, 1)[0, 0].sigmoid()
two = QN.soft_suppress_logit(lg_c, pos_c, R, 0.002, 2)[0, 0].sigmoid()
hard = RO.suppress_close_nodes(np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]), np.array([0.9, 0.8, 0.7]), R, 0.5)
check(list(hard) == [True, False, True], hard)
check([bool(v >= 0.5) for v in one] == [True, False, False], one)                         # one pass: C is damped by B, which is itself damped (the pass count matters)
check([bool(v >= 0.5) for v in two] == [True, False, True], two)                          # two passes: the hard rule
# 3. ties: two cells with exactly equal e and close positions: the lower index stays (as in the hard rule), the other is damped
pos_t = torch.tensor([[[[0.0, 0.0]], [[0.0, 1.0]]]]).reshape(1, 2, 1, 2)
lg_t = torch.logit(torch.tensor([[[0.8, 0.8]]]))
tt = QN.soft_suppress_logit(lg_t, pos_t, R, 0.002, 3)[0, 0].sigmoid()
check(float(tt[0]) > 0.79 and float(tt[1]) < 0.2, tt)
# 4. order: flipping the grid (rows and columns reversed) flips the result; no information flows by position in the array except the tie-break (all e of the frame differ)
pos, e, act = grid_frame(np.random.default_rng(5))
s1, o1 = soft_keep(pos, e)
s2, o2 = soft_keep(pos[:, ::-1, ::-1].copy(), e[::-1, ::-1].copy())
check((s1 == s2[::-1, ::-1]).all(), "the result depends on the position of the cell in the array")
# 5. damping only: e' <= e everywhere; isolated nodes (pairs at least R + 0.8 apart) keep their probability
check(bool((o1.sigmoid() <= torch.tensor(e, dtype=torch.float32).unsqueeze(0) + 1e-6).all()), "e' above e")
pos_i = torch.tensor([[[[0.0, 0.0]], [[0.0, 3.0]]]]).reshape(1, 2, 1, 2)
iso = QN.soft_suppress_logit(torch.logit(torch.tensor([[[0.9, 0.7]]])), pos_i, R, 0.1, 3)[0, 0].sigmoid()
check(abs(float(iso[0]) - 0.9) < 0.01 and abs(float(iso[1]) - 0.7) < 0.01, iso)
# 6. gradient: the loss on the weaker node's e' reaches the stronger node's logit (raising e_A lowers e'_B), and both positions through the smooth radius; nothing is NaN
lg = torch.logit(torch.tensor([[[0.8, 0.6]]])).requires_grad_(True)
ps = torch.tensor([[[[0.0, 0.0]], [[0.0, 1.4]]]]).reshape(1, 2, 1, 2).requires_grad_(True)
out = QN.soft_suppress_logit(lg, ps, R, 0.1, 2)
out[0, 0, 1].backward()
check(torch.isfinite(lg.grad).all() and torch.isfinite(ps.grad).all(), "NaN gradient")
check(float(lg.grad[0, 0, 0]) < 0.0, lg.grad)                                           # the stronger node: more probability, a lower e' of the weaker one
check(float(lg.grad[0, 0, 1]) > 0.0, lg.grad)                                           # the weaker node itself: its own probability
check(float(ps.grad.abs().sum()) > 0.0, ps.grad)                                        # the positions: moving the nodes apart lifts e' (the radius is smooth)
# 7. borders and shapes: a node in the corner, no NaN, shape kept, float32 output for a float16 / bfloat16 input
lg_b = torch.logit(torch.full((2, 5, 5), 0.5))
out_b = QN.soft_suppress_logit(lg_b.bfloat16(), torch.zeros(2, 2, 5, 5).bfloat16(), R, 0.1, 3)
check(out_b.shape == (2, 5, 5) and out_b.dtype == torch.float32 and torch.isfinite(out_b).all(), (out_b.shape, out_b.dtype))
# 8. the switch in the model: default = no change (no e_raw, no new parameters); on: the damped logit is `no['e']`, the head's own is `no['e_raw']`, the weights are the same
torch.manual_seed(0)
plain = QG.QGraph(width=16, f05=16, d=32)
soft = QG.QGraph(width=16, f05=16, d=32, soft_nms=(1.5, 0.1, 3))
soft.load_state_dict(plain.state_dict())
check(set(plain.state_dict()) == set(soft.state_dict()) and plain.soft_nms is None and soft.soft_nms == (1.5, 0.1, 3.0), "the switch must not add parameters")
x = torch.randn(1, 39, 200, 200)
with torch.no_grad():
    o_p = plain({"x": x}, upto="nodes", gt=False)["no"]
    o_s = soft({"x": x}, upto="nodes", gt=False)["no"]
check("e_raw" not in o_p, "the default model must be the E273 model")
check(torch.equal(o_s["e_raw"], o_p["e"]), "the raw logit of the soft model is the plain model's logit")
check(torch.allclose(o_s["e"], QN.soft_suppress_logit(o_p["e"], o_p["pos"], 1.5, 0.1, 3)), "no['e'] must be the damped logit")
check(float(o_s["e"].sigmoid().sum()) < float(o_p["e"].sigmoid().sum()) + 1e-3, "the step only damps")
# 9. the checkpoint carries the switch: load_model builds the soft model from args['soft_nms'] and the plain one when there is none (a checkpoint of E273 / E276)
with tempfile.TemporaryDirectory() as d:
    m0 = QG.QGraph(width=48)
    torch.save(dict(state=m0.state_dict(), args=dict(width=48, soft_nms=[1.5, 0.1, 3.0])), os.path.join(d, "model.pt"))
    check(QP.load_model(d, "cpu").soft_nms == (1.5, 0.1, 3.0), "load_model must restore the soft step")
    torch.save(dict(state=m0.state_dict(), args=dict(width=48)), os.path.join(d, "model.pt"))
    check(QP.load_model(d, "cpu").soft_nms is None, "an older checkpoint stays the plain head")
    # the checkpoint of a RUNNING arm (ckpt.pt + config.json of the trainer) is a soft head too, not a plain one (critic pass 2026-11-08)
    os.remove(os.path.join(d, "model.pt"))
    torch.save(dict(model=m0.state_dict()), os.path.join(d, "ckpt.pt"))
    check(QP.load_model(d, "cpu").soft_nms is None, "no config.json: plain")
    import json
    json.dump(dict(args=dict(width=48, soft_nms=[1.5, 0.1, 3.0])), open(os.path.join(d, "config.json"), "w"))
    check(QP.load_model(d, "cpu").soft_nms == (1.5, 0.1, 3.0), "load_model of a running arm must restore the soft step from config.json")


# 10. the model-level path with a hand-built node head: the chain of test 2 inside `QGraph.forward`: the pass count and the radius of the model setting are the ones used (a model that hard-wires one pass, or
# a wrong radius, fails here although the function itself is right)
class Stub(torch.nn.Module):
    def __init__(self, e_logit, pos):
        super().__init__()
        self.e, self.pos = e_logit, pos

    def forward(self, *a):
        return dict(e=self.e.clone(), pos=self.pos.clone())


def stub_model(setting):
    m = QG.QGraph(width=16, f05=16, d=32, soft_nms=setting)
    m.nodes = Stub(lg_c, pos_c.reshape(1, 2, 1, 3))
    return m


with torch.no_grad():
    x_ = torch.randn(1, 39, 200, 200)
    e_two = stub_model((1.5, 0.002, 2))({"x": x_}, upto="nodes", gt=False)["no"]
    e_one = stub_model((1.5, 0.002, 1))({"x": x_}, upto="nodes", gt=False)["no"]
    e_small_radius = stub_model((0.5, 0.002, 2))({"x": x_}, upto="nodes", gt=False)["no"]
check([bool(v >= 0.5) for v in e_two["e"].sigmoid()[0, 0]] == [True, False, True], e_two["e"].sigmoid())
check([bool(v >= 0.5) for v in e_one["e"].sigmoid()[0, 0]] == [True, False, False], e_one["e"].sigmoid())
check(torch.allclose(e_small_radius["e"], lg_c, atol=1e-4) and torch.equal(e_two["e_raw"], lg_c), "radius 0.5 m: the nodes 1 m apart are not damped; e_raw is the head's logit")
# the phase gate reads the same damped logit (critic pass 2026-11-08: `_node_stats` used to call the raw node head)
sys.path.insert(0, os.path.join(HERE, "..", "train"))
import qgates as QGt  # noqa: E402

with torch.no_grad():
    _, no_g = QGt._node_stats(stub_model((1.5, 0.002, 2)), {"x": x_}, "cpu")
check(torch.allclose(no_g["e"], e_two["e"]) and not torch.allclose(no_g["e"], lg_c), "the gate must read the damped logit")
# 11. a batch is a stack of independent frames; the real grid (50 x 50); the production setting (tau 0.1, 3 passes) against the hard rule on saturated probabilities stays close (about 1 %: the step is soft)
rng = np.random.default_rng(3)
frames_ = [grid_frame(rng, 50, 50, 0.35) for _ in range(3)]
lg3 = torch.logit(torch.tensor(np.stack([f[1] for f in frames_]), dtype=torch.float32))
ps3 = torch.tensor(np.stack([f[0] for f in frames_]), dtype=torch.float32)
out3 = QN.soft_suppress_logit(lg3, ps3, R, 0.1, 3)
for k in range(3):
    check(torch.equal(out3[k], QN.soft_suppress_logit(lg3[k:k + 1], ps3[k:k + 1], R, 0.1, 3)[0]), "a batch must equal its frames one by one")
    h = hard_keep(frames_[k][0], frames_[k][1])
    s = (out3[k].sigmoid() >= 0.5).numpy()
    gap = float((h != s).sum()) / max(int(h.sum()), 1)
    check(gap < 0.03, f"production setting: {gap:.3f} of the hard-kept cells differ")
# 12. no dead gradient at the extremes (critic pass: a clamp at 1e-6 cut it beyond |logit| 13.8); an isolated node returns its own logit
for v in (-30.0, -14.5, 14.5, 30.0):
    l1 = torch.tensor([[[v]]], requires_grad=True)
    o1 = QN.soft_suppress_logit(l1, torch.zeros(1, 2, 1, 1), R, 0.1, 3)
    o1.sum().backward()
    check(abs(float(o1) - v) < 1e-3 and abs(float(l1.grad) - 1.0) < 1e-3, (v, float(o1), float(l1.grad)))
# 13. arguments: tau = 0, no pass, a radius above 2 m (pairs two cells apart would be missed) are refused, in the function and in the model
for bad in ((1.5, 0.0, 3), (1.5, 0.1, 0), (2.5, 0.1, 3), (1.5, 0.1, 2.5)):
    try:
        QG.QGraph(width=16, f05=16, d=32, soft_nms=bad)
        check(False, f"accepted {bad}")
    except AssertionError:
        check(True, "")
    try:
        QN.soft_suppress_logit(lg_c, pos_c, *bad)
        check(False, f"function accepted {bad}")
    except AssertionError:
        check(True, "")
print(f"soft_nms tests ok ({n_ok} checks; {used} random frames against the hard rule, {pairs} with suppressions)")
