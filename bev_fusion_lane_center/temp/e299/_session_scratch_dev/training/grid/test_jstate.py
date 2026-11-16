"""E299 tests (CPU, torch): the recursive output state of the junction pair head (`jstate.py`, `JEnds(state=True)`) and the loader hook for the previous keyframe (`lane_gt_data.QData(prev=True)`).
 - shapes and the state dict: the flag off has the keys of E312 / E320 / E324, the flag on adds only `state_attn.*` (created last: the other tensors are the same for the same seed);
 - the zero start: with the gate at 0 the flag on is the flag off, bit for bit (outputs and every gradient), also with a state; a nonzero gate makes the output depend on the state (and a layer that ignores it fails);
 - the warp: a known pose in both directions (`lane_gt_data.rel_pose` + `jstate.warp`), against `evaluation/jtime.warp` (step 0), and through the layer (the matched mass of a static scene);
 - the attention: the counterpart gets the mass, an entry never attends an exit, a far / turned / invalid state entry gets none; the row summary by hand;
 - a scene start (`has` False) skips the layer; NO gradient reaches the state; the gate gets a gradient at 0, the layer weights after it moved (20 steps);
 - the loader: the previous row is fetched by ROW (a strided frame list), x_prev is the features of that row, the poses are the CAM_FRONT lane frame of the infos (VAL) and of one scene order (TRAIN: 8202 of 8414 rows with a previous keyframe);
 - mutations (each must FAIL its check): the state of the NEXT row, the warp in the wrong direction, the gate not zero at the start, a state that is not detached, a layer that ignores the state.
Run: python test_jstate.py   (`_py14.sh grid/test_jstate.py`; needs the cached features, the stored lane GT and the CRN infos of VAL for the last part)"""
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(HERE, "..", "data"), HERE):
    sys.path.insert(0, os.path.normpath(p))
for sub in ("training/grid", "training/data", "training/data/prep", "evaluation", "tools/viz", ".."):
    sys.path.append(os.path.normpath(os.path.join(ROOT, sub)))
import jends as JE  # noqa: E402
import jstate as JS  # noqa: E402
import lane_gt_data as LD  # noqa: E402

torch.set_num_threads(2)
K = JE.K


def must_fail(check, *args):
    """a mutation must make its check fail"""
    try:
        check(*args)
    except AssertionError:
        return
    raise RuntimeError(f"the mutation survived {check.__name__}")


def mk_in(seed=0, B=2):
    """a synthetic frame as in the self-test of jends.py (random heat: many real peaks) + the GT ends"""
    g = torch.Generator().manual_seed(seed)
    jout = dict(j1=torch.randn(B, 8, 100, 100, generator=g) * 0.1 - 2.0, j05=torch.randn(B, 2, 200, 200, generator=g) - 2.0, f1=torch.randn(B, 64, 100, 100, generator=g))
    jout["j1"][:, 4], jout["j1"][:, 6], jout["j1"][:, 5], jout["j1"][:, 7] = 0.0, 0.0, 5.0, 5.0
    P = dict(centre1=torch.randn(B, 12, 100, 100, generator=g))
    ent = torch.tensor([[[-1.75, -8.0, 0.0, 1.0], [1.75, -8.0, 0.0, 1.0]]] * B)
    ext = torch.tensor([[[-1.75, 8.0, 0.0, 1.0], [1.75, 8.0, 0.0, 1.0]]] * B)
    gt = dict(ent=ent, ext=ext, ent_m=torch.ones(B, 2, dtype=torch.bool), ext_m=torch.ones(B, 2, dtype=torch.bool))
    return jout, P, gt


def mk_model(state, seed=0, **kw):
    torch.manual_seed(seed)
    return JE.JEnds(d=64, order=True, evidence=True, scales=True, state=state, **kw)


def mk_state(seed, B=2):
    """the output state of a teacher on another synthetic frame"""
    jout, P, _ = mk_in(seed, B)
    with torch.no_grad():
        return JS.build_state(mk_model(False, seed=100 + seed)(jout, P, gt=None))


def same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a if torch.is_tensor(a[k]))


EYE = (torch.eye(2).expand(2, 2, 2), torch.zeros(2, 2))


# ---------------------------------------------------------------------------------------------------------------------------------- the warp
def _check_warp(warp):
    """the previous frame faces +y at the origin; the ego then drove 10 m ahead and turned 90 degrees left: a point 20 m ahead of the old frame lies 10 m to the RIGHT of the new one, the old heading is 'right'"""
    Rc, tc = np.array([[0.0, -1.0], [1.0, 0.0]]), np.array([0.0, 10.0])
    R, t = LD.rel_pose(Rc, tc, np.eye(2), np.zeros(2))
    assert np.allclose(R, [[0, 1], [-1, 0]], atol=1e-6) and np.allclose(t, [-10, 0], atol=1e-6), (R, t)
    pos, d = warp(torch.tensor([[[0.0, 20.0], [3.0, 0.0]]]), torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]), torch.from_numpy(R)[None], torch.from_numpy(t)[None])
    assert torch.allclose(pos, torch.tensor([[[10.0, 0.0], [-10.0, -3.0]]]), atol=1e-5), pos
    assert torch.allclose(d, torch.tensor([[[1.0, 0.0], [0.0, -1.0]]]), atol=1e-5), d


def test_warp():
    _check_warp(JS.warp)
    must_fail(_check_warp, lambda pos, d, R, t: (pos @ R + t[:, None, :], d @ R))                      # R instead of R^T: the wrong direction
    # the round trip and the step-0 reference (`evaluation/jtime.warp`, validated on VAL): random poses, ends (x, y, tx, ty)
    import jtime as JT
    rng = np.random.RandomState(3)
    rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])  # noqa: E731
    Rp, tp, Rc, tc = rot(rng.uniform(-3, 3)), rng.uniform(-50, 50, 2), rot(rng.uniform(-3, 3)), rng.uniform(-50, 50, 2)
    E = np.concatenate([rng.uniform(-40, 40, (6, 2)), np.eye(2)[rng.randint(0, 2, 6)]], 1)
    ref = JT.warp(E, (Rp, tp), (Rc, tc))
    R, t = LD.rel_pose(Rc, tc, Rp, tp)
    pos, d = JS.warp(torch.from_numpy(E[None, :, :2]).float(), torch.from_numpy(E[None, :, 2:]).float(), torch.from_numpy(R)[None], torch.from_numpy(t)[None])
    assert np.allclose(torch.cat([pos, d], -1)[0].numpy(), ref, atol=1e-4), "the warp must equal evaluation/jtime.warp (step 0)"
    R2, t2 = LD.rel_pose(Rp, tp, Rc, tc)
    back, _ = JS.warp(pos, d, torch.from_numpy(R2)[None], torch.from_numpy(t2)[None])
    assert np.allclose(back[0].numpy(), E[:, :2], atol=1e-3), "there and back must be the identity"
    print("warp: known pose, both directions, = jtime.warp, round trip; a wrong direction fails: ok")


# ------------------------------------------------------------------------------------------------------------------------------ the layer
def toy_state(K_=4, shift=(0.0, 0.0), turn=False, valid=None):
    """K_ entries on a line (12 m apart), K_ exits 8 m beside them; the state = the same keypoints (+ a shift), pair logits: entry i -> exit i"""
    pos = torch.stack([torch.tensor([12.0 * i, 0.0]) for i in range(K_)] + [torch.tensor([12.0 * i, 8.0]) for i in range(K_)])[None]
    D = torch.tensor([0.0, 1.0]).expand(1, 2 * K_, 2).clone()
    z = torch.full((1, K_, K_), -4.0)
    z[0].fill_diagonal_(3.0)
    st = dict(pos=pos + torch.tensor(shift), dir=-D if turn else D, p=torch.full((1, 2 * K_), 0.9), valid=torch.ones(1, 2 * K_, dtype=torch.bool) if valid is None else valid, logit=z,
              allowed=torch.ones(1, K_, K_, dtype=torch.bool), has=torch.ones(1, dtype=torch.bool))
    return pos, D, st


def test_attention():
    torch.manual_seed(0)
    layer = JS.StateAttn(d=64)
    t = torch.randn(1, 8, 64)
    pos, D, st = toy_state()
    sf = JS.state_features(st)
    assert sf.shape == (1, 8, JS.NF)
    w = layer.attend(t, pos, D, st, sf)
    assert w.shape == (1, 8, 8) and (w.argmax(-1)[0] == torch.arange(8)).all() and float(w.diagonal(dim1=1, dim2=2).min()) > 0.4, w[0]
    assert float(w[:, :4, 4:].abs().max()) == 0.0 and float(w[:, 4:, :4].abs().max()) == 0.0, "an entry never attends an exit"
    for kw, what in ((dict(shift=(0.0, 30.0)), "30 m away"), (dict(turn=True), "turned by 180 degrees")):
        w2 = layer.attend(t, pos, D, toy_state(**kw)[2], sf)
        assert float(w2.sum(-1).max()) < 0.02, (what, w2.sum(-1))                                        # no counterpart: the mass goes to the null slot
    vm = torch.ones(1, 8, dtype=torch.bool)
    vm[0, 2] = False
    w3 = layer.attend(t, pos, D, toy_state(valid=vm)[2], sf)
    assert float(w3[:, :, 2].abs().max()) == 0.0, "an invalid state entry gets no mass"
    # the row summary by hand: entry 0 has two allowed partners (exit 1 at logit 2, exit 3 at logit -2); the exit's rows are the columns
    st2 = dict(st, logit=torch.zeros(1, 4, 4), allowed=torch.zeros(1, 4, 4, dtype=torch.bool))
    st2["logit"][0, 0, 1], st2["logit"][0, 0, 3] = 2.0, -2.0
    st2["allowed"][0, 0, 1], st2["allowed"][0, 0, 3] = True, True
    f = JS.state_features(st2)[0]
    a, b = math.exp(2.0) / (math.exp(2.0) + math.exp(-2.0)), math.exp(-2.0) / (math.exp(2.0) + math.exp(-2.0))
    exp_rel = (a * st["pos"][0, 5] + b * st["pos"][0, 7] - st["pos"][0, 0]) / 20.0
    assert abs(float(f[0, 6]) - 2 / 8) < 1e-6 and abs(float(f[0, 7]) - 1 / (1 + math.exp(-2.0))) < 1e-6 and torch.allclose(f[0, 8:10], exp_rel, atol=1e-5), f[0]
    assert abs(float(f[1, 6]) + 1.0) < 1e-6 and float(f[1, 7]) == 0.0 and float(f[1, 8:].abs().max()) == 0.0, "an entry without an allowed partner: -8 / 8, probability 0, zeros"
    assert abs(float(f[5, 6]) - 2 / 8) < 1e-6, "the exit 1 sees the same pair from its column"
    # the same scene seen from a moved ego: the state warped by the pose matches the current keypoints (matched mass), the wrong direction does not
    Rc, tc = np.array([[0.0, -1.0], [1.0, 0.0]]), np.array([3.0, 10.0])
    R, t_ = LD.rel_pose(Rc, tc, np.eye(2), np.zeros(2))
    prev = toy_state()[2]
    cur_pos, cur_dir = JS.warp(prev["pos"], prev["dir"], torch.from_numpy(R)[None], torch.from_numpy(t_)[None])

    def mass(warp_state):
        s = warp_state(prev, torch.from_numpy(R)[None], torch.from_numpy(t_)[None])
        return float(layer.attend(t, cur_pos, cur_dir, s, JS.state_features(s)).sum(-1).min())

    assert mass(JS.warp_state) > 0.4 and mass(lambda s, R, t: dict(s, pos=s["pos"] @ R + t[:, None, :], dir=s["dir"] @ R)) < 0.02
    print("attention: the counterpart gets the mass, kinds apart, far / turned / invalid get none; row summary by hand; the warped static scene matches: ok")


def test_zero_start_and_state():
    jout, P, gt = mk_in(0)
    off, on = mk_model(False), mk_model(True)
    assert set(on.state_dict()) - set(off.state_dict()) and all(k.startswith("state_attn.") for k in set(on.state_dict()) - set(off.state_dict())) and set(off.state_dict()) <= set(on.state_dict())
    assert all(torch.equal(off.state_dict()[k], on.state_dict()[k]) for k in off.state_dict()), "the same seed: the tensors of the flag off are those of the flag on"
    assert float(on.state_attn.gate.abs().sum()) == 0.0
    print("state dict: the flag off = E312 keys, the flag on adds only state_attn.* (%d parameters), the other tensors equal" % sum(p.numel() for p in on.state_attn.parameters()))

    def grads(m, st, dev=None):
        jr = dict(jout, j1=jout["j1"].clone().requires_grad_(True))
        m.zero_grad()
        out = m(jr, P, gt=gt, state=st) if st is not None else m(jr, P, gt=gt)
        l, _ = JE.jends_loss(out, dict(ent=gt["ent"], ext=gt["ext"], ent_m=gt["ent_m"], ext_m=gt["ext_m"], pairs=torch.tensor([[[0, 0], [1, 1]]] * 2), pscale=torch.full((2, 2, 2), 0.3)))
        l.backward()
        return out, {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None and not n.startswith("state_attn")}, jr["j1"].grad.clone()

    st = JS.warp_state(mk_state(1), *EYE)
    o0, g0, j0 = grads(off, None)
    o1, g1, j1 = grads(on, st)
    assert same(o0, o1) and g0.keys() == g1.keys() and all(torch.equal(g0[k], g1[k]) for k in g0) and torch.equal(j0, j1), "gate 0 + a state == the flag off (outputs, every gradient, the heat)"
    o2, _, _ = grads(on, None)
    assert same(o0, o2), "the flag on without a state == the flag off"
    print("zero start: the gate at 0 + a state = the flag off, bit for bit (outputs, all gradients, the heat gradient): ok")
    try:
        off(jout, P, gt=gt, state=st)
        raise RuntimeError("a state for a head without the layer must raise")
    except AssertionError:
        pass


def _check_uses_state(layer_forward):
    """a NONZERO gate: the pair logits depend on the state (another state, no state) and a scene start (has False) is the flag off"""
    jout, P, gt = mk_in(0)
    m = mk_model(True)
    m.state_attn.gate.data.fill_(0.5)
    if layer_forward is not None:
        m.state_attn.forward = layer_forward(m.state_attn)
    sa, sb = JS.warp_state(mk_state(1), *EYE), JS.warp_state(mk_state(2), *EYE)
    with torch.no_grad():
        la, lb, ln = m(jout, P, gt=gt, state=sa)["logit"], m(jout, P, gt=gt, state=sb)["logit"], m(jout, P, gt=gt)["logit"]
        lstart = m(jout, P, gt=gt, state=dict(sa, has=torch.zeros(2, dtype=torch.bool)))["logit"]
    assert float((la - lb).abs().max()) > 1e-3, "the output must depend on WHICH state"
    assert float((la - ln).abs().max()) > 1e-3, "the output must depend on the state"
    assert torch.equal(lstart, ln), "a scene start skips the layer"


def test_uses_state():
    _check_uses_state(None)
    ignore = lambda layer: (lambda t, pos, D, state: layer.__class__.forward(layer, t, pos, D, dict(state, pos=state["pos"] * 0 + 1.0, dir=state["dir"] * 0 + 1.0, p=state["p"] * 0, logit=state["logit"] * 0)))  # noqa: E731
    must_fail(_check_uses_state, ignore)                                                              # a layer that ignores the state fails
    print("a nonzero gate makes the logits depend on the state; a scene start is the flag off; a layer that ignores the state fails: ok")


def _check_zero_start(init_gate):
    jout, P, gt = mk_in(0)
    m = mk_model(True)
    if init_gate:
        m.state_attn.gate.data.fill_(init_gate)
    st = JS.warp_state(mk_state(1), *EYE)
    with torch.no_grad():
        assert torch.equal(m(jout, P, gt=gt, state=st)["logit"], mk_model(False)(jout, P, gt=gt)["logit"]), "the model with the layer must start as the flag off"


def test_gate_not_zero_mutation():
    _check_zero_start(0.0)
    must_fail(_check_zero_start, 0.1)
    print("a gate that does not start at 0 fails the zero-start check: ok")


def _check_detached(build):
    """the student's loss must not reach the teacher: neither its parameters nor its input"""
    jout, P, gt = mk_in(0)
    teacher, student = mk_model(False, seed=5), mk_model(True)
    student.state_attn.gate.data.fill_(0.5)
    jt = dict(jout, j1=jout["j1"].clone().requires_grad_(True))
    out_t = teacher(jt, P, gt=None)
    st = JS.warp_state(build(out_t), *EYE)
    student(jout, P, gt=gt, state=st)["logit"].sum().backward()
    assert jt["j1"].grad is None and all(p.grad is None for p in teacher.parameters()), "a gradient reached the state's network"
    assert any(p.grad is not None for p in student.state_attn.parameters())


def test_no_gradient_into_state():
    _check_detached(JS.build_state)
    nodetach = lambda out: dict(pos=out["kp_pos"], dir=out["kp_dir"], p=out["kp_p"], valid=out["valid"].flatten(1), logit=out["logit"].float(), allowed=out["allowed"], has=torch.ones(out["kp_pos"].shape[0], dtype=torch.bool))  # noqa: E731
    must_fail(_check_detached, nodetach)
    print("no gradient reaches the state (the teacher's parameters and heat); a state that is not detached fails: ok")


def test_gradient_reaches_layer():
    """the card's kill: the gate has a gradient at 0, the layer weights only once it moved (after a few steps they all have one)"""
    jout, P, gt = mk_in(0)
    m = mk_model(True)
    st = JS.warp_state(mk_state(1), *EYE)
    batch = dict(ent=gt["ent"], ext=gt["ext"], ent_m=gt["ent_m"], ext_m=gt["ext_m"], pairs=torch.tensor([[[0, 0], [1, 1]]] * 2), pscale=torch.full((2, 2, 2), 0.3))
    m(jout, P, gt=gt, state=st)["logit"].sum().backward()
    assert float(m.state_attn.gate.grad.abs().sum()) > 0 and float(sum(p.grad.abs().sum() for n, p in m.state_attn.named_parameters() if n != "gate")) == 0.0, "at gate 0: only the gate has a gradient"
    opt = torch.optim.Adam(m.state_attn.parameters(), lr=3e-2)
    for _ in range(20):
        opt.zero_grad()
        loss, _ = JE.jends_loss(m(jout, P, gt=gt, state=st), batch)
        loss.backward()
        opt.step()
    assert float(m.state_attn.gate.abs()) > 0.05 and all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.state_attn.parameters()), [(n, p.grad is None) for n, p in m.state_attn.named_parameters()]
    print("gradient: the gate at 0 alone, then (20 steps) the gate moved to %.2f and every layer weight has one: ok" % float(m.state_attn.gate))


# ------------------------------------------------------------------------------------------------------------------------------ the loader
def _check_prev_rows(link):
    """three scenes in time order (a scene start has no previous row): the previous row is the one before it in time, of the same scene, never a later one"""
    prev = np.array([-1, 0, 1, -1, 3, 4, -1, 6])
    scene, ts = np.array([0, 0, 0, 1, 1, 1, 2, 2]), np.arange(8) * 0.5
    pose = dict(prev=prev, R=np.tile(np.eye(2), (8, 1, 1)), t=np.zeros((8, 2)))
    for r in range(8):
        p, R, t = link(pose, r)
        assert p == prev[r], (r, p)
        if p >= 0:
            assert scene[p] == scene[r] and ts[p] < ts[r], "the previous row must be earlier and of the same scene"
        else:
            assert np.allclose(R, np.eye(2)) and np.allclose(t, 0), "a scene start: no previous row, identity pose"


def test_prev_rows():
    _check_prev_rows(LD.prev_link)
    nxt = lambda pose, r: (r + 1 if r + 1 < len(pose["prev"]) else -1, np.eye(2), np.zeros(2))                 # the NEXT row instead of the previous
    must_fail(_check_prev_rows, nxt)
    print("prev_link: the previous row of the scene by row index, a scene start has none; the next row fails: ok")


def test_loader_real():
    import jtime as JT
    import qgraph_view as QV
    HOME = os.path.expanduser("~")
    # TRAIN: the sequence in poses_train.npz (the card: 8202 of 8414 rows have a previous keyframe)
    for part, n_prev in (("train", 8202), ("val", None)):
        z = np.load(f"{HOME}/exp_gcar_b123/poses_{part}.npz")
        has = z["prev"] >= 0
        if n_prev:
            assert int(has.sum()) == n_prev and len(has) == 8414, (int(has.sum()), len(has))
        p = z["prev"][has]
        assert (z["scene"][p] == z["scene"][has]).all() and (z["ts"][p] < z["ts"][has]).all() and ((z["dt"][has] > 0.3) & (z["dt"][has] < 0.7)).all(), part
    # VAL: poses_val.npz = the CAM_FRONT lane frame of the infos (what `jtime` used in step 0)
    ds = LD.QData("val_clean", pixels=False, prev=True, ends=True, map_mode="exact")
    infos = QV._infos()
    z = np.load(f"{HOME}/exp_gcar_b123/poses_val.npz")
    for k in np.linspace(0, len(ds.bd.rows) - 1, 25).astype(int):
        row = int(ds.bd.rows[k])
        A, c = JT.lane_frame(infos[ds.tokens[row]])
        assert np.allclose(A, z["R"][row], atol=1e-6) and np.allclose(c, z["t"][row], atol=1e-6), "poses_val.npz is not the lane frame of the infos"
    # the loader: a strided frame list (the previous row of an item is mostly NOT in it), compared with the item of the previous row of the full list
    sub = LD.QData("val_clean", ks=np.arange(0, len(ds.bd.rows), 3), pixels=False, prev=True, map_mode="exact")
    n_chk = 0
    for i in range(1, 12):
        k = sub.ks[i]
        row = int(sub.bd.rows[k])
        it = sub[i]
        prow = int(z["prev"][row])
        if prow < 0:
            assert not bool(it["has_prev"]) and float(it["x_prev"].float().abs().sum()) == 0.0 and torch.equal(it["R_rel"], torch.eye(2)) and float(it["t_rel"].abs().sum()) == 0.0
            continue
        assert bool(it["has_prev"]) and prow == row - 1 and ds.tokens[prow] != ds.tokens[row], (row, prow)
        full = ds[ds.ks.index(ds.bd.row2k[prow])]
        assert torch.equal(it["x_prev"], full["x"]), "x_prev must be the features of the previous ROW"
        Rr, tr = LD.rel_pose(z["R"][row], z["t"][row], z["R"][prow], z["t"][prow])
        assert torch.equal(it["R_rel"], torch.from_numpy(Rr)) and torch.equal(it["t_rel"], torch.from_numpy(tr))
        n_chk += 1
    assert n_chk >= 5, n_chk
    b = LD.collate([sub[i] for i in range(1, 4)])
    assert b["x_prev"].shape == (3, 39, 200, 200) and b["has_prev"].shape == (3,) and b["R_rel"].shape == (3, 2, 2) and b["t_rel"].shape == (3, 2)
    off = LD.QData("val_clean", ks=np.arange(0, len(ds.bd.rows), 3), pixels=False, map_mode="exact")
    assert not any(k in off[1] for k in ("x_prev", "has_prev", "R_rel", "t_rel")) and torch.equal(off[1]["x"], sub[1]["x"]), "the flag off: the batch carries nothing new"
    # the poses are the lane frame of the GT: the GT ends of the previous frame, warped by the relative pose, land on ends of the current frame (the card's 'the warp maps >= 95 % of the GT ends'; here 1 m / 30 degrees)
    pairs = []
    for i in range(1, 80):
        row = int(ds.bd.rows[ds.ks[i]])
        prow = int(z["prev"][row])
        if prow >= 0 and prow in ds.bd.row2k:
            pairs.append((row, prow, ds[ds.ks.index(ds.bd.row2k[prow])], ds[i]))
    assert len(pairs) >= 40, len(pairs)

    def hit_rate(rel):
        hit = tot = 0
        for row, prow, prv, cur in pairs:
            R, t = rel(z["R"][row], z["t"][row], z["R"][prow], z["t"][prow])
            for key in ("ent", "ext"):
                P_, C_ = prv[key].numpy(), cur[key].numpy()
                if not len(P_):
                    continue
                pos, d = P_[:, :2] @ R.T + t, P_[:, 2:] @ R.T
                for p_, d_ in zip(pos[(np.abs(pos[:, 0]) < 35) & (pos[:, 1] > -10) & (pos[:, 1] < 35)], d[(np.abs(pos[:, 0]) < 35) & (pos[:, 1] > -10) & (pos[:, 1] < 35)]):
                    tot += 1
                    hit += bool(len(C_) and ((np.linalg.norm(C_[:, :2] - p_, axis=1) < 1.0) & (C_[:, 2:] @ d_ > math.cos(math.radians(30)))).any())
        return hit / max(tot, 1), tot

    def _check_gt_warp(rel):
        assert hit_rate(rel)[0] >= 0.95, "the warped GT ends of the previous frame must land on the GT ends of the current one"

    _check_gt_warp(LD.rel_pose)
    must_fail(_check_gt_warp, lambda Rc, tc, Rp, tp: LD.rel_pose(Rp, tp, Rc, tc))                       # the poses of the two frames swapped: the wrong direction
    rate, n_end = hit_rate(LD.rel_pose)
    print("loader: previous row by ROW in a strided list, x_prev = the features of that row, poses = the CAM_FRONT lane frame (25 VAL rows against the infos), the flag off carries nothing;", n_chk, "items checked; the warped GT ends of the previous frame land on the current frame's: %.3f of %d" % (rate, n_end))


if __name__ == "__main__":
    test_warp()
    test_attention()
    test_zero_start_and_state()
    test_uses_state()
    test_gate_not_zero_mutation()
    test_no_gradient_into_state()
    test_gradient_reaches_layer()
    test_prev_rows()
    test_loader_real()
    print("test_jstate ok")
