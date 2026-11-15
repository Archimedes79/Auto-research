"""E354 tests of the recursive feature state (`jrecur.py`, `data/lane_seq_data.py`, `train/jrecur_train.py`, `evaluation/jrecur_eval.py`), CPU only (the GPU is hidden), 2 threads, the stored E341 model of seed 0 and real VAL frames; about 2 minutes.

    wsl bash tools/env/_py14.sh training/grid/test_jrecur.py [name ...]          # all tests, or the named ones (with or without the `test_` prefix)

Known answers: the warp moves a marked cell by one cell under a 2 m translation in x and in y and to the rotated cell under 90 degrees, the inverse pose fails each of them; a real pose pair moves a static world point where
`evaluation/jtime.warp` puts it; `rel_pose` against the global coordinates; the zero gate = the model of `jends_train.py`'s own builder with the E341 weights bit for bit (no state and a random state), the same initial tensors for a seed,
only the two fusion tensors missing (74368 parameters); the loss of frame 3 reaches the first maps-head features of frame 1 with the gate open and a state, not without a state and not through a zero gate; sequences on a toy (a
missing row, a scene start) and on the TRAIN poses; the stream order and the state plan of the four read-outs on a toy and on the VAL rows; the dump: zero gate = the stored E341 dump, off = cold, trunc3, shuffle by hand.
"""
import os
import pickle
import sys
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                      # this test never touches the GPU
os.environ["BFLC_SIG"] = "1.0"
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
for sub_ in (".", "..", "../data", "../data/prep", "../train", "../../evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub_)))
import jends_eval as JEV  # noqa: E402
import jends_train as JT  # noqa: E402
import jrecur as JR  # noqa: E402
import jrecur_eval as JRE  # noqa: E402
import jrecur_train as JRT  # noqa: E402
import jtime as JTM  # noqa: E402
import lane_gt_data as LD  # noqa: E402
import lane_seq_data as LS  # noqa: E402

torch.set_num_threads(2)
RUN = "e341_p_s0"
POSES = os.path.expanduser("~/exp_gcar_b123/poses_%s.npz")
_FX = {}


def fx():
    """the shared fixture: E341 seed 0 through the frozen loader and through `jrecur.load_model`, the VAL data set with the junction targets, the VAL poses, the first 4 keyframes of two scenes in stream order"""
    if not _FX:
        LD.QData.DEFAULT_JUNCTION = True
        frozen, _ = JEV.load_model(RUN, "cpu")
        new, args = JR.load_model(RUN, "cpu", strict=False)
        ds = LD.QData("val_clean", map_mode="error", pixels=False)
        pose = dict(np.load(POSES % "val"))
        order = LS.stream_order(pose, [int(ds.bd.rows[k]) for k in ds.ks])
        starts = [i for i, (_, r) in enumerate(order) if r]
        short = [order[i] for s in starts[:2] for i in range(s, s + 4)]
        _FX.update(frozen=frozen, new=new, args=args, ds=ds, pose=pose, order=order, short=short, i_of_row={int(ds.bd.rows[k]): i for i, k in enumerate(ds.ks)})
    return _FX


def must_fail(fn, *a, **k):
    """a mutation check: fn must fail its assertion"""
    try:
        fn(*a, **k)
    except AssertionError:
        return
    raise RuntimeError(f"{fn.__name__} passed with the mutated input")


def _open_gate(m, g=0.5):
    torch.manual_seed(1)
    with torch.no_grad():
        m.jmodel.jm.fuse.g.fill_(g)


def _inverse(R, t):
    return R.transpose(1, 2), -torch.einsum("bji,bj->bi", R, t)


def _check_marked(pose_of, R, t, cell):
    s = torch.zeros(1, 1, 50, 50)
    s[0, 0, 20, 30] = 1.0                                                    # the cell at x = -9, y = 11
    w, valid = JR.warp_state(s, *pose_of(R, t))
    assert abs(float(w[0, 0, cell[0], cell[1]]) - 1.0) < 1e-5 and abs(float(w.sum()) - 1.0) < 1e-4, (cell, w.nonzero().tolist())
    return valid


def test_warp():
    eye, rot = torch.eye(2)[None], torch.tensor([[[0.0, -1.0], [1.0, 0.0]]])                         # rot: p_cur = (-y, x), 90 degrees
    cases = [(eye, torch.tensor([[2.0, 0.0]]), (21, 30)), (eye, torch.tensor([[0.0, 2.0]]), (20, 31)), (rot, torch.zeros(1, 2), (19, 20))]   # (-9, 11) -> (-11, -9) = cell (19, 20)
    for R, t, cell in cases:
        valid = _check_marked(lambda R_, t_: (R_, t_), R, t, cell)
        must_fail(_check_marked, _inverse, R, t, cell)                       # the wrong direction (current -> previous) puts the mark elsewhere
    assert float(valid.min()) > 1 - 1e-5, "a rotation by 90 degrees about the centre keeps the whole view"
    valid = _check_marked(lambda R_, t_: (R_, t_), *cases[0])
    assert float(valid[0, 0, 0].abs().max()) < 1e-6 and float(valid[0, 0, 1:].min()) > 1 - 1e-5, "2 m forward in x: the first row of the current view was outside the old view, the rest inside"
    # a real pose pair: a static world point marked in the previous frame lands where `jtime.warp` puts it (the centroid of the warped mark, bilinear: a fraction of a cell)
    pose = dict(np.load(POSES % "val"))
    has = np.nonzero(pose["prev"] >= 0)[0][:400]
    d = [float(np.linalg.norm(pose["t"][r] - pose["t"][pose["prev"][r]])) for r in has]
    row = int(has[int(np.argmax(d))])
    prow = int(pose["prev"][row])
    assert max(d) > 3.0, "the pair must move by more than a cell"
    R, t = LS.rel_pose(pose["R"][row], pose["t"][row], pose["R"][prow], pose["t"][prow])
    want = JTM.warp(np.array([[-9.0, 11.0, 0.0, 1.0]]), (pose["R"][prow], pose["t"][prow]), (pose["R"][row], pose["t"][row]))[0, :2]
    c = -50.0 + (torch.arange(50).float() + 0.5) * 2.0

    def centroid(Rt):
        s = torch.zeros(1, 1, 50, 50)
        s[0, 0, 20, 30] = 1.0
        w = JR.warp_state(s, *Rt)[0][0, 0]
        return np.array([float((w.sum(1) * c).sum() / w.sum()), float((w.sum(0) * c).sum() / w.sum())])

    Rt = (torch.from_numpy(R)[None], torch.from_numpy(t)[None])
    err, err_inv = np.linalg.norm(centroid(Rt) - want), np.linalg.norm(centroid(_inverse(*Rt)) - want)
    assert err < 0.3 and err_inv > 2.0, (err, err_inv)
    # rel_pose against the global coordinates of random poses
    rng = np.random.RandomState(0)
    for _ in range(5):
        a, b = rng.uniform(-3, 3, 2)
        Rc, Rp = (np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]]) for x in (a, b))
        tc, tp, p = rng.uniform(-50, 50, 2), rng.uniform(-50, 50, 2), rng.uniform(-20, 20, 2)
        R, t = LS.rel_pose(Rc, tc, Rp, tp)
        assert np.allclose(R @ p + t, Rc.T @ (tp + Rp @ p - tc), atol=1e-4)
    print(f"warp: one cell in x and in y, 90 degrees, the inverse pose fails; a real pair ({max(d):.1f} m): centroid error {err:.2f} m (inverse {err_inv:.1f} m); rel_pose: ok")


def test_zero_gate():
    f = fx()
    frozen, new = f["frozen"], f["new"]
    sd0, sd1 = frozen.state_dict(), new.state_dict()
    extra = sorted(set(sd1) - set(sd0))
    assert extra == ["jmodel.jm.fuse.conv.weight", "jmodel.jm.fuse.g"] and not set(sd0) - set(sd1), extra
    assert sum(sd1[k].numel() for k in extra) == 129 * 64 * 9 + 64 and float(sd1["jmodel.jm.fuse.g"].abs().sum()) == 0.0
    assert all(torch.equal(sd0[k], sd1[k]) for k in sd0)
    b = LD.collate([f["ds"][f["i_of_row"][r]] for r, _ in f["short"][:2]])
    x = b["x"].float()
    torch.manual_seed(0)
    state = torch.randn(2, 64, 50, 50)
    th = torch.tensor([0.3, -1.2])
    rel = (torch.stack([torch.stack([th.cos(), -th.sin()], -1), torch.stack([th.sin(), th.cos()], -1)], 1), torch.tensor([[3.0, -4.0], [0.5, 7.0]]))
    gt = {k: b[k] for k in ("ent", "ext", "ent_m", "ext_m")}

    def same(g, st):
        with torch.no_grad():
            _, j0, o0 = frozen(x, gt=g)
            _, j1, o1 = new(x, gt=g, state=st, rel=rel if st is not None else None)
        assert torch.equal(j0["j1"], j1["j1"]) and torch.equal(j0["f1"], j1["f1"]), "maps"
        assert all(torch.equal(o0[k], o1[k]) for k in ("logit", "kp_pos", "kp_dir", "kp_p", "valid", "allowed", "scale")), "pair head"
        assert j1["state"].shape == (2, 64, 50, 50)

    for g in (None, gt):                                                     # the read-out and the training path (GT cells injected)
        for st in (None, state):
            same(g, st)
    g0 = new.jmodel.jm.fuse.g.detach().clone()
    _open_gate(new, 0.05)                                                    # the mutation: an open gate changes the outputs, without and with a state
    must_fail(same, None, None)
    must_fail(same, None, state)
    with torch.no_grad():
        new.jmodel.jm.fuse.g.copy_(g0)
    # the same seed gives every tensor of `JEndModel` the same initial values (the fusion is created last)
    ma = f["args"]["maps_args"]
    torch.manual_seed(3)
    a0 = JT.JEndModel(ma, scales=True, e2e=True).state_dict()
    torch.manual_seed(3)
    a1 = JR.JRecurModel(ma, scales=True).state_dict()
    assert all(torch.equal(a0[k], a1[k]) for k in a0) and float(a1["jmodel.jm.fuse.conv.weight"].abs().sum()) > 0
    print("zero gate: the E341 weights give j1, f1, the keypoints and the pair logits of the frozen model bit for bit (no state, random state; with and without GT cells); an open gate does not; same initial tensors: ok")


def test_gradient_through_time():
    f = fx()
    m = f["new"]
    sq = LS.SeqData("val_clean", K=3, map_mode="error", pixels=False, pscale=True)                 # the fitted handle scales of the true pairs: the scale term of the E341 loss
    b = LS.collate([sq[0]])
    rows = [int(fb["row"][0]) for fb in b["frames"]]
    assert b["rel_R"].shape == (1, 2, 2, 2) and b["rel_t"].shape == (1, 2, 2) and all(int(f["pose"]["prev"][r]) == p for p, r in zip(rows[:-1], rows[1:])), rows
    R, t = LS.rel_pose(f["pose"]["R"][rows[2]], f["pose"]["t"][rows[2]], f["pose"]["R"][rows[1]], f["pose"]["t"][rows[1]])
    assert torch.equal(b["rel_R"][0, 1], torch.from_numpy(R)) and torch.equal(b["rel_t"][0, 1], torch.from_numpy(t)), "rel[:, k - 1] is the pose of frame k - 1 in frame k"
    c0 = []
    hook = m.jmodel.jm.p2.register_forward_hook(lambda mod, i, o: (o.retain_grad(), c0.append(o)))    # c0 = the first features of the maps head, per frame

    def grad_f1(state, gate):
        """|d loss(frame 3) / d c0(frame 1)| summed"""
        _open_gate(m, gate)
        c0.clear()
        m.zero_grad()
        losses, info = JRT.frame_losses(m, b, SimpleNamespace(state=state, cross_w=1.0, maps_w=1.0))
        assert len(losses) == 3 and all(torch.isfinite(x) for x in losses) and info["gate"] == gate
        losses[2].backward()
        return 0.0 if c0[0].grad is None else float(c0[0].grad.abs().sum()), float(m.jmodel.jm.fuse.conv.weight.grad.abs().sum())

    try:
        on, off, shut = grad_f1("on", 0.5), grad_f1("off", 0.5), grad_f1("on", 0.0)
    finally:
        hook.remove()
        _open_gate(m, 0.0)
    assert on[0] > 0 and on[1] > 0, on
    assert off[0] == 0.0 and off[1] > 0, f"--state off: frame 3 must not reach frame 1 (the fusion itself still trains on c0): {off}"
    assert shut[0] == 0.0, f"a zero gate passes nothing: {shut}"
    assert all(p.grad is None for p in m.jmodel.pyr.parameters()), "the frozen pyramid gets no gradient"
    print(f"gradient through time: the loss of frame 3 on the features of frame 1: {on[0]:.3g} with the state, exactly 0 with --state off and through a zero gate: ok")


def test_sequences():
    prev = np.array([-1, 0, 1, 2, -1, 4, 5])                                # two scenes: rows 0-3 and 4-6
    assert LS.chains(prev, [0, 1, 2, 3, 4, 6], 3) == [(0, 1, 2), (1, 2, 3)], "row 5 is missing: no sequence of the second scene; none crosses the scene start at row 4"
    assert LS.chains(prev, range(7), 3) == [(0, 1, 2), (1, 2, 3), (4, 5, 6)] and LS.chains(prev, [0, 1, 2, 3, 4, 6], 2) == [(0, 1), (1, 2), (2, 3)]
    z = np.load(POSES % "train")
    n = len(z["prev"])
    assert n == 8414 and int((z["prev"] < 0).sum()) == 212
    seqs = LS.chains(z["prev"], range(n), 3)
    p1 = z["prev"]
    assert len(seqs) == int(((p1 >= 0) & (p1[np.maximum(p1, 0)] >= 0)).sum()) and len(seqs) > 7000, len(seqs)
    for a, b, c in seqs[::97]:
        assert z["prev"][c] == b and z["prev"][b] == a and z["scene"][a] == z["scene"][b] == z["scene"][c] and z["ts"][a] < z["ts"][b] < z["ts"][c]
    drop = set(range(0, n, 50))                                              # rows without lane GT: no sequence holds one
    seqs = LS.chains(z["prev"], [r for r in range(n) if r not in drop], 3)
    assert seqs and not any(r in drop for s in seqs for r in s)
    print(f"sequences: toy (missing row, scene start) and TRAIN ({len(LS.chains(z['prev'], range(n), 3))} sequences of 3 over 212 scenes, linked by prev, one scene, time order): ok")


def test_stream_plan():
    pose = dict(scene=np.array([1, 1, 1, 1, 0, 0, 0, 0]), ts=np.array([10, 11, 12, 13, 5, 6, 7, 8]), prev=np.array([-1, 0, 1, 2, -1, 4, 5, 6]))
    order = LS.stream_order(pose, [3, 0, 2, 1, 7, 4, 6, 5])
    assert order == [(4, True), (5, False), (6, False), (7, False), (0, True), (1, False), (2, False), (3, False)], order
    assert LS.stream_order(pose, [0, 1, 3, 4, 5, 6, 7])[4:] == [(0, True), (1, False), (3, True)], "a gap (row 2 is not in the list) starts the state anew"
    reset = [r for _, r in order[:7]]                                        # scenes of 4 and 3 frames
    assert JRE.state_plan(reset, "stream") == [-1, 0, 1, 2, -1, 4, 5] and JRE.state_plan(reset, "off") == [-1] * 7
    assert JRE.state_plan(reset, "trunc3") == [-1, 0, 1, -1, -1, 4, 5]
    assert JRE.state_plan(reset, "shuffle") == [-1, 4, 5, 6, -1, 0, 1], "frame n reads the state of frame n - 1 of the next scene (the donor's last state when it is shorter), the last scene reads the first"
    f = fx()
    order, pose = f["order"], f["pose"]
    rows = [r for r, _ in order]
    assert len(rows) == 873 and len(set(rows)) == 873
    for i, (r, rs) in enumerate(order):
        assert rs == (pose["prev"][r] < 0), "VAL: the state starts anew exactly at the scene starts (no gaps inside a scene)"
        assert rs or (pose["prev"][r] == rows[i - 1] and pose["ts"][r] > pose["ts"][rows[i - 1]] and pose["scene"][r] == pose["scene"][rows[i - 1]])
    src = JRE.state_plan([r for _, r in order], "shuffle")
    assert all(s < 0 or pose["scene"][rows[s]] != pose["scene"][r] for s, r in zip(src, rows))
    print(f"stream order and state plan: toy and VAL (873 frames, {sum(r for _, r in order)} scene starts, time order, a reset exactly at prev < 0): ok")


def test_dump_modes():
    f = fx()
    m, ds, pose, short = f["new"], f["ds"], f["pose"], f["short"]
    reset = [r for _, r in short]
    assert reset == [True, False, False, False] * 2
    run = lambda mode, **k: JRE.run_stream(m, ds, short, JRE.state_plan(reset, mode), pose, "cpu", workers=0, **k)  # noqa: E731
    eq = lambda a, b: np.array_equal(a["logit"], b["logit"]) and np.array_equal(a["kp_pos"], b["kp_pos"])  # noqa: E731
    # zero gate: the stream dump is the cold dump and the stored E341 dump (made on the GPU with batch 8: float tolerance)
    z_stream, tot, _ = run("stream")
    z_off = run("off")[0]
    assert [fr["row"] for fr in z_stream] == [r for r, _ in short] and all(eq(a, b) for a, b in zip(z_stream, z_off))
    stored = {fr["row"]: fr for fr in pickle.load(open(os.path.join(JR.ROOT, "runs", RUN, "ends_val.pkl"), "rb"))}
    for fr in z_stream:
        st = stored[fr["row"]]
        assert set(fr) == set(st) and fr["token"] == st["token"] and np.array_equal(fr["valid"], st["valid"]) and all(fr[k].shape == st[k].shape and fr[k].dtype == st[k].dtype for k in ("kp_pos", "kp_dir", "kp_p", "logit", "allowed", "scale"))
        v = fr["valid"].reshape(-1).astype(bool)
        assert np.abs(fr["kp_pos"][v] - st["kp_pos"][v]).max() < 0.05 and np.abs(fr["logit"] - st["logit"])[fr["allowed"] & st["allowed"]].max(initial=0.0) < 0.1, fr["row"]
    assert tot["ent_n"] > 0 and set(tot) >= {"ent_hit", "ent_kp", "ent_ok", "ext_hit", "pairs_n", "pairs_hit", "pairs_kp", "pairs_ok"}
    # an open gate: the modes differ where they must
    _open_gate(m, 0.5)
    try:
        stream, off, tr3 = (run(mode)[0] for mode in ("stream", "off", "trunc3"))
        donor = run("stream", keep=True)[2]
        shuf = run("shuffle", donor=donor)[0]
        for i in range(8):
            n = i % 4
            assert eq(stream[i], off[i]) == (n == 0), f"frame {i}: the state changes the output except at a scene start"
            assert eq(tr3[i], off[i] if n == 3 else stream[i]), f"frame {i}: trunc3 is cold at every third frame of a scene and the stream in between"
            assert eq(shuf[i], off[i]) == (n == 0) and (n == 0 or not eq(shuf[i], stream[i])), f"frame {i}: the shuffled state is neither the own state nor none"
        with torch.no_grad():                                                # off = a cold model; shuffle by hand: frame 1 of the first scene reads the state of frame 0 of the second (stream index 4), warped with its own pose
            x = ds[f["i_of_row"][short[1][0]]]["x"][None].float()
            row = short[1][0]
            prow = int(pose["prev"][row])
            R, t = LS.rel_pose(pose["R"][row], pose["t"][row], pose["R"][prow], pose["t"][prow])
            cold = m(x)[2]["logit"][0].numpy()
            hand = m(x, state=donor[4].float(), rel=(torch.from_numpy(R)[None], torch.from_numpy(t)[None]))[2]["logit"][0].numpy()
        assert np.array_equal(cold, off[1]["logit"]) and np.array_equal(hand, shuf[1]["logit"])
    finally:
        _open_gate(m, 0.0)
    print("dump: zero gate = cold = the stored E341 dump (8 frames); open gate: stream / off / trunc3 / shuffle differ where they must, off = a cold model, shuffle = the donor state by hand: ok")


TESTS = [test_warp, test_sequences, test_stream_plan, test_zero_gate, test_gradient_through_time, test_dump_modes]

if __name__ == "__main__":
    want = [n if n.startswith("test_") else "test_" + n for n in sys.argv[1:]]
    assert all(n in {t.__name__ for t in TESTS} for n in want), want
    for t_ in TESTS:
        if not want or t_.__name__ in want:
            t_()
    print("test_jrecur ok")
