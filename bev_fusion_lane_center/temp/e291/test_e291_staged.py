"""E291 tests of the STAGED copies (jends.py, lane_gt_data.py, jends_train.py, jends_eval.py + jprior.py), CPU, 2 threads; run through the stage launcher (the rules of jprior.py itself are in `python jprior.py`):

    wsl bash temp/e291/run_staged.sh test_e291_staged.py [--only name,name]          sections: inject identity_random keys shuffle features_real smoke bfsig dump  (+ identity_real, which needs the reference below)
    BFLC_STAGE=none wsl bash temp/e291/run_staged.sh test_e291_staged.py --only ref    the REAL files (E312 code): writes the reference forward of the stored E312 model on 24 VAL frames to temp/_session_scratch_e291/ref_e312.pt
    wsl bash temp/e291/run_staged.sh test_e291_staged.py --only identity_real            the staged model with the prior ON and the gate closed, loaded from runs/e312_p_s0 (strict=False), against that reference: 0 difference
Run it with BFLC_STAGE=none and the sections must FAIL (the rules are new): `inject identity_random keys shuffle smoke bfsig dump` all do.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

ROOT = os.environ["BFLC_ROOT"]
SCR = os.path.join(ROOT, "temp", "_session_scratch_e291")
os.makedirs(SCR, exist_ok=True)
torch.set_num_threads(2)
import jends as JE  # noqa: E402
import jprior as JP  # noqa: E402
import qjunction as QJ  # noqa: E402


def real_module(name, rel):
    """the REAL file of the project under another name (the finder serves the staged copy for `import jends`)"""
    spec = importlib.util.spec_from_file_location(name + "_real", os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def synth(B=2, seed=0):
    """a random junction maps output and pyramid as the selftest of jends.py builds them, entry heading north, exit heading south"""
    g = torch.Generator().manual_seed(seed)
    jout = dict(j1=torch.randn(B, 8, 100, 100, generator=g) * 0.1 - 2.0, j05=torch.randn(B, 2, 200, 200, generator=g) - 2.0, f1=torch.randn(B, 64, 100, 100, generator=g))
    jout["j1"][:, 4], jout["j1"][:, 5], jout["j1"][:, 6], jout["j1"][:, 7] = 0.0, 5.0, 0.0, -5.0
    return jout, dict(centre1=torch.randn(B, 12, 100, 100, generator=g))


def gt_ends(B):
    ent = torch.tensor([[[-7.5, -9.5, 0.0, 1.0], [7.5, -9.5, 0.0, 1.0]]] * B)
    ext = torch.tensor([[[-5.5, -9.5, 0.0, -1.0], [5.5, -9.5, 0.0, -1.0]]] * B)
    return dict(ent=ent, ext=ext, ent_m=torch.ones(B, 2, dtype=torch.bool), ext_m=torch.ones(B, 2, dtype=torch.bool))


def random_arms(B, n=12, seed=0):
    rs = np.random.RandomState(seed)
    tabs = []
    for _ in range(B):
        a = rs.uniform(0, 2 * np.pi, n)
        tabs.append(dict(c=rs.uniform(-40, 40, (n, 2)).astype(np.float32), t=np.stack([np.cos(a), np.sin(a)], 1).astype(np.float32), n_in=rs.randint(0, 5, n), n_out=rs.randint(0, 5, n)))
    return JP.collate(tabs)


def same(a, b, what):
    assert set(a) == set(b), (what, set(a) ^ set(b))
    for k in a:
        if isinstance(a[k], torch.Tensor):
            assert torch.equal(a[k], b[k]), (what, k, float((a[k].float() - b[k].float()).abs().max()))


def t_identity_random():
    """flag off = the real head, bit for bit (weights and outputs); flag on with the gate closed (zero start) = the real head with the same weights; the prior is built last, so the other weights draw the same numbers"""
    real = real_module("jends", "training/grid/jends.py")
    jout, P = synth()
    gt, arms = gt_ends(2), random_arms(2)
    for kw in (dict(), dict(scales=True), dict(scales=True, unc=True)):
        torch.manual_seed(3)
        mr = real.JEnds(d=64, order=True, evidence=True, **kw)
        torch.manual_seed(3)
        ms = JE.JEnds(d=64, order=True, evidence=True, **kw)
        torch.manual_seed(3)
        mp = JE.JEnds(d=64, order=True, evidence=True, prior=True, **kw)
        same(mr.state_dict(), ms.state_dict(), "flag-off weights")
        assert mp.prior is not None and ms.prior is None and all(torch.equal(v, mp.state_dict()[k]) for k, v in mr.state_dict().items()), "the prior must be built last"
        missing = mp.load_state_dict(mr.state_dict(), strict=False)
        assert not missing.unexpected_keys and all(k.startswith("prior.") for k in missing.missing_keys) and len(missing.missing_keys) == 6, missing
        for gtx in (gt, None):
            for m in (mr, ms, mp):
                m.eval()
            with torch.no_grad():
                o_r, o_s, o_p = mr(jout, P, gt=gtx), ms(jout, P, gt=gtx), mp(jout, P, gt=gtx, arms=arms)
            assert "prior" in o_p and "prior" not in o_r and "prior" not in o_s
            o_p = {k: v for k, v in o_p.items() if k != "prior"}
            same(o_r, o_s, "flag off"), same(o_r, o_p, "flag on, gate closed")
            assert float(o_s["valid"].sum()) > 0
        pi = mp(jout, P, gt=None, arms=arms)["prior"]
        assert float((pi["n"] >= 0).sum()) > 0 and float(pi["term"].abs().max()) == 0.0, "the table reaches keypoints, the term is exactly zero"
    try:
        mp(jout, P, gt=gt)
        raise RuntimeError("a prior head without a table must refuse")
    except AssertionError as e:
        assert "arm table" in str(e)
    print("identity_random ok")


def t_inject():
    """the features of a real keypoint do not change with the GT injection, and an injected keypoint is not counted: on a frame with no real peak every arm has found = 0 whatever is injected"""
    m = JE.JEnds(d=64, order=True, evidence=False, prior=True)
    cap = {}
    m.prior.feat.register_forward_pre_hook(lambda _m, a: cap.__setitem__("f", a[0].detach().clone()))
    arms = JP.collate([dict(c=np.array([[0.0, -10.0]], np.float32), t=np.array([[0.0, 1.0]], np.float32), n_in=np.array([3]), n_out=np.array([2]))] * 2)
    jout, P = synth()
    jout["j1"][:, 2:4] = -9.0
    for cell in ((46, 40), (50, 40), (54, 40)):                                                  # three real entry peaks at x = -3.5, 0.5, 4.5, y = -9.5 (1 m cells, row = x)
        jout["j1"][:, 2, cell[0], cell[1]] = 3.0
    jout["j1"][:, 3, 48, 40] = 3.0                                                               # one real exit peak at x = -1.5
    gt = gt_ends(2)
    with torch.no_grad():
        o_g = m(jout, P, gt=gt, arms=arms)
        f_g = cap["f"].clone()
        o_n = m(jout, P, gt=None, arms=arms)
        f_n = cap["f"].clone()
    assert int(o_n["valid"].sum()) == 2 * 4 and int(o_g["valid"].sum()) == 2 * 8, (o_n["valid"].sum(), o_g["valid"].sum())
    K = m.k
    for b in range(2):
        for kind in (0, 1):
            ix_n = {int(c): k for k, c in enumerate(o_n["idx"][b, kind]) if bool(o_n["valid"][b, kind, k])}
            ix_g = {int(c): k for k, c in enumerate(o_g["idx"][b, kind]) if bool(o_g["valid"][b, kind, k])}
            assert ix_n and set(ix_n) < set(ix_g), (ix_n, ix_g)
            for c, k in ix_n.items():
                assert torch.equal(f_n[b, kind * K + k], f_g[b, kind * K + ix_g[c]]), "the features of a real keypoint must not depend on the injected ends"
    # no real peak at all: three entries and one exit are injected, none is counted
    jq = dict(jout, j1=jout["j1"].clone())
    jq["j1"][:, 2:4] = -9.0
    gt3 = dict(ent=torch.tensor([[[-3.5, -9.5, 0.0, 1.0], [0.5, -9.5, 0.0, 1.0], [4.5, -9.5, 0.0, 1.0]]] * 2), ext=torch.tensor([[[-1.5, -9.5, 0.0, -1.0]]] * 2), ent_m=torch.ones(2, 3, dtype=torch.bool), ext_m=torch.ones(2, 1, dtype=torch.bool))
    with torch.no_grad():
        oq = m(jq, P, gt=gt3, arms=arms)
    fq, vq = cap["f"], oq["valid"].flatten(1)
    assert int(vq.sum()) == 2 * 4 and float(fq[vq][:, 1].sum()) == 0.0, "the injected keypoints are assigned to the arm"
    ent, ext = fq[:, :K][vq[:, :K]], fq[:, K:][vq[:, K:]]
    assert torch.allclose(ent[:, 5], torch.full_like(ent[:, 5], -3 / 4)) and torch.allclose(ent[:, 4], torch.zeros_like(ent[:, 4])), "found = 0 and rank 0 on a frame with no real peak (entries: n = 3)"
    assert torch.allclose(ext[:, 5], torch.full_like(ext[:, 5], -2 / 4)), "exits: n = 2"
    print("inject ok")


def _batch_items(prior, n=6, split="val_clean"):
    import lane_gt_data as LD
    LD.QData.DEFAULT_JUNCTION, LD.QData.DEFAULT_PRIOR = True, tuple(prior)
    ds = LD.QData(split, map_mode="error", pixels=False)
    return ds, [ds[i] for i in range(0, n * 40, 40)]


def t_keys():
    """leak check on real frames: the table has exactly the four keys, the item only the key `arms` more, the collated table the same"""
    import lane_gt_data as LD
    ds, items = _batch_items((JP.Spec("clean"), JP.Spec.readout("U"), JP.Spec.train("mix")))
    _, base = _batch_items(())
    for it, bt in zip(items, base):
        assert set(it) - set(bt) == {"arms"} and set(bt) <= set(it) and len(it["arms"]) == 3
        for tb in it["arms"]:
            assert tuple(tb) == JP.KEYS and tb["c"].shape[1:] == (2,) and tb["t"].shape[1:] == (2,) and tb["n_in"].ndim == 1
    b = LD.collate(items)
    assert len(b["arms"]) == 3 and all(tuple(x) == JP.KEYS for x in b["arms"]) and "arms" not in LD.collate(base)
    cl, u = b["arms"][0], b["arms"][1]
    assert (cl["n_in"] > JP.PAD).sum() > 0 and (cl["n_in"] == JP.UNK).sum() == 0 and ((u["n_in"] == JP.UNK) == (cl["n_in"] > JP.PAD)).all(), "U = the clean table with every count unknown, geometry kept"
    assert torch.equal(u["c"], cl["c"]) and torch.equal(u["t"], cl["t"])
    print("keys ok; frames with arms:", int((cl["n_in"] > JP.PAD).any(1).sum()), "of", len(items))


def t_shuffle():
    """S = the table of another frame, deterministic, equal to the clean table of the partner (`jprior.partners`); differs from the clean table of the frame in almost all frames"""
    import lane_gt_data as LD
    ds, items = _batch_items((JP.Spec("clean"), JP.Spec.readout("S"), JP.Spec.readout("SC")), n=12)
    ds2, items2 = _batch_items((JP.Spec.readout("S"), JP.Spec.readout("SC")), n=12)
    toks = [ds.tokens[int(ds.bd.rows[k])] for k in ds.ks]
    diff = n_arm = sc_diff = 0
    for i, (it, it2) in zip(range(0, 12 * 40, 40), zip(items, items2)):
        c, s, sc = it["arms"]
        assert all(np.array_equal(s[k], it2["arms"][0][k]) and np.array_equal(sc[k], it2["arms"][1][k]) for k in JP.KEYS), "S and SC are deterministic"
        token = toks[i]
        partner = next(tk for tk in JP.partners(token, toks) if QJ.JG.junction_gt(ds.lg.load(tk)) is not None)
        want = JP.table(QJ.JG.junction_gt(ds.lg.load(partner)), token, JP.Spec("clean"))
        assert all(np.array_equal(s[k], want[k]) for k in JP.KEYS), "S is the clean table of the partner"
        assert np.array_equal(sc["c"], c["c"]) and np.array_equal(sc["t"], c["t"]), "SC keeps the frame's own arm centres and axes"
        k_ = len(want["n_in"])
        assert all(np.array_equal(sc[k], want[k][np.arange(len(c[k])) % k_]) for k in ("n_in", "n_out")), "SC: arm i gets the count of arm i modulo the number of arms of the partner"
        sc_diff += len(c["n_in"]) > 0 and not (np.array_equal(sc["n_in"], c["n_in"]) and np.array_equal(sc["n_out"], c["n_out"]))
        n_arm += len(c["n_in"]) > 0
        diff += not all(c[k].shape == s[k].shape and np.array_equal(c[k], s[k]) for k in JP.KEYS)
    assert diff >= 10, f"the shuffled table must differ from the clean one in almost all frames: {diff} of 12"
    assert n_arm >= 6 and sc_diff >= 0.6 * n_arm, f"the SC counts must differ from the clean ones on most frames with arms: {sc_diff} of {n_arm}"
    print("shuffle ok:", diff, "of 12 differ; SC counts differ in", sc_diff, "of", n_arm, "frames with arms")


def ref_features(pos, D, kind, ok, tb):
    """the six features of the keypoints of ONE frame by plain loops (numpy): the reference of `jprior.features` (n = count of the keypoint's direction, flag, axis distance, heading agreement, rank, found - n) and the arm index"""
    N, A = len(pos), len(tb["n_in"])
    ref = [[tb["t"][a] if kind[i] == 0 else -tb["t"][a] for a in range(A)] for i in range(N)]
    rel = [[pos[i] - tb["c"][a] for a in range(A)] for i in range(N)]
    lat = np.array([[rel[i][a] @ np.array([-ref[i][a][1], ref[i][a][0]]) for a in range(A)] for i in range(N)]).reshape(N, A)
    cos = np.array([[D[i] @ ref[i][a] for a in range(A)] for i in range(N)]).reshape(N, A)
    elig = np.array([[np.linalg.norm(rel[i][a]) <= JP.RADIUS and cos[i][a] >= JP.COS_MIN for a in range(A)] for i in range(N)]).reshape(N, A)
    f, arm = np.zeros((N, JP.NF)), -np.ones(N, int)
    for i in range(N):
        cand = [a for a in range(A) if elig[i][a]]
        if not cand:
            f[i, 1] = 1.0
            continue
        a = min(cand, key=lambda x: (abs(lat[i][x]), x))
        arm[i] = a
        n = int((tb["n_in"] if kind[i] == 0 else tb["n_out"])[a])
        f[i, 2], f[i, 3] = abs(lat[i][a]) / 20.0, cos[i][a]
        if n == JP.UNK:
            f[i, 1] = 1.0
            continue
        mem = [k for k in range(N) if ok[k] and kind[k] == kind[i] and elig[k][a]]
        # the lateral offset of k to the axis of arm a: the signed one of THAT arm (heading of the kind)
        rank = sum(1 for k in mem if lat[k][a] > lat[i][a] or (lat[k][a] == lat[i][a] and k < i))
        f[i, 0], f[i, 4], f[i, 5] = n / 4.0, rank / max(n, 1), (len(mem) - n) / 4.0
    return f, arm, elig


def t_features_real():
    """`jprior.features` against plain loops (numpy) on the real stored E312 dumps and the real clean tables, and the arm-side shares of keypoints fewer / equal / more than the count against the independent loop of the step 0 (temp/e291/step0.py, loose setting: peaks >= 0.1, 10 m, 60 degrees: the same quantity)"""
    import json
    import pickle
    sys.path.insert(0, os.path.join(ROOT, "tools", "viz"))
    import qgraph_view as QV
    st0 = json.load(open(os.path.join(ROOT, "temp", "e291", "step0_thr0.1_r10.json")))["stats"]["e312_p_s0"]
    frames = {f["token"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", "e312_p_s0", "ends_val.pkl"), "rb"))}
    cnt = {k: 0 for k in ("in_def", "in_eq", "in_sur", "out_def", "out_eq", "out_sur")}
    n_frames = n_kp = n_seen = 0
    for sid, (name, toks) in QV.val_scenes().items():
        for t in toks:
            jg = QJ.JG.junction_gt(QV.load_gt(t))
            if jg is None or t not in frames:
                continue
            f = frames[t]
            tb = JP.table(jg, t, JP.Spec("clean"))
            kind = np.r_[np.zeros(32, int), np.ones(32, int)]
            ok = f["valid"].reshape(-1)
            fe, arm = JP.features(torch.from_numpy(f["kp_pos"])[None].float(), torch.from_numpy(f["kp_dir"])[None].float(), torch.from_numpy(kind)[None], torch.from_numpy(ok)[None], JP.collate([tb]))
            fr, ar, elig = ref_features(f["kp_pos"].astype(np.float64), f["kp_dir"].astype(np.float64), kind, ok, tb)
            assert np.array_equal(arm[0].numpy(), ar) and np.abs(fe[0].numpy() - fr).max() < 1e-4, (t, np.abs(fe[0].numpy() - fr).max())
            n_frames += 1
            n_kp += int(ok.sum())
            n_seen += int(((arm[0].numpy() >= 0) & ok).sum())
            for a_i in range(len(tb["n_in"])):
                for k, nn_ in ((0, tb["n_in"][a_i]), (1, tb["n_out"][a_i])):
                    if nn_ <= 0:
                        continue
                    found = int((elig[:, a_i] & ok & (kind == k)).sum())
                    cnt[("in_" if k == 0 else "out_") + ("def" if found < nn_ else ("eq" if found == nn_ else "sur"))] += 1
    for d in ("in", "out"):
        tot = sum(cnt[f"{d}_{x}"] for x in ("def", "eq", "sur"))
        ref = {x: st0[f"{x}_{d}"] / st0[f"arms_with_{d}"] for x in ("def", "eq", "sur")}
        got = {x: cnt[f"{d}_{x}"] / tot for x in ("def", "eq", "sur")}
        print(f"  {d}: deficit / exact / surplus  here {got['def']:.3f} {got['eq']:.3f} {got['sur']:.3f}   step 0 {ref['def']:.3f} {ref['eq']:.3f} {ref['sur']:.3f}")
        assert all(abs(got[x] - ref[x]) < 0.004 for x in got), (d, got, ref)
    print(f"features_real ok: {n_frames} frames with junction GT, {n_kp} real keypoints, {n_seen} of them in reach of an arm ({n_seen / n_kp:.3f})")


def fake_main(mod, argv, steps=20, record=None):
    n_steps = steps
    """`mod.main()` with the run folder and the optimiser loop stubbed: `steps` real steps of AdamW on the trainer's parameter set, no checkpoint; `record` collects what the stub saw"""
    class Stop(Exception):
        pass

    class FakeRun:
        def __init__(self, name):
            self.name, self.lines = name, []

        def log(self, *a):
            self.lines.append(" ".join(str(x) for x in a))
            print("  log:", self.lines[-1][:300])

        def path(self, f):
            return os.devnull

        def mark_done(self):
            pass

    def fake_fit(model, loss_fn, run, *, steps, loader, lr, ckpt_every, seed, args, params, config):
        opt = torch.optim.AdamW(params, lr=1e-3)
        before = [p.detach().clone() for p in params]
        model.train()
        it = iter(loader)
        seen = []
        for s in range(n_steps):
            b = next(it)
            loss, info = loss_fn(model, b)
            assert torch.isfinite(loss), (s, info)
            opt.zero_grad()
            loss.backward()
            if "arms" in b:
                seen.append(b["arms"][0])
            opt.step()
        record.update(model=model, params=params, before=before, info=info, run=run, arms=seen, config=config)
        raise Stop()

    mod.T.Run, mod.T.fit = FakeRun, fake_fit
    sys.argv = ["x"] + argv
    try:
        mod.main()
    except Stop:
        pass
    return record


def maps_ckpt(sigma, name):
    import jmaps as JM
    torch.manual_seed(0)
    p = os.path.join(SCR, name)
    torch.save(dict(state=JM.JModel(width=48, d=64, ctx="dil").state_dict(), args=dict(ctx="dil", d=64, seed=0, base=None, raster="legacy", sigma=sigma)), p)
    return p


def t_smoke():
    """20 steps on the CPU, real TRAIN frames, random weights, the trainer's own main(): the loss is finite, the prior parameters get a gradient and move, the term against the token is logged, the batch tables are corrupted (mix) or clean (0), flag off builds no prior"""
    os.environ["BFLC_SIG"] = "1.5"
    import jends_train as JT
    ck = maps_ckpt(1.5, "maps_smoke.pt")
    common = ["--name", "smoke_e291", "--maps", ck, "--e2e", "--scales", "--maps-w", "1.0", "--bs", "2", "--workers", "0"]
    rec = fake_main(JT, common + ["--lane-prior", "--prior-noise", "mix"], steps=20, record={})
    m = rec["model"]
    assert m.je.prior is not None and all(any(p is q for q in rec["params"]) for p in m.je.prior.parameters()), "the prior is inside the trainer's parameter set"
    moved = [not torch.equal(a, p) for a, p in zip(rec["before"], rec["params"])]
    prior_ids = {id(p) for p in m.je.prior.parameters()}
    assert all(mv for mv, p in zip(moved, rec["params"]) if id(p) in prior_ids), "every prior parameter moved"
    log = [l for l in rec["run"].lines if l.startswith("prior gradient norm")]
    assert len(log) == 3 and "feat" in log[0], log
    gnorm = [float(l.split("feat ")[1].split(",")[0]) for l in log]
    assert all(g > 0 for g in gnorm), f"the feature layer has a gradient from step 1: {gnorm}"
    gate = [float(l.split("gate ")[1].split(" ")[0]) for l in log]
    assert gate[0] == 0.0 and gate[2] > 0.0, f"the gate has no gradient at step 1 (f starts at zero) and has one from step 2 or 3: {gate}"
    assert rec["info"]["prior_ratio"] > 0 and 0 < rec["info"]["prior_gate"] < 1, rec["info"]
    cat = torch.cat([a["n_in"].flatten() for a in rec["arms"]])
    assert (cat == JP.UNK).any() and (cat > JP.PAD).any(), "the training tables of `mix` hold unknown counts"
    rec0 = fake_main(JT, common + ["--lane-prior", "--prior-noise", "0"], steps=6, record={})
    cat0 = torch.cat([a["n_in"].flatten() for a in rec0["arms"]])
    assert not (cat0 == JP.UNK).any() and (cat0 > JP.PAD).any(), "the clean arm has no unknown counts"
    off = fake_main(JT, common, steps=3, record={})
    assert off["model"].je.prior is None and not off["arms"] and "prior_ratio" not in off["info"] and "lane_prior" not in off["config"], "flag off: no prior, no table"
    try:
        fake_main(JT, common + ["--prior-noise", "mix"], steps=1, record={})
        raise RuntimeError("--prior-noise mix without --lane-prior must refuse")
    except AssertionError as e:
        assert "--lane-prior" in str(e)
    print("smoke ok; prior gradient norms at steps 1, 2, 3:", gnorm, "| term / token", round(rec["info"]["prior_ratio"], 5))


def t_bfsig():
    """a launcher without BFLC_SIG stops the run: the sigma of the maps checkpoint must be the one of the environment"""
    import jends_train as JT
    ck = maps_ckpt(1.0, "maps_sig1.pt")
    for env, ok in (("1.0", True), (None, False), ("1.5", False)):
        if env is None:
            os.environ.pop("BFLC_SIG", None)
        else:
            os.environ["BFLC_SIG"] = env
        stopped = None
        try:
            fake_main(JT, ["--name", "smoke_e291", "--maps", ck, "--e2e", "--scales", "--bs", "2", "--workers", "0"], steps=1, record={})
        except AssertionError as e:
            stopped = str(e)
        assert (stopped is None) == ok and (ok or ("BFLC_SIG" in stopped and "maps checkpoint" in stopped)), f"BFLC_SIG={env} with sigma-1.0 maps: ok={ok}, stopped with {stopped!r}"
    print("bfsig ok")


def t_dump():
    """the dump of a model with the prior: ONE pass writes a new folder per read-out (model and config copied, ends_val.pkl, ends_head.json with the prior numbers), never over an existing one; the read-outs differ; a model without the prior dumps as before"""
    import pickle
    os.environ["BFLC_SIG"] = "1.5"
    import jends_eval as JV
    import jends_train as JT
    import tempfile
    tmp = tempfile.TemporaryDirectory(prefix="dump_root_", dir=SCR)                                  # a fresh folder per run (the dump refuses an existing read-out folder); removed at the end
    root = tmp.name
    os.makedirs(os.path.join(root, "runs"), exist_ok=True)
    JV.ROOT = root                                                                              # only the runs folder of the dump moves; the data loaders keep the project's ROOT
    e312 = torch.load(os.path.join(ROOT, "runs", "e312_p_s0", "model.pt"), map_location="cpu")                       # the stored E312 model: real keypoints (a random-weight head has no peak above 0.1)
    for name, prior in (("p_s0", True), ("c_s0", False)):
        m = JT.JEndModel(e312["args"]["maps_args"], order=True, evidence=True, scales=True, e2e=True, prior=prior)
        m.load_state_dict(e312["state"], strict=not prior)
        torch.manual_seed(0)
        if prior:                                                                               # open the gate: a trained term
            for p in m.je.prior.parameters():
                torch.nn.init.normal_(p, 0.0, 0.5)
        os.makedirs(os.path.join(root, "runs", name), exist_ok=True)
        torch.save(dict(state=m.state_dict(), args=dict(e312["args"], lane_prior=prior)), os.path.join(root, "runs", name, "model.pt"))
        open(os.path.join(root, "runs", name, "config.json"), "w").write("{}")
    ns = argparse.Namespace(run="p_s0", thr=0.1, max=16, prior_mode="clean,noise,pm1,u,s,sc", prior_noise=0.3)
    JV.dump(ns)
    names = ["clean", "noise30", "pm1", "u", "s", "sc"]
    fr = {}
    for n in names:
        d = os.path.join(root, "runs", f"p_s0_{n}_max16")                                           # a --max dump has its own folder names
        assert all(os.path.exists(os.path.join(d, f)) for f in ("model.pt", "config.json", "ends_val.pkl", "ends_head.json")), n
        fr[n] = pickle.load(open(os.path.join(d, "ends_val.pkl"), "rb"))
        assert len(fr[n]) == 16 and set(fr[n][0]["prior"]) == {"gate", "term", "tok", "n", "arm"}
    assert not os.path.exists(os.path.join(root, "runs", "p_s0", "ends_val.pkl")), "a prior model's dump never goes into the training run's folder"
    n_of = lambda n: np.concatenate([f["prior"]["n"] for f in fr[n]])  # noqa: E731
    arm_of = lambda n: np.concatenate([f["prior"]["arm"] for f in fr[n]])  # noqa: E731
    assert np.array_equal(arm_of("sc"), arm_of("clean")) and np.array_equal(arm_of("u"), arm_of("clean")), "SC and U keep the geometry: the same keypoints see the same arms"
    assert (n_of("sc") >= 0).any() and not np.array_equal(n_of("sc"), n_of("clean")) and not np.array_equal(n_of("sc"), n_of("s")), "SC: the own arms with the counts of another frame"
    assert (n_of("clean") >= 0).any() and (n_of("u") == -1).all() and not np.array_equal(n_of("clean"), n_of("s")) and not np.array_equal(n_of("clean"), n_of("noise30")), "the read-outs must differ"
    for k in ("kp_pos", "kp_p", "valid"):                                                       # the prior acts on the tokens only: the keypoints of one model are the same in every read-out
        assert all(np.array_equal(fr["clean"][i][k], fr["s"][i][k]) for i in range(16)), k
    assert any(not np.array_equal(fr["clean"][i]["logit"], fr["u"][i]["logit"]) for i in range(16)), "an open gate: the table changes the pair logits"
    import json
    heads = {n: json.load(open(os.path.join(root, "runs", f"p_s0_{n}_max16", "ends_head.json")))["prior"] for n in names}
    assert heads["clean"]["term_over_token"] > 0 and all(0.0 <= h["arm_share"] <= 1.0 for h in heads.values()) and abs(heads["sc"]["arm_share"] - heads["clean"]["arm_share"]) < 1e-9 and heads["s"]["arm_share"] <= heads["clean"]["arm_share"]
    print("  share of the valid keypoints with an arm in reach:", {n: round(h["arm_share"], 3) for n, h in heads.items()})
    runs = lambda: sorted(os.listdir(os.path.join(root, "runs")))  # noqa: E731
    before = runs()
    try:
        JV.dump(argparse.Namespace(run="p_s0", thr=0.1, max=16, prior_mode="noise10,sc", prior_noise=0.0))                        # sc exists, noise10 does not: all names are checked first, nothing is made
        raise RuntimeError("an existing read-out folder must be refused")
    except AssertionError as e:
        assert "never overwritten" in str(e) and runs() == before, (str(e), runs())
    import lane_gt_data as LD
    real_loader = LD.loader

    def boom(*args, **kw):
        raise ValueError("a crash in the pass")
    LD.loader = boom
    try:
        JV.dump(argparse.Namespace(run="p_s0", thr=0.1, max=16, prior_mode="noise10,noise20", prior_noise=0.0))
        raise RuntimeError("the crash must propagate")
    except ValueError:
        assert runs() == before, "a crash leaves no folder (empty or not)"
    finally:
        LD.loader = real_loader
    JV.dump(argparse.Namespace(run="c_s0", thr=0.1, max=16, prior_mode=None, prior_noise=0.0))
    old = pickle.load(open(os.path.join(ROOT, "runs", "e312_p_s0", "ends_val.pkl"), "rb"))[:16]
    new = pickle.load(open(os.path.join(root, "runs", "c_s0", "ends_val.pkl"), "rb"))
    assert len(new) == 16 and all(set(n_) == set(o_) and "prior" not in n_ and n_["token"] == o_["token"] for n_, o_ in zip(new, old)), "flag off: the dump has the keys and the frames of the stored E312 dump"
    canon = lambda f: (lambda q: q[np.lexsort((q[:, 1], q[:, 0]))])(f["kp_pos"][f["valid"].reshape(-1)])  # noqa: E731    (the order of keypoints with equal scores differs between the GPU and the CPU: compare the sets)
    assert all(n_["valid"].sum() == o_["valid"].sum() and np.abs(canon(n_) - canon(o_)).max() < 1e-2 for n_, o_ in zip(new, old)), "flag off: the keypoints of the stored E312 dump (made on the GPU) are reproduced"
    try:
        JV.dump(argparse.Namespace(run="c_s0", thr=0.1, max=16, prior_mode="clean", prior_noise=0.0))
        raise RuntimeError("a model without the prior must refuse --prior-mode")
    except AssertionError as e:
        assert "no lane prior" in str(e)
    tmp.cleanup()
    print("dump ok")


def _loader_val(n=24, prior=()):
    import lane_gt_data as LD
    LD.QData.DEFAULT_JUNCTION, LD.QData.DEFAULT_PRIOR = True, tuple(prior)
    ds = LD.QData("val_clean", ks=None, map_mode="error", pixels=False)
    ds.ks = ds.ks[:n]
    return torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=LD.collate)


def _e312_model(prior):
    import jends_train as JT
    ck = torch.load(os.path.join(ROOT, "runs", "e312_p_s0", "model.pt"), map_location="cpu")
    a = ck["args"]
    m = JT.JEndModel(a["maps_args"], order=not a["no_order"], evidence=not a["no_evidence"], scales=bool(a.get("scales", False)), ntok=bool(a.get("ntok", False)), unc=bool(a.get("unc", False)), attach=bool(a.get("attach", False)), **({"prior": True} if prior else {}))
    res = m.load_state_dict(ck["state"], strict=not prior)
    if prior:
        assert not res.unexpected_keys and res.missing_keys and all(k.startswith("je.prior.") for k in res.missing_keys), res
    return m.eval()


def t_ref():
    """(BFLC_STAGE=none) the forward of the stored E312 model with the REAL code on the first 24 VAL frames"""
    m, outs = _e312_model(False), []
    with torch.no_grad():
        for b in _loader_val():
            _, _, o = m(b["x"].float(), gt=None)
            outs.append({k: v for k, v in o.items() if isinstance(v, torch.Tensor)})
    ref = {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
    torch.save(ref, os.path.join(SCR, "ref_e312.pt"))
    print("ref written:", {k: tuple(v.shape) for k, v in ref.items()})


def t_identity_real():
    """kill criterion (2): the staged model with the prior ON and the gate closed (zero start), loaded with the stored E312 checkpoint (strict=False for the new parameters), reproduces the E312 forward of the real code to 0 on 24 VAL frames"""
    if not os.path.exists(os.path.join(SCR, "ref_e312.pt")):
        print("identity_real SKIPPED: no ref_e312.pt (it is made by `--only ref` with the code BEFORE the install; after the install the pinned sums of jends.py's selftest take its place)")
        return
    ref = torch.load(os.path.join(SCR, "ref_e312.pt"))
    m = _e312_model(True)
    outs, n_shown = [], 0
    with torch.no_grad():
        for b in _loader_val(prior=(JP.Spec("clean"),)):
            _, _, o = m(b["x"].float(), gt=None, arms={k: v for k, v in b["arms"][0].items()})
            n_shown += int((o["prior"]["n"] >= 0).sum())
            assert float(o["prior"]["term"].abs().max()) == 0.0
            outs.append({k: v for k, v in o.items() if isinstance(v, torch.Tensor)})
    new = {k: torch.cat([o[k] for o in outs]) for k in outs[0] if k in ref}
    assert set(new) == set(ref), set(new) ^ set(ref)
    for k in ref:
        assert torch.equal(new[k], ref[k]), (k, float((new[k].float() - ref[k].float()).abs().max()))
    assert n_shown > 0, "the table must reach real keypoints (else the identity says nothing)"
    print("identity_real ok: 24 frames, outputs", sorted(ref), "equal to 0;", n_shown, "keypoints saw a count")


SECTIONS = dict(inject=t_inject, identity_random=t_identity_random, keys=t_keys, shuffle=t_shuffle, features_real=t_features_real, smoke=t_smoke, bfsig=t_bfsig, dump=t_dump, ref=t_ref, identity_real=t_identity_real)
DEFAULT = ("inject", "identity_random", "keys", "shuffle", "features_real", "smoke", "bfsig", "dump")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(DEFAULT))
    for name in ap.parse_args().only.split(","):
        print(f"--- {name}", flush=True)
        SECTIONS[name]()
    print("test_e291_staged ok")
