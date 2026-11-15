"""E317 scratch: real-frame tests of the DEV copies on CPU (2 threads), no GPU.
usage: wsl bash tools/env/_py14.sh e317/t_real.py <fwd orig|fwd dev | compare | geom | bench | flops>
  fwd <v>   the trained E312 model (runs/e312_p_s0) through the <v> copies (jends, qjunction, lane_gt_data, jends_train) on 4 VAL frames: outputs -> temp/e317/real_<v>.pkl
  compare   the two pickles are bit-identical (flag-off regression on real data)
  geom      dev: the data side (ppts present, the pair order equals the junction GT's) and the geometry loss of E312's OWN curves on 16 TRAIN frames (the starting point of E317: shape distance, end error)
  bench     dev: seconds of jends_loss with and without the geometry loss on a real batch of 8 (CPU)
  flops     dev: parameters and MACs of the pair head with and without --unc, batch 1 (torch.utils.flop_counter)
"""
import os
import pickle
import sys
import time

import numpy as np
import torch

torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
mode = sys.argv[1]
var = sys.argv[2] if len(sys.argv) > 2 else "dev"


def use(folder):
    """import jends / qjunction / lane_gt_data / jends_train from temp/e317/<folder> as the plain module names"""
    import importlib.util
    from importlib.machinery import SourceFileLoader                           # *.pysrc: no second jends.py under temp/ (it would make `_py14.sh jends_train.py` ambiguous and stop the chains)
    mods = {}
    for n in ("jends", "qjunction", "lane_gt_data", "jends_train"):
        for k in [k for k in sys.modules if k == n]:
            del sys.modules[k]
    for n in ("jends", "qjunction", "lane_gt_data", "jends_train"):
        ld = SourceFileLoader(n, os.path.join(HERE, folder, n + ".pysrc"))
        spec = importlib.util.spec_from_loader(n, ld)
        m = importlib.util.module_from_spec(spec)
        sys.modules[n] = m
        ld.exec_module(m)
        mods[n] = m
    return mods


def model_e312(M, scales_args=True):
    ck = torch.load(os.path.join(ROOT, "runs", "e312_p_s0", "model.pt"), map_location="cpu")
    a = ck["args"]
    m = M["jends_train"].JEndModel(a["maps_args"], order=not a["no_order"], evidence=not a["no_evidence"], scales=bool(a.get("scales", False)), capacity=bool(a.get("capacity", False)), ntok=bool(a.get("ntok", False)))
    m.load_state_dict(ck["state"])
    return m.eval()


def batches(M, split, n, bs, ppts=False, shuffle=False):
    LD = M["lane_gt_data"]
    LD.QData.DEFAULT_JUNCTION = True
    LD.QData.DEFAULT_PSCALE = True
    if hasattr(LD.QData, "DEFAULT_PPTS"):
        LD.QData.DEFAULT_PPTS = ppts
    ds, dl = LD.loader(split, bs=bs, workers=0, shuffle=shuffle, map_mode="error" if split != "train" else "train", pixels=False)
    out = []
    for b in dl:
        out.append(b)
        if len(out) * bs >= n:
            break
    return ds, out


if mode == "fwd":
    M = use(var)
    m = model_e312(M)
    ds, bl = batches(M, "val_clean", 4, 4)
    b = bl[0]
    with torch.no_grad():
        _, _, out = m(b["x"].float(), gt=None)
    res = {k: v.numpy() for k, v in out.items() if torch.is_tensor(v)}
    pickle.dump(res, open(os.path.join(HERE, f"real_{var}.pkl"), "wb"))
    print(var, "forward on", len(b["token"]), "VAL frames; keys", sorted(res))
elif mode == "compare":
    a, b = (pickle.load(open(os.path.join(HERE, f"real_{v}.pkl"), "rb")) for v in ("orig", "dev"))
    assert set(a) == set(b), set(a) ^ set(b)
    for k in a:
        assert np.array_equal(a[k], b[k]), f"{k} differs"
    print("real-frame regression: all", len(a), "output tensors of the E312 model are bit-identical between the original and the dev copies")
elif mode == "geom":
    M = use("dev")
    JE, QJ = M["jends"], M["qjunction"]
    m = model_e312(M)
    ds, bl = batches(M, "train", 16, 8, ppts=True)
    # the data side: pairs of the batch == pairs of the junction GT, ppts finite where there is a pair
    for i in range(6):
        o = ds[i]
        g = ds.lg.load(o["token"])
        import junction_paint as JP
        try:
            paint = JP.load(o["token"])
        except FileNotFoundError:
            paint = None
        jt, jg = QJ.junction_targets(g, paint)
        if jg is None:
            continue
        assert len(o["pairs"]) == len(jg["pairs"]) and np.array_equal(o["pairs"].numpy(), jg["pairs"][:, :2].astype(int)), "the order of the batch's pairs differs from the junction GT's"
        assert o["ppts"].shape == (len(jg["pairs"]), QJ.PPTS_N, 2) and torch.isfinite(o["ppts"]).all()
    print("data side ok: pair order identical to the junction GT, ppts finite")
    tot = {}
    nb = 0
    for b in bl:
        gt = {k: b[k] for k in ("ent", "ext", "ent_m", "ext_m")}
        with torch.no_grad():
            _, jout, out = m(b["x"].float(), gt=gt)
        loss, info = JE.jends_loss(out, b, driven_w=0.0, geom_w=0.5)
        for k, v in info.items():
            tot[k] = tot.get(k, 0.0) + v
        nb += 1
    print("E312's own curves on", 8 * nb, "TRAIN frames (mean over batches):", {k: round(v / nb, 3) for k, v in tot.items() if k.startswith("geom")})
elif mode == "bench":
    M = use("dev")
    JE = M["jends"]
    m = model_e312(M)
    ds, bl = batches(M, "train", 8, 8, ppts=True)
    b = bl[0]
    gt = {k: b[k] for k in ("ent", "ext", "ent_m", "ext_m")}
    with torch.no_grad():
        _, jout, _ = m(b["x"].float(), gt=gt)
    jr = {k: v.detach().clone() for k, v in jout.items()}
    P = None
    with torch.no_grad():
        P, _ = m.jmodel(b["x"].float(), joint=False)
    res = {}
    for gw in (0.0, 0.5):
        ts = []
        for r in range(12):
            out = m.je(jout, P, gt=gt)
            t0 = time.perf_counter()
            loss, info = JE.jends_loss(out, b, geom_w=gw)
            ts.append(time.perf_counter() - t0)
        res[gw] = float(np.median(ts[2:]))
    print(f"jends_loss on a real batch of 8 (CPU, forward only, median of 10): geom_w=0 {1000 * res[0.0]:.1f} ms, geom_w=0.5 {1000 * res[0.5]:.1f} ms, the geometry term adds {1000 * (res[0.5] - res[0.0]):.1f} ms")
    # with the backward pass
    for gw in (0.0, 0.5):
        ts = []
        for r in range(8):
            jr2 = {k: v.detach().clone().requires_grad_(k in ("j1",)) for k, v in jout.items()}
            out = m.je(jr2, P, gt=gt)
            t0 = time.perf_counter()
            loss, info = JE.jends_loss(out, b, geom_w=gw)
            loss.backward()
            ts.append(time.perf_counter() - t0)
        res[("bw", gw)] = float(np.median(ts[2:]))
    print(f"loss + backward through the pair head (CPU): geom_w=0 {1000 * res[('bw', 0.0)]:.1f} ms, geom_w=0.5 {1000 * res[('bw', 0.5)]:.1f} ms, the geometry term adds {1000 * (res[('bw', 0.5)] - res[('bw', 0.0)]):.1f} ms per step")
elif mode == "step":
    # the LIVE modules (the installed files), the E307 maps of seed 0, e2e training for a few steps on CPU with batch 2: finite loss, a gradient in the maps head and the pair head, the geometry numbers move in the log
    import jends as JE
    import jends_train as JT
    import jmaps as JM
    import lane_gt_data as LD
    assert JE.__file__.startswith(os.path.join(ROOT, "training")), JE.__file__
    for unc in (False, True):
        torch.manual_seed(0)
        LD.QData.DEFAULT_JUNCTION, LD.QData.DEFAULT_PSCALE, LD.QData.DEFAULT_PPTS = True, True, True
        ck = torch.load(os.path.join(ROOT, "runs", "e307_n_s0", "model.pt"), map_location="cpu")
        m = JT.JEndModel(ck["args"], order=True, evidence=True, scales=True, e2e=True, unc=unc)
        m.jmodel.load_state_dict(ck["state"])
        ds, dl = LD.loader("train", bs=2, workers=0, shuffle=True, map_mode="train", pixels=False)
        ps = list(m.je.parameters()) + list(m.jmodel.jm.parameters())
        for p in m.parameters():
            p.requires_grad_(False)
        for p in ps:
            p.requires_grad_(True)
        opt = torch.optim.Adam(ps, lr=1e-3)
        m.train()
        hist = []
        for n, b in enumerate(dl):
            gt = {k: b[k] for k in ("ent", "ext", "ent_m", "ext_m")}
            _, jout, out = m(b["x"].float(), gt=gt)
            loss, info = JE.jends_loss(out, b, cross_w=1.0, driven_w=0.0, geom_w=0.5)
            lm, im = JM.jmaps_loss(jout, b)
            loss = loss + 1.0 * lm
            assert torch.isfinite(loss), (n, info)
            opt.zero_grad()
            loss.backward()
            gm = sum(float(p.grad.abs().sum()) for p in m.jmodel.jm.parameters() if p.grad is not None)
            gl = float(m.je.logsig.weight.grad.abs().sum()) + float(m.je.logsig.bias.grad.abs().sum()) if unc and m.je.logsig.weight.grad is not None else 0.0
            assert gm > 0, "no gradient in the maps head"
            opt.step()
            hist.append((round(float(loss), 3), {k: round(v, 3) for k, v in info.items() if k.startswith("geom")}, round(gl, 4)))
            if n >= 3:
                break
        print(f"live modules, unc={unc}: 4 CPU steps ok; (loss, geometry info, |grad logsig|) per step:")
        for h in hist:
            print("  ", h)
elif mode == "step_attach":
    # the DEV modules: JEndModel(attach=True) on the E307 maps + B's node head (runs/e273_p_s0), e2e, 6 CPU steps with batch 2 on real train frames: the attached ends, the loss pieces, a gradient into the maps head and the attachment
    M = use("dev")
    JE, JT, LD, JM = M["jends"], M["jends_train"], M["lane_gt_data"], __import__("jmaps")
    torch.manual_seed(0)
    LD.QData.DEFAULT_JUNCTION, LD.QData.DEFAULT_PSCALE, LD.QData.DEFAULT_PPTS = True, True, True
    ck = torch.load(os.path.join(ROOT, "runs", "e307_n_s0", "model.pt"), map_location="cpu")
    m = JT.JEndModel(ck["args"], order=True, evidence=True, scales=True, e2e=True, attach=True)
    m.jmodel.load_state_dict(ck["state"])
    bs_ = torch.load(os.path.join(ROOT, "runs", "e273_p_s0", "model.pt"), map_location="cpu")["state"]
    m.nodes.load_state_dict({k[6:]: v for k, v in bs_.items() if k.startswith("nodes.")})
    ds, dl = LD.loader("train", bs=2, workers=0, shuffle=True, map_mode="train", pixels=False)
    ps = list(m.je.parameters()) + list(m.jmodel.jm.parameters())
    for p in m.parameters():
        p.requires_grad_(False)
    for p in ps:
        p.requires_grad_(True)
    print("trainable", sum(p.numel() for p in ps), "of which the attachment", sum(p.numel() for p in m.je.attach.parameters()))
    opt = torch.optim.Adam(ps, lr=1e-3)
    m.train()
    for n, b in enumerate(dl):
        gt = {k: b[k] for k in ("ent", "ext", "ent_m", "ext_m")}
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=bool(os.environ.get("AUTOCAST"))):                         # the trainer runs the forward AND the loss under bf16 autocast on the GPU
            _, jout, out = m(b["x"].float(), gt=gt)
            loss, info = JE.jends_loss(out, b, cross_w=1.0, driven_w=0.0, geom_w=0.5)
            lm, im = JM.jmaps_loss(jout, b)
            loss = loss + 1.0 * lm
        assert torch.isfinite(loss), (n, info)
        opt.zero_grad()
        loss.backward()
        ga = sum(float(p.grad.abs().sum()) for p in m.je.attach.parameters() if p.grad is not None)
        gm = sum(float(p.grad.abs().sum()) for p in m.jmodel.jm.parameters() if p.grad is not None)
        assert ga > 0 and gm > 0, (ga, gm)
        opt.step()
        print(n, round(float(loss), 3), {k: round(v, 3) for k, v in info.items() if k.startswith(("geom", "att"))}, "valid candidates per keypoint", round(float(out["att_cvalid"].sum(-1).float().mean()), 2))
        if n >= 5:
            break
    print("step_attach ok")
elif mode == "readout":
    # the attached read-out pipeline on 8 VAL frames: a CPU-built attach checkpoint (untrained attachment), dump (dev jends_eval) with the att_* keys, then jmaneuver (dev) with plain / p0.1 / p0.1s / p0.1a
    from importlib.machinery import SourceFileLoader
    from types import SimpleNamespace
    import importlib.util
    M = use("dev")
    JT = M["jends_train"]
    torch.manual_seed(0)
    ck = torch.load(os.path.join(ROOT, "runs", "e312_p_s0", "model.pt"), map_location="cpu")
    a = dict(ck["args"])
    m = JT.JEndModel(a["maps_args"], order=not a["no_order"], evidence=not a["no_evidence"], scales=True, e2e=True, attach=True)
    st = ck["state"]
    m.load_state_dict({k: v for k, v in st.items()}, strict=False)                                    # E312's maps head + pair head; the attachment stays at its random start
    bs_ = torch.load(os.path.join(ROOT, "runs", "e273_p_s0", "model.pt"), map_location="cpu")["state"]
    m.nodes.load_state_dict({k[6:]: v for k, v in bs_.items() if k.startswith("nodes.")})
    a.update(attach=True, geom_w=0.5, unc=False)
    d = os.path.join(ROOT, "runs", "e320_smoke_cpu_s0")
    os.makedirs(d, exist_ok=True)
    torch.save(dict(state=m.state_dict(), args=a), os.path.join(d, "model.pt"))
    import json
    json.dump(dict(seed=0), open(os.path.join(d, "config.json"), "w"))
    def modsrc(name):
        ld = SourceFileLoader(name, os.path.join(HERE, "dev", name + ".pysrc"))
        mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, ld))
        ld.exec_module(mod)
        return mod
    ev = modsrc("jends_eval")
    ev.dump(SimpleNamespace(run="e320_smoke_cpu_s0", thr=0.5, max=8))
    fr = pickle.load(open(os.path.join(d, "ends_val.pkl"), "rb"))
    f = fr[0]
    assert all(k in f for k in ("att_pos", "att_dir", "att_p")) and f["att_pos"].shape == f["kp_pos"].shape, sorted(f)
    print("dump has the attached ends:", {k: f[k].shape for k in ("kp_pos", "att_pos", "att_dir", "att_p")}, "mean attach prob", float(np.mean([x["att_p"].mean() for x in fr])))
    jm = modsrc("jmaneuver")
    jm.score(SimpleNamespace(run="e320_smoke_cpu_s0", seed=0, variants=["plain", "p0.1", "p0.1s", "p0.1a"], max=8, rows=False, chord=False))
elif mode == "score_attach":
    # qends_eval.cmd_score (dev) with --attach on the smoke dump of 8 frames: the card file name, the number of curves, the attached ends used (B's dump truncated to the 8 frames, val_frames patched)
    from importlib.machinery import SourceFileLoader
    from types import SimpleNamespace
    import importlib.util
    import qgraph_eval as QGE
    full = pickle.load(open(os.path.join(ROOT, "temp", "e276", "predfull_e273_p_s0.pkl"), "rb"))[:8]
    f8 = os.path.join(HERE, "full8.pkl")
    pickle.dump(full, open(f8, "wb"))
    orig_vf = QGE.val_frames
    QGE.val_frames = lambda: orig_vf()[:8]
    ld = SourceFileLoader("qends_eval", os.path.join(HERE, "dev", "qends_eval.pysrc"))
    qe = importlib.util.module_from_spec(importlib.util.spec_from_loader("qends_eval", ld))
    ld.exec_module(qe)
    base = dict(run="e320_smoke_cpu_s0", seed=0, thr=float(os.environ.get("THR", 0.1)), full=f8, base=None, skip_base_check=True, chord=False, snap=False)
    for kw in ((dict(attach=True),) if os.environ.get("THR") else (dict(), dict(attach=True), dict(snap=True))):
        print("---", kw)
        qe.cmd_score(SimpleNamespace(**dict(base, **kw)))
    print(sorted(x for x in os.listdir(os.path.join(ROOT, "runs", "e320_smoke_cpu_s0")) if x.startswith("ends_card")))
elif mode == "cost":
    # CPU cost of the attachment: parameters, MACs (flop counter) and milliseconds of Attach.forward at batch 1, and of the frozen node head + the whole forward at batch 2 with and without it
    M = use("dev")
    JE, JT = M["jends"], M["jends_train"]
    from torch.utils.flop_counter import FlopCounterMode
    ck = torch.load(os.path.join(ROOT, "runs", "e307_n_s0", "model.pt"), map_location="cpu")
    ds, bl = batches(M, "val_clean", 2, 2)
    x = bl[0]["x"].float()
    res = {}
    for att in (False, True):
        m = JT.JEndModel(ck["args"], order=True, evidence=True, scales=True, e2e=True, attach=att).eval()
        m.jmodel.load_state_dict(ck["state"])
        ts = []
        for r in range(4):
            t0 = time.perf_counter()
            with torch.no_grad():
                P, jout, out = m(x)
            ts.append(time.perf_counter() - t0)
        res[att] = float(np.median(ts[1:]))
        if att:
            with torch.no_grad():
                nd = m.nodes(P["feat2"], P["feat1"], P["feat05"], P["road2"], P["centre1"])
                tt = torch.randn(1, 64, 64)
                pos, D = out["kp_pos"][:1], out["kp_dir"][:1]
                kind = torch.cat([torch.zeros(1, 32), torch.ones(1, 32)], 1)
                nd1 = {k: v[:1] for k, v in nd.items()}
                with FlopCounterMode(display=False) as fc:
                    m.je.attach(tt, pos, D, kind, nd1)
                ts2 = []
                for r in range(20):
                    t0 = time.perf_counter()
                    m.je.attach(tt, pos, D, kind, nd1)
                    ts2.append(time.perf_counter() - t0)
                t3 = []
                for r in range(5):
                    t0 = time.perf_counter()
                    m.nodes(P["feat2"], P["feat1"], P["feat05"], P["road2"], P["centre1"])
                    t3.append(time.perf_counter() - t0)
            print(f"Attach: parameters {sum(p.numel() for p in m.je.attach.parameters())}, {fc.get_total_flops() / 2e6:.2f} MMACs at batch 1 (flop counter / 2; the gating by distance and the gathers are not counted), {1000 * np.median(ts2):.2f} ms on the CPU (2 threads)")
            print(f"B's frozen node head (needed only because this process re-computes it; the deployed graph has it anyway): {1000 * np.median(t3) / 2:.0f} ms per frame on the CPU")
    print(f"whole forward, batch 2, CPU (2 threads): without the attachment {1000 * res[False]:.0f} ms, with it (node head + attachment) {1000 * res[True]:.0f} ms")
elif mode == "kperr_att":
    from importlib.machinery import SourceFileLoader
    import importlib.util
    ld = SourceFileLoader("jdiag", os.path.join(HERE, "dev", "jdiag.pysrc"))
    jd = importlib.util.module_from_spec(importlib.util.spec_from_loader("jdiag", ld))
    ld.exec_module(jd)
    jd.kperr(["e320_smoke_cpu_s0"])
elif mode == "train_smoke":
    os.environ["BFLC_SIG"] = "1.0"
    M = use("dev")
    sys.argv = ["jends_train.py", "--name", "e320_smoke_train_cpu_s0", "--seed", "0", "--maps", os.path.join(ROOT, "runs", "e307_n_s0", "model.pt"), "--e2e", "--scales", "--maps-w", "1.0", "--geom-w", "0.5", "--attach", "--bs", "1", "--workers", "0", "--smoke"]
    TT = M["jends_train"].T

    def fake_fit(model, loss_fn, run, *, steps, loader, **kw):                                         # the trainer needs a GPU: call the loss function of main() 101 times on the CPU (no backward) to run the step-100 log line
        n = 0
        with torch.no_grad():
            for b in loader:
                loss_fn(model, b)
                n += 1
                if n >= 101:
                    break
    TT.fit = fake_fit
    M["jends_train"].main()
    print(open(os.path.join(ROOT, "runs", "e320_smoke_train_cpu_s0", "train.log")).read()[-900:])
elif mode == "selftest_dev":
    from importlib.machinery import SourceFileLoader
    import importlib.util
    M = use("dev")

    def modsrc(name):
        ld = SourceFileLoader(name, os.path.join(HERE, "dev", name + ".pysrc"))
        mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, ld))
        sys.modules[name] = mod
        ld.exec_module(mod)
        return mod
    for n in ("qends_eval", "jmaneuver", "jends_eval", "jdiag"):
        mod = modsrc(n)
        if n == "jdiag":
            mod.selftest()
        elif n == "jends_eval":
            mod.selftest()
        else:
            mod.selftest()
        print(n, "selftest ok")
elif mode == "regress_readout":
    # the read-out modules, ORIGINAL (temp/e320/orig = the committed 7386e0e1) against DEV, on the 8-frame smoke dump: cmd_score for the old read-outs (keypoint ends: proximity / snap / snap-nogeom / snap-all / chord) and jmaneuver for the old variants must give the same numbers
    from importlib.machinery import SourceFileLoader
    from types import SimpleNamespace
    import importlib.util
    import json
    import qgraph_eval as QGE
    orig_vf = QGE.val_frames
    QGE.val_frames = lambda: orig_vf()[:8]
    d = os.path.join(ROOT, "runs", "e320_smoke_cpu_s0")
    f8 = os.path.join(HERE, "full8.pkl")
    if not os.path.exists(f8):
        pickle.dump(pickle.load(open(os.path.join(ROOT, "temp", "e276", "predfull_e273_p_s0.pkl"), "rb"))[:8], open(f8, "wb"))

    def modsrc(folder, name):
        ld = SourceFileLoader(name, os.path.join(HERE, folder, name + ".pysrc"))
        mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, ld))
        sys.modules[name] = mod
        ld.exec_module(mod)
        return mod
    base = dict(run="e320_smoke_cpu_s0", seed=0, thr=0.1, full=f8, base=None, skip_base_check=True, chord=False, snap=False)
    sets = (dict(), dict(snap=True), dict(snap=True, snap_nogeom=True), dict(snap=True, snap_all=True), dict(chord=True), dict(snap=True, chord=True, snap_guard=True))
    res = {}
    for folder in ("orig", "dev"):
        qe = modsrc(folder, "qends_eval")
        for i, kw in enumerate(sets):
            qe.cmd_score(SimpleNamespace(**dict(base, **kw)))
            name = "ends_card_" + ("chord_" if kw.get("chord") else "") + ("snap" + "".join("_" + k[5:] for k in ("snap_all", "snap_guard", "snap_dedup", "snap_nogeom", "snap_both") if kw.get(k)) + "_" if kw.get("snap") else "") + "thr0.1.json"
            c = json.load(open(os.path.join(d, name)))
            res[(folder, i)] = (c["numbers"], c["difference"], c["curves"], c["wired_both_ends"])
        jm = modsrc(folder, "jmaneuver")
        QGE.val_frames = orig_vf                                                                              # jmaneuver reads the full B dump itself: all 873 frames, --max 8 stops it
        jm.score(SimpleNamespace(run="e320_smoke_cpu_s0", seed=0, variants=["plain", "p0.5", "p0.1", "p0.1s", "oracle", "oracle+fit"], max=8, rows=False, chord=False))
        res[(folder, "man")] = json.load(open(os.path.join(d, "maneuver_val.json")))["variants"]
        QGE.val_frames = lambda: orig_vf()[:8]
    for i in range(len(sets)):
        assert res[("orig", i)] == res[("dev", i)], (i, res[("orig", i)], res[("dev", i)])
    assert res[("orig", "man")] == res[("dev", "man")], (res[("orig", "man")], res[("dev", "man")])
    print("regress_readout ok: cmd_score (6 read-outs) and jmaneuver (6 variants) identical between the committed and the dev modules")
elif mode == "readouts_new":
    # the E320 read-outs on the 8-frame smoke dump (dev modules): primary (attached + one nearest piece per end, geometry kept), proximity, tight 1.5 / 1.0; the file names, the extra statistics, jmaneuver p0.1a / an / ar1.5 / ar1.0
    from importlib.machinery import SourceFileLoader
    from types import SimpleNamespace
    import importlib.util
    import json
    import qgraph_eval as QGE
    orig_vf = QGE.val_frames
    f8 = os.path.join(HERE, "full8.pkl")

    def modsrc(name):
        ld = SourceFileLoader(name, os.path.join(HERE, "dev", name + ".pysrc"))
        mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, ld))
        sys.modules[name] = mod
        ld.exec_module(mod)
        return mod
    qe, jm = modsrc("qends_eval"), modsrc("jmaneuver")
    base = dict(run="e320_smoke_cpu_s0", seed=0, thr=0.1, full=f8, base=None, skip_base_check=True, chord=False, snap=False)
    d = os.path.join(ROOT, "runs", "e320_smoke_cpu_s0")
    QGE.val_frames = lambda: orig_vf()[:8]
    for kw in (dict(attach=True), dict(attach=True, snap=True, snap_nogeom=True), dict(attach=True, attach_r=1.5), dict(attach=True, attach_r=1.0), dict()):
        qe.cmd_score(SimpleNamespace(**dict(base, **kw)))
    print(sorted(x for x in os.listdir(d) if x.startswith("ends_card")))
    for nme in ("ends_card_attach_thr0.1.json", "ends_card_snap_nogeom_attach_thr0.1.json", "ends_card_attach_r1.5_thr0.1.json", "ends_card_attach_r1.0_thr0.1.json", "ends_card_thr0.1.json"):
        c = json.load(open(os.path.join(d, nme)))
        print(nme, "topo", round(c["numbers"]["head"]["topo_F1"], 4), "curves", c["curves"], "wired", c["wired_both_ends"], "dropped", c["long_links_dropped"], "within 1 m", {k: round(v, 3) if isinstance(v, float) else v for k, v in c["ends_within_1m_of_a_delivered_piece_end"].items()})
    QGE.val_frames = orig_vf
    jm.score(SimpleNamespace(run="e320_smoke_cpu_s0", seed=0, variants=["plain", "p0.1", "p0.1s", "p0.1a", "p0.1an", "p0.1ar1.5", "p0.1ar1.0"], max=8, rows=False, chord=False))
elif mode == "boot_test":
    # the paired bootstrap of jdiag on the counts that the dev cmd_score wrote for two read-outs of the smoke dump (8 frames, few scenes: a pipeline check)
    from importlib.machinery import SourceFileLoader
    import importlib.util
    ld = SourceFileLoader("jdiag", os.path.join(HERE, "dev", "jdiag.pysrc"))
    jd = importlib.util.module_from_spec(importlib.util.spec_from_loader("jdiag", ld))
    ld.exec_module(jd)
    jd.boot("e320_smoke_cpu_s0", "ends_card_snap_nogeom_attach_thr0.1", "e320_smoke_cpu_s0", "ends_card_attach_thr0.1", n=30)
elif mode == "flops":
    M = use("dev")
    JE = M["jends"]
    from torch.utils.flop_counter import FlopCounterMode
    B = 1
    jout = dict(j1=torch.randn(B, 8, 100, 100) * 0.1 - 2.0, j05=torch.randn(B, 2, 200, 200) - 2.0, f1=torch.randn(B, 64, 100, 100))
    jout["j1"][:, 5], jout["j1"][:, 7] = 5.0, 5.0
    P = dict(centre1=torch.randn(B, 12, 100, 100))
    for unc in (False, True):
        torch.manual_seed(0)
        mm = JE.JEnds(d=64, order=True, evidence=True, scales=True, unc=unc).eval()
        npar = sum(p.numel() for p in mm.parameters())
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            mm(jout, P, gt=None)
        print(f"JEnds scales=True unc={unc}: parameters {npar}, forward {fc.get_total_flops() / 2e6:.2f} MMACs (flop counter / 2; the keypoint tensors, the order bookkeeping and the evidence samples are not counted by the counter)")
