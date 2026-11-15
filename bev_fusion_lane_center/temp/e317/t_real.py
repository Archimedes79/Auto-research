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
