import os, sys, json, tempfile, io, contextlib
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
for sub in ("training/train", "training/grid", "training/data", "training", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qnode_pilot as QP
import zone_metrics as ZM

tmp = tempfile.mkdtemp(prefix="gate_")
QP.ROOT = tmp
os.makedirs(os.path.join(tmp, "temp", "e282"), exist_ok=True)
rng = np.random.RandomState(0)

def counts(nf=110, rec=0.9, exact=0.5, frec=0.7, off=0.1, dens=14.0, seed=0):
    r = np.random.RandomState(seed)
    cs = []
    for _ in range(nf):
        n_free, n_fixed = r.randint(40, 80), r.randint(20, 40)
        length = 100.0 + 50 * r.rand()
        fr = r.binomial(n_free, rec); fe = r.binomial(fr, min(1.0, exact / max(rec, 1e-9)))
        xr = r.binomial(n_fixed, frec)
        nd = r.poisson(dens * length / 100)
        cs.append(dict(n_free=int(n_free), n_fixed=int(n_fixed), free_rec=int(fr), free_exact=int(fe), fixed_rec=int(xr), fixed_exact=int(xr // 2), n_deliv=int(nd), n_off=int(r.binomial(nd, off)), dens=[int(nd), float(length)]))
    return cs

def arm(name, at05, atm, prec=0.9, rec=0.8, win=None, tm=0.45):
    return dict(arm=name, steps=2000, seed=0, tag="", frames=110, t_match=tm, dens_match_train=14.4, t_match_ok=True,
                **{"at_0.5": dict(t=0.5, pool=ZM.pool(at05), counts=at05), "at_match": dict(t=tm, pool=ZM.pool(atm), counts=atm)},
                dense_node_gate=dict(precision=prec, recall=rec, delivered=100, gt=100), **({} if win is None else dict(win_off_gt_end=win)))

def run(label, d, s, m, expect):
    for k, v in (("d", d), ("s", s), ("m", m)):
        p = os.path.join(tmp, "runs", f"e282_p_{k}")
        os.makedirs(p, exist_ok=True)
        json.dump(v, open(os.path.join(p, "result.json"), "w"), default=float)
    class A: tag = ""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        QP.gate(A)
    o = json.load(open(os.path.join(tmp, "temp", "e282", "pilot_outcome.json")))
    flag = "ok " if o["outcome"] == expect else "MISMATCH"
    print(f"{flag} {label:60s} -> {o['outcome']:9s} (expected {expect}) floors {o['floors']} s_passes {o['s_passes']} boot {o['bootstrap_m_minus_s_matched']['lo90']:.3f}")
    return o

D = lambda **k: arm("d", counts(rec=0.92, exact=0.06, frec=0.75, off=0.12, dens=31, seed=1), counts(seed=2), **k)
S_bad = arm("s", counts(rec=0.1, exact=0.03, frec=0.2, off=0.1, dens=3, seed=3), counts(rec=0.5, exact=0.15, frec=0.5, off=0.12, dens=14, seed=4))
S_good = arm("s", counts(seed=5), counts(rec=0.88, exact=0.7, frec=0.7, off=0.1, dens=14, seed=6))
M = lambda **k: arm("m", counts(rec=k.pop("rec", 0.88), exact=k.pop("exact", 0.65), frec=k.pop("frec", 0.55), off=k.pop("off", 0.1), dens=k.pop("dens", 14), seed=7), counts(seed=8), win=k.pop("win", 0.5), **k)
run("d invalid (prec 0.7)", D(prec=0.7), S_bad, M(), "INVALID")
run("d invalid (rec 0.49)", D(rec=0.49), S_bad, M(), "INVALID")
run("d valid at exactly 0.75/0.5", D(prec=0.75, rec=0.5), S_bad, M(), "GO")
run("s passes", D(), S_good, M(), "S-PASSES")
run("GO", D(), S_bad, M(), "GO")
run("STOP-B gain fails (m exact 0.52 vs s 0.15)? use s better", D(), arm("s", counts(seed=5), counts(rec=0.88, exact=0.45, frec=0.5, off=0.14, dens=14, seed=6)), M(exact=0.55), "STOP-B")
run("STOP-B winners stay on GT", D(), S_bad, M(win=0.1), "STOP-B")
run("STOP-A recall low", D(), S_bad, M(rec=0.6, exact=0.3), "STOP-A")
run("STOP-A density 25", D(), S_bad, M(dens=25), "STOP-A")
run("STOP-A off-zone 0.2", D(), S_bad, M(off=0.2), "STOP-A")
run("STOP-A fixed recall low", D(), S_bad, M(frec=0.3), "STOP-A")
run("win None", D(), S_bad, M(win=None), "STOP-B")
# NaN: s with zero free zones
nan_s = arm("s", counts(seed=5), [dict(n_free=0, n_fixed=0, free_rec=0, free_exact=0, fixed_rec=0, fixed_exact=0, n_deliv=0, n_off=0, dens=[0, 100.0]) for _ in range(110)])
run("NaN in s matched metrics", D(), nan_s, M(), "STOP-A")
