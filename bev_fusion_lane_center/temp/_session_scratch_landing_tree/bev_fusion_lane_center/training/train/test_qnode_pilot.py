"""E282 tests of the CPU mini-pilot runner (`qnode_pilot.py`, CPU, no model, no lane GT): the outcome logic of the gate against card section 3 (INVALID / INCONCLUSIVE-control / S-PASSES / GO / STOP-A /
STOP-B on fake result files, incl. the boundaries), the arms' targets from a synthetic cache (arm d reads the DENSE maps, the zone arms also the zone maps), and the run folder naming.
Run: python test_qnode_pilot.py   (`_py14.sh test_qnode_pilot.py`)"""
import contextlib
import io
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training/train", "training/grid", "training/data", "training", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qnode_pilot as QP  # noqa: E402
import zone_metrics as ZM  # noqa: E402

TMP = tempfile.mkdtemp(prefix="qnode_pilot_test_")
QP.ROOT = TMP
os.makedirs(os.path.join(TMP, "temp", "e282"), exist_ok=True)


def counts(nf=110, rec=0.9, exact=0.5, frec=0.7, off=0.1, dens=14.0, seed=0, free=True):
    """fake per-frame zone counts with the given rates (binomial draws)"""
    r = np.random.RandomState(seed)
    cs = []
    for _ in range(nf):
        n_free, n_fixed = (r.randint(40, 80) if free else 0), r.randint(20, 40)
        length = 100.0 + 50 * r.rand()
        fr = r.binomial(n_free, rec)
        fe = r.binomial(fr, min(1.0, exact / max(rec, 1e-9)))
        xr = r.binomial(n_fixed, frec)
        nd = r.poisson(dens * length / 100)
        cs.append(dict(n_free=int(n_free), n_fixed=int(n_fixed), free_rec=int(fr), free_exact=int(fe), fixed_rec=int(xr), fixed_exact=int(xr // 2), n_deliv=int(nd), n_off=int(r.binomial(nd, off)),
                       dens=[int(nd), float(length)]))
    return cs


def arm(name, at05, atm, prec=0.9, rec=0.8, win=None, tm=0.45):
    return dict(arm=name, steps=2000, seed=0, tag="", frames=110, t_match=tm, dens_match_train=14.4, t_match_ok=True,
                **{"at_0.5": dict(t=0.5, pool=ZM.pool(at05), counts=at05), "at_match": dict(t=tm, pool=ZM.pool(atm), counts=atm)},
                dense_node_gate=dict(precision=prec, recall=rec, delivered=100, gt=100), **({} if win is None else dict(win_off_gt_end=win)))


def run_gate(label, d, s, m, expect, m5=None):
    for k, v in (("d", d), ("s", s), ("m", m)) + ((("m5", m5),) if m5 else ()):
        p = os.path.join(TMP, "runs", QP.run_name(k, 2000, ""))
        os.makedirs(p, exist_ok=True)
        json.dump(v, open(os.path.join(p, "result.json"), "w"), default=float)

    class A:
        steps, tag, cache = 2000, "", ""

    with contextlib.redirect_stdout(io.StringIO()):
        o = QP.gate(A)
    assert o["outcome"] == expect, (label, o["outcome"], expect, o["floors"], o["s_passes"], o["bootstrap_m_minus_s_matched"]["lo90"])
    return o


D = lambda **k: arm("d", counts(rec=0.92, exact=0.06, frec=0.75, off=0.12, dens=31, seed=1), counts(seed=2), **k)  # noqa: E731
S_BAD = arm("s", counts(rec=0.1, exact=0.03, frec=0.2, off=0.1, dens=3, seed=3), counts(rec=0.5, exact=0.15, frec=0.5, off=0.12, dens=14, seed=4))
S_GOOD = arm("s", counts(seed=5), counts(rec=0.88, exact=0.7, frec=0.7, off=0.1, dens=14, seed=6))


def M(**k):
    return arm("m", counts(rec=k.pop("rec", 0.88), exact=k.pop("exact", 0.65), frec=k.pop("frec", 0.55), off=k.pop("off", 0.1), dens=k.pop("dens", 14), seed=7), counts(seed=8), win=k.pop("win", 0.5), **k)


def test_gate_outcomes():
    run_gate("d invalid: precision 0.7", D(prec=0.7), S_BAD, M(), "INVALID")
    run_gate("d invalid: recall 0.49", D(rec=0.49), S_BAD, M(), "INVALID")
    run_gate("d valid at exactly 0.75 / 0.5", D(prec=0.75, rec=0.5), S_BAD, M(), "GO")
    run_gate("s passes the floors at matched density", D(), S_GOOD, M(), "S-PASSES")
    o = run_gate("GO", D(), S_BAD, M(), "GO")
    assert all(o["floors"].values()) and o["valid"] and not o["s_passes"] and o["s_matched_density_ok"]
    run_gate("STOP-B: the gain over s is missing (s is as good as m)", D(), arm("s", counts(seed=5), counts(rec=0.88, exact=0.45, frec=0.5, off=0.14, dens=14, seed=6)), M(exact=0.55), "STOP-B")
    run_gate("STOP-B: the winners stay on the GT cell", D(), S_BAD, M(win=0.1), "STOP-B")
    run_gate("STOP-B: no winner share recorded", D(), S_BAD, M(win=None), "STOP-B")
    run_gate("STOP-A: free-zone recall low", D(), S_BAD, M(rec=0.6, exact=0.3), "STOP-A")
    run_gate("STOP-A: density 25", D(), S_BAD, M(dens=25), "STOP-A")
    run_gate("STOP-A: off-zone 0.2", D(), S_BAD, M(off=0.2), "STOP-A")
    run_gate("STOP-A: fixed-zone recall low", D(), S_BAD, M(frec=0.3), "STOP-A")
    print("test_gate_outcomes ok")


def test_control_not_matched():
    far = arm("s", counts(seed=5), counts(rec=0.5, exact=0.15, frec=0.5, off=0.12, dens=25, seed=6))                        # s delivers 25 nodes per 100 m on the VAL frames at its fitted threshold
    o = run_gate("control not at the matched density", D(), far, M(), "INCONCLUSIVE-control")
    assert not o["s_matched_density_ok"]
    nan_s = arm("s", counts(seed=5), counts(rec=0.5, exact=0.15, frec=0.5, off=0.12, dens=14.4, seed=6, free=False))        # no free zone: the control's free-zone numbers are undefined
    run_gate("a NaN metric of the control", D(), nan_s, M(), "INCONCLUSIVE-control")
    o = run_gate("m5 floors are computed when m5 is given", D(), S_BAD, M(), "GO", m5=arm("m5", counts(rec=0.4, exact=0.1, frec=0.5, off=0.1, dens=14, seed=9), counts(seed=8), win=0.0))
    assert "floors_m5" in o and not all(o["floors_m5"].values())
    print("test_control_not_matched ok")


def test_cache_targets_and_names():
    d = QP.cache_dir("unit")
    os.makedirs(d, exist_ok=True)
    N = 3
    shapes = dict(feat2=(2, 5, 5), feat1=(2, 10, 10), feat05=(2, 20, 20), road2=(2, 5, 5), centre1=(12, 10, 10))
    for split in ("train", "val"):
        for n, sh in shapes.items():
            mm = np.lib.format.open_memmap(os.path.join(d, f"{split}_{n}.npy"), mode="w+", dtype=np.float16, shape=(N,) + sh)
            mm[:] = 1.0
            mm.flush()
        t = dict(tokens=np.array(["a", "b", "c"]), lens=np.array([100.0, 110.0, 120.0]))
        for k in QP.SPARSE_KEYS:
            t[k] = np.full((N, 3, 4), 1.0, np.float32)                                      # the sparse maps hold 1
            t["d_" + k] = np.full((N, 3, 4), 2.0, np.float32)                               # the dense maps hold 2
        for k in QP.ZONE_KEYS:
            t[k] = np.full((N, 3, 4), 7.0, np.float32)
        np.savez_compressed(os.path.join(d, f"{split}_targets.npz"), **t)
    json.dump(dict(train=dict(signature="x"), val=dict(signature="x")), open(os.path.join(d, "meta.json"), "w"))
    c = QP.Cache("unit", "train")
    idx = np.array([0, 2])
    bd, bs, bm = c.batch(idx, "d"), c.batch(idx, "s"), c.batch(idx, "m")
    assert all(float(bd[k].mean()) == 2.0 for k in QP.SPARSE_KEYS), "arm d reads the DENSE maps"
    assert all(float(bs[k].mean()) == 1.0 for k in QP.SPARSE_KEYS), "arm s reads the sparse maps"
    assert not any(k in bd or k in bs for k in QP.ZONE_KEYS) and all(k in bm and float(bm[k].mean()) == 7.0 for k in QP.ZONE_KEYS), "only the zone arms carry the zone maps"
    assert all(float(bm[k].mean()) == 1.0 for k in QP.SPARSE_KEYS) and bm["feat2"].shape == (2,) + shapes["feat2"] and bm["feat2"].dtype.is_floating_point
    assert QP.run_name("m", 2000) != QP.run_name("m", 4000) and QP.run_name("m", 20, "smoke").endswith("_smoke") and "2000" in QP.run_name("d", 2000), "the steps are part of the run folder name"
    assert QP.ARMS["d"]["dense"] and not QP.ARMS["s"]["dense"] and QP.ARMS["m"]["bias"] == 0.05 and QP.ARMS["m5"]["bias"] == 0.5 and QP.ARMS["m"]["loss"] == "zone"
    print("test_cache_targets_and_names ok")


if __name__ == "__main__":
    test_gate_outcomes()
    test_control_not_matched()
    test_cache_targets_and_names()
    print("test_qnode_pilot: all ok")
