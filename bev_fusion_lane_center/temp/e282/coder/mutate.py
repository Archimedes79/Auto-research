"""scratch (E282 coder): mutation check of evaluation/zone_metrics.py on a PRIVATE copy (temp/e282/coder/mut/): every mutant must make test_zone_metrics.py fail.
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 bash tools/env/_py14.sh mutate.py"""
import os
import shutil
import subprocess
import sys

ROOT = os.environ["BFLC_ROOT"]
SRC = os.path.join(ROOT, "evaluation", "zone_metrics.py")
TST = os.path.join(ROOT, "evaluation", "tests", "test_zone_metrics.py")
MUT = os.path.join(ROOT, "temp", "e282", "coder", "mut")

TREE = """        d, i = cKDTree(pos).query(P)
        zone = np.where(d <= ZONE_TOL, zone_of[i], -1)
"""
MUTANTS = {
    "membership by cell instead of position": (TREE, """        cz = {tuple(c): zone_of[m] for m, c in enumerate(zs["nd"]["cell"])}
        zone = np.array([cz.get(tuple(np.clip(np.floor((p + 50.0) / 2.0).astype(int), 0, 49)), -1) for p in P], int)
"""),
    "any in-zone dense node within 2 m instead of the nearest dense node": (TREE, """        tr = cKDTree(pos)
        for k, p in enumerate(P):
            zz = [zone_of[j] for j in tr.query_ball_point(p, ZONE_TOL) if zone_of[j] >= 0]
            zone[k] = zz[0] if zz else -1
"""),
    "tree on the sparse GT nodes only": (TREE, """        d, i = cKDTree(pos[zs["gt_node"]]).query(P)
        zone = np.where(d <= ZONE_TOL, zone_of[np.asarray(zs["gt_node"])[i]], -1)
"""),
    "exact as >= 1": ("free_exact=int((n[free] == 1).sum())", "free_exact=int((n[free] >= 1).sum())"),
    "fixed exact as >= 1": ("fixed_exact=int((n[fixed] == 1).sum())", "fixed_exact=int((n[fixed] >= 1).sum())"),
    "wrong length (half)": ("dens=[len(P), float(ref_len)]", "dens=[len(P), float(ref_len) / 2]"),
    "density per 10 m instead of 100 m": ("nodes_per_100m=100.0 * r(", "nodes_per_100m=10.0 * r("),
    "off_zone over the in-zone nodes": ('off_zone=r(c["n_off"], c["n_deliv"])', 'off_zone=r(c["n_off"], c["n_deliv"] - c["n_off"])'),
    "radius < instead of <=": ("d <= ZONE_TOL", "d < ZONE_TOL"),
    "radius 2.5 m": ("ZONE_TOL = 2.0 ", "ZONE_TOL = 2.5 "),
    "free and fixed swapped": ("free, fixed = kind == 1, kind == 0", "free, fixed = kind == 0, kind == 1"),
    "pool = mean of the frame ratios": ("    return _numbers(_table(cs).sum(0))", "    return {k: float(np.nanmean([_numbers(r)[k] for r in _table(cs)])) for k in _numbers(_table(cs).sum(0))}"),
    "bootstrap not paired": ("d = np.array([_numbers(ta[i].sum(0))[metric] - _numbers(tb[i].sum(0))[metric] for i in idx])",
                             "j = np.random.RandomState(seed + 1).randint(0, len(ta), size=(n, len(ta)))\n    d = np.array([_numbers(ta[i].sum(0))[metric] - _numbers(tb[k].sum(0))[metric] for i, k in zip(idx, j)])"),
    "matched threshold with e > thr": ('side="left"', 'side="right"'),
    "matched threshold grid not rounded": ("thr = np.round(np.asarray(grid, float), 9)", "thr = np.asarray(grid, float)"),
    "matched threshold: ties to the highest threshold": ("i = int(np.argmin(np.abs(dens - target)))", "i = len(dens) - 1 - int(np.argmin(np.abs(dens - target)[::-1]))"),
}


def run(name, old, new):
    src = open(SRC).read()
    assert src.count(old) == 1, (name, src.count(old))
    shutil.rmtree(MUT, ignore_errors=True)
    os.makedirs(os.path.join(MUT, "tests"))
    open(os.path.join(MUT, "zone_metrics.py"), "w").write(src.replace(old, new))
    t = open(TST).read()
    old_path = 'for sub in ("..", "../../training/grid"):\n    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub)))'
    assert old_path in t
    new_path = f'for sub in ("{ROOT}/evaluation", "{ROOT}/training/grid", ".."):\n    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub)))'
    open(os.path.join(MUT, "tests", "test_zone_metrics.py"), "w").write(t.replace(old_path, new_path))
    code = ("import test_zone_metrics as T, traceback\nbad = []\n"
            "for n in [k for k in dir(T) if k.startswith('test_')]:\n    try:\n        getattr(T, n)()\n    except BaseException:\n        bad.append(n)\n"
            "print('FAILED:', ' '.join(bad)); raise SystemExit(1 if bad else 0)")
    p = subprocess.run([sys.executable, "-c", code], cwd=os.path.join(MUT, "tests"), capture_output=True, text=True)
    return p.returncode, (p.stdout.strip().splitlines() or [""])[-1][:200] if p.returncode else ""


if __name__ == "__main__":
    only = sys.argv[1:]
    survived = []
    for name, (old, new) in MUTANTS.items():
        rc, msg = run(name, old, new)
        print(("killed  " if rc else "SURVIVED"), name, "|", msg, flush=True)
        if not rc:
            survived.append(name)
    rc, msg = run("identity", "ZONE_TOL = 2.0 ", "ZONE_TOL = 2.0 ")
    print("identity (the copy itself passes):", "yes" if rc == 0 else "NO " + msg)
    shutil.rmtree(MUT, ignore_errors=True)
    print("survivors:", survived)
