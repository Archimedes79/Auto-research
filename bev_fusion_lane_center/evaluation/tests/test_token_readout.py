"""Tests of `token_readout.py` (the read-out helpers of the token heads); CPU, no data: the hand cases of its selftest (the E366 cell rule, the existence AP, the every-3 subsample of a TRAIN dump) and textual mutants of the module, every one of which must make
the selftest fail by an assertion. `modes` / `ap` / `every3` on real dumps are commands (README row): on the E374 runs they reproduce the numbers of card E374 section 3 (cell 0.1 / 0.6 / off for every arm; AP 0.9539 / 0.9442 of m_s0 / z_s0).

    wsl bash tools/env/_py14.sh test_token_readout.py
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = os.path.join(HERE, "..", "token_readout.py")

MUTANTS = [
    ("dup guard dropped", 'if r["dup_frac"] <= max_dup]', "if True]"),
    ("dup guard strict", 'if r["dup_frac"] <= max_dup]', 'if r["dup_frac"] < max_dup]'),
    ("tie goes to the farthest cell", "key=lambda r: dist(r[1]))[0]", "key=lambda r: dist(r[1]))[-1]"),
    ("gate ignored in the tie distance", "c[2] != rec[2])", "False)"),
    ("record gate off", "REC = (0.3, 0.5, True)", "REC = (0.3, 0.5, False)"),
    ("keep threshold compared first", "dist = lambda c: (abs(c[0] - rec[0]), abs(c[1] - rec[1]), c[2] != rec[2])", "dist = lambda c: (abs(c[1] - rec[1]), abs(c[0] - rec[0]), c[2] != rec[2])"),
    ("best taken as the lowest", "best = max(r[0] for r in ok)", "best = min(r[0] for r in ok)"),
    ("AP of the inverted labels", "average_precision_score(np.asarray(y, bool),", "average_precision_score(~np.asarray(y, bool),"),
    ("every second frame", "m3[k] = v[::3]", "m3[k] = v[::2]"),
    ("frames not subsampled", "return m3, frames[::3]", "return m3, frames"),
    ("every flag missing", 'm3["every"] = 3', 'm3["every"] = 1'),
    ("use lists not cut", "m3[\"use\"] = {k: (v[::3] if", "m3[\"use\"] = {k: (v if"),
    ("sample check removed", 'assert m3["idx"] == sample(ntrain, "train", 3).tolist(),', "assert True,"),
]


def run(src, name):
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "evaluation"))
        p = os.path.join(d, "evaluation", "token_readout.py")
        open(p, "w", encoding="utf-8").write(src)
        env = dict(os.environ, BFLC_ROOT=d, OMP_NUM_THREADS="1")
        r = subprocess.run([sys.executable, p, "selftest"], capture_output=True, text=True, env=env, cwd=d, timeout=300)
    return r


def test_selftest_and_mutants():
    src = open(MODULE, encoding="utf-8").read()
    r = run(src, "original")
    assert r.returncode == 0 and "selftest ok" in r.stdout, r.stdout + r.stderr
    survived = []
    for name, old, new in MUTANTS:
        assert src.count(old) == 1, f"mutant '{name}': the text to change occurs {src.count(old)} times"
        r = run(src.replace(old, new), name)
        if r.returncode == 0 or ("AssertionError" not in r.stderr and "must be refused" not in r.stderr):          # killed = the selftest's own assertion, or its 'accepted although it must be refused' error (not a crash)
            survived.append((name, r.returncode, r.stderr[-200:]))
    assert not survived, f"mutants not killed by an assertion: {survived}"
    print(f"token_readout: selftest ok, {len(MUTANTS)} mutants killed")


if __name__ == "__main__":
    test_selftest_and_mutants()
