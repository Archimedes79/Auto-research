"""Tests of the predicted-node cache (`data/lane_tokcache.py`) and its loader (`data/lane_ptok_data.py`).

    wsl bash tools/env/_py14.sh test_lane_tokcache.py            # the selftests of both modules, then the hand mutants of the label and selection code: every mutant must make `lane_tokcache.py selftest` FAIL
    wsl bash tools/env/_py14.sh test_lane_tokcache.py selftests  # the selftests only (fast)

A mutant = one textual change of the source (a wrong assignment, a swapped link direction, the contraction removed, ...); the mutated copy runs in a temporary folder with BFLC_ROOT pointing at this project.
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "training", "data", "lane_tokcache.py")

MUTANTS = [
    ("tokens are taken in reverse order (not by descending existence)", "for t in range(min(K, near.shape[0])):", "for t in reversed(range(min(K, near.shape[0]))):"),
    ("a token takes the FARTHEST near node", "g = int(np.argmin(np.where(cand, d[t], np.inf)))", "g = int(np.argmax(np.where(cand, d[t], -np.inf)))"),
    ("a node can be assigned twice", "cand = near[t] & ~used", "cand = near[t]"),
    ("the heading cosine is ignored", "& (d <= r) & (cs >= cos), d", "& (d <= r), d"),
    ("the radius is exclusive", "& (d <= r) & (cs >= cos), d", "& (d < r) & (cs >= cos), d"),
    ("the contraction is removed (every GT node counts as covered)", "sub = sub_table(table, covered)", "sub = sub_table(table, np.ones(M, bool)); covered = np.ones(M, bool)"),
    ("the link direction is swapped", "a, b = int(kept[sub[\"src\"][l]]), int(kept[sub[\"dst\"][l]])", "b, a = int(kept[sub[\"src\"][l]]), int(kept[sub[\"dst\"][l]])"),
    ("a token is its own target", "            if j == i:\n                continue\n", ""),
    ("a pair of two classes keeps the short one", "if plong[(i, int(j))] or not lg:", "if not plong[(i, int(j))] or lg:"),
    ("the type priority is reversed (a lane flag beats fork)", "for code, cols in ((6, (4, 5, 6, 7)), (5, (1,)), (4, (0,)), (3, (3,)), (2, (2,))):", "for code, cols in ((2, (2,)), (3, (3,)), (4, (0,)), (5, (1,)), (6, (4, 5, 6, 7))):"),
    ("the tokens are sorted ascending", "descending=True, stable=True", "descending=False, stable=True"),
    ("the active threshold is 0.5", "g1(ep) > P_ACTIVE", "g1(ep) > 0.5"),
    ("the probabilities come from the logits of the gathered tokens only (no change expected on the toy: kept for the dtype check)", "e_p=g1(ep)", "e_p=g1(ep).half().float()"),
    ("the fixture tolerates a difference", "                if v != 0.0:\n                    bad.append((i, name))", "                if v > 1e-3:\n                    bad.append((i, name))"),
    ("the fixture does not compare the headings", '("dirv", d["dirv"][k], f["dirv"][i, j]), ', ""),
    ("the gate is read from the wrong pair", "g_ = bool(allowed[i, j]) if allowed is not None else False", "g_ = bool(allowed[j, i]) if allowed is not None else False"),
    ("the stats use the stored K for every budget", "np.array([near[:min(k, T)].any(0).sum() for k in ks], np.int32)", "np.array([near[:K].any(0).sum() for k in ks], np.int32)"),
    ("gt_out counts the contracted links", "gt_out=np.int32(len(src) - both(covered))", "gt_out=np.int32(len(sub[\"src\"]))"),
]


def run_selftests():
    for mod in ("lane_tokcache.py", "lane_ptok_data.py"):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "training", "data", mod), "selftest"], capture_output=True, text=True, env=dict(os.environ, BFLC_ROOT=ROOT))
        assert r.returncode == 0, f"{mod} selftest failed:\n{r.stdout[-800:]}{r.stderr[-1200:]}"
        print(r.stdout.strip().splitlines()[-1])


def run_mutants():
    src = open(SRC, encoding="utf-8").read()
    survived = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, a, b in MUTANTS:
            if src.count(a) != 1:
                print(f"NOT APPLIED  {name}: {src.count(a)} matches")
                survived.append(name)
                continue
            p = os.path.join(tmp, "lane_tokcache.py")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(src.replace(a, b))
            r = subprocess.run([sys.executable, p, "selftest"], capture_output=True, text=True, env=dict(os.environ, BFLC_ROOT=ROOT))
            print(("killed       " if r.returncode != 0 else "SURVIVED     ") + name)
            if r.returncode == 0:
                survived.append(name)
    print(f"{len(MUTANTS) - len(survived)} of {len(MUTANTS)} killed")
    return survived


if __name__ == "__main__":
    run_selftests()
    if len(sys.argv) < 2 or sys.argv[1] != "selftests":
        left = run_mutants()
        assert not left, f"mutants survived: {left}"
    print("test_lane_tokcache ok")
