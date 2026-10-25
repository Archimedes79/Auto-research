"""Clean up the scratch of a finished loop. Plain python 3, no GPU.

    wsl python3 tools/maintenance/clean_temp.py loop18             # dry run: what would go, and what live code still imports from it
    wsl python3 tools/maintenance/clean_temp.py loop18 --move      # move it to _to_delete/temp_loop18/ (reversible)
    wsl python3 tools/maintenance/clean_temp.py loop18 --delete    # delete it for good (scratch only)
    wsl python3 tools/maintenance/clean_temp.py --selftest

Convention: scratch of a loop lives in temp/loopNN/ (also found: temp/analysis/loopNN, temp/launchers/loopNN). Run this when the loop is filed
(results in experiments/, report built, ledger and state files written). It refuses to touch a folder when
  - training/, evaluation/, tools/ or reports/build/ import a python module that exists only there (promote it first), or
  - a file in it changed in the last 6 hours (a job may still use it; --force overrides).
"""
import os
import re
import shutil
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEMP = os.path.join(ROOT, "temp")
LIVE = ["training", "evaluation", "tools", "reports/build"]
IMP = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)


def find(name, temp=TEMP):
    out = []
    for base in (temp, os.path.join(temp, "analysis"), os.path.join(temp, "launchers")):
        p = os.path.join(base, name)
        if os.path.isdir(p):
            out.append(p)
    return out


def modules_in(folders):
    mods = {}
    for f in folders:
        for dp, _, fn in os.walk(f):
            for x in fn:
                if x.endswith(".py"):
                    mods.setdefault(x[:-3], os.path.join(dp, x))
    return mods


def live_imports(mods, root=ROOT, live=LIVE):
    hits = {}
    other = set()
    for d in live:
        for dp, dn, fn in os.walk(os.path.join(root, d)):
            dn[:] = [x for x in dn if x != "__pycache__"]
            for x in fn:
                if x.endswith(".py"):
                    other.add(x[:-3])
    for d in live:
        for dp, dn, fn in os.walk(os.path.join(root, d)):
            dn[:] = [x for x in dn if x != "__pycache__"]
            for x in fn:
                if x.endswith(".py"):
                    t = open(os.path.join(dp, x), encoding="utf-8", errors="ignore").read()
                    for m in set(IMP.findall(t)):
                        if m in mods and m not in other:
                            hits.setdefault(m, []).append(os.path.join(dp, x)[len(root) + 1:])
    return hits


def newest(folders):
    best = 0
    for f in folders:
        for dp, _, fn in os.walk(f):
            for x in fn:
                best = max(best, os.path.getmtime(os.path.join(dp, x)))
    return best


def run(name, move=False, delete=False, force=False, root=ROOT):
    temp = os.path.join(root, "temp")
    folders = find(name, temp)
    if not folders:
        print(f"nothing named {name} under temp/")
        return 0
    n = sum(len(fn) for f in folders for _, _, fn in os.walk(f))
    print(f"{name}: {n} files in " + ", ".join(os.path.relpath(f, root) for f in folders))
    hits = live_imports(modules_in(folders), root)
    for m, users in sorted(hits.items()):
        print(f"  BLOCKED: module {m} is imported by {users[:3]} - copy it to training/ or evaluation/ first")
    age_h = (time.time() - newest(folders)) / 3600
    if age_h < 6 and not force:
        print(f"  BLOCKED: newest file is {age_h:.1f} h old (a job may use it); --force overrides")
        return 1
    if hits:
        return 1
    if not (move or delete):
        print("dry run: nothing changed (--move or --delete)")
        return 0
    for f in folders:
        if delete:
            shutil.rmtree(f)
            print("deleted", os.path.relpath(f, root))
        else:
            dst = os.path.join(root, "_to_delete", f"temp_{name}", os.path.relpath(f, temp))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(f, dst)
            print("moved", os.path.relpath(f, root), "->", os.path.relpath(dst, root), "(add a line to _to_delete/README.md)")
    return 0


def selftest():
    import tempfile
    root = tempfile.mkdtemp()
    for d in ("training", "evaluation", "tools", "reports/build", "temp/loop99", "_to_delete"):
        os.makedirs(os.path.join(root, d))
    open(os.path.join(root, "temp/loop99/probe.py"), "w").write("x = 1\n")
    open(os.path.join(root, "training/user.py"), "w").write("import probe\n")
    old = time.time() - 48 * 3600
    os.utime(os.path.join(root, "temp/loop99/probe.py"), (old, old))
    assert run("loop99", move=True, root=root) == 1, "must refuse while live code imports it"
    assert os.path.exists(os.path.join(root, "temp/loop99/probe.py"))
    os.remove(os.path.join(root, "training/user.py"))
    assert run("loop99", root=root) == 0 and os.path.exists(os.path.join(root, "temp/loop99"))  # dry run
    assert run("loop99", move=True, root=root) == 0
    assert not os.path.exists(os.path.join(root, "temp/loop99"))
    assert os.path.exists(os.path.join(root, "_to_delete/temp_loop99/loop99/probe.py"))
    shutil.rmtree(root)
    print("clean_temp selftest passed")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a:
        selftest()
    elif not a or a[0].startswith("-"):
        print(__doc__)
    else:
        sys.exit(run(a[0], "--move" in a, "--delete" in a, "--force" in a))
