"""E347 scratch: run a project script or test with the STAGED copies of the frozen files in front of the real ones (the lane code digest of E343 is frozen, nothing under training/ evaluation/ changes until the TEST use ends).

    wsl bash temp/e347/run_staged.sh <script.py | basename> [args ...]       (the wrapper builds the environment of tools/env/_py14.sh and calls this file)

BFLC_STAGE = the stage root (default temp/_session_scratch_e347/stage, files at their project paths: training/grid/jmaps.py ...); BFLC_STAGE=none = the real files only (control run, the unfixed code).
The flat module names the project uses (`import jmaps`, `import jends`, ...) are redirected by a meta path finder that sits BEFORE every sys.path entry, so the `sys.path.insert(0, <real folder>)` lines of the trainers and
tests cannot bring the real copy back. At exit the staged modules that were loaded are printed with their files, and a module that has a staged copy but was loaded from elsewhere is an error.
A script is looked up like `_py14.sh` does (training evaluation tools reports/build temp, not `_session_scratch_`), a staged copy first; cwd = the folder of the script.
"""
import atexit
import importlib.abc
import importlib.util
import os
import runpy
import sys

ROOT = os.environ["BFLC_ROOT"]
STAGE = os.environ.get("BFLC_STAGE") or os.path.join(ROOT, "temp", "_session_scratch_e347", "stage")
INDEX = {}
if STAGE != "none":
    for dp, dn, fn in os.walk(STAGE):
        for f in fn:
            if f.endswith(".py"):
                name = f[:-3]
                assert name not in INDEX, f"two staged files named {f}: {INDEX[name]} and {os.path.join(dp, f)}"
                INDEX[name] = os.path.join(dp, f)


class StageFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        p = INDEX.get(name)
        return None if p is None else importlib.util.spec_from_file_location(name, p)


sys.meta_path.insert(0, StageFinder())


def resolve(s):
    if os.path.isfile(s):
        p = os.path.abspath(s)
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        staged = os.path.join(STAGE, rel)                                                       # a project path that has a staged copy runs the staged one
        return staged if STAGE != "none" and not rel.startswith("..") and os.path.isfile(staged) else p
    base = os.path.basename(s)
    if base[:-3] in INDEX and (s == base or INDEX[base[:-3]].endswith(s.replace("\\", "/").lstrip("./"))):
        return INDEX[base[:-3]]
    hits = []
    for sub in ("training", "evaluation", "tools", "reports/build", "temp"):
        for dp, dn, fn in os.walk(os.path.join(ROOT, sub)):
            dn[:] = [d for d in dn if "_session_scratch_" not in d and d != "__pycache__"]
            if base in fn and (s == base or os.path.join(dp, base).replace("\\", "/").endswith("/" + s.replace("\\", "/"))):
                hits.append(os.path.join(dp, base))
    assert len(hits) == 1, f"{s}: {len(hits)} candidates {hits}"
    return hits[0]


def report():
    rel = lambda p: os.path.relpath(p, ROOT).replace("\\", "/")  # noqa: E731
    used, wrong = [], []
    for name, p in INDEX.items():
        m = sys.modules.get(name)
        f = getattr(m, "__file__", None) if m is not None else None
        if f is None:
            continue
        (used if os.path.realpath(f) == os.path.realpath(p) else wrong).append((name, f))
    print("run_staged: stage =", "none (the real files)" if STAGE == "none" else rel(STAGE), "| staged modules loaded:", ", ".join(f"{n} -> {rel(INDEX[n])}" for n, _ in used) or "-", file=sys.stderr)
    assert not wrong, f"modules with a staged copy that were loaded from elsewhere: {wrong}"


if __name__ == "__main__":
    script = resolve(sys.argv[1])
    print("run_staged:", os.path.relpath(script, ROOT).replace("\\", "/"), file=sys.stderr)
    atexit.register(report)
    os.chdir(os.path.dirname(script))
    sys.argv = [script] + sys.argv[2:]
    sys.path.insert(0, os.path.dirname(script))
    runpy.run_path(script, run_name="__main__")
