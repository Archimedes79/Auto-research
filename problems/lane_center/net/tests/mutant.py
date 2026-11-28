"""A textual mutant in place of its module: the mutated copy of a module of the package is built from its text and registered under the module's
dotted name (and as its package's attribute) before it runs, so every importer gets the mutant, also one that imports it back; everything else
comes from the package itself. Used by the tests that run mutants in a child process (test_lane_tokcache.py, test_lane_net.py) and, in the
process, by test_jlive_train.py (`module`).

    python -P -m lane_center.net.tests.mutant <module> <mutated file> [args]                    the copy runs as the program (__main__) and is the module
    python -P -m lane_center.net.tests.mutant <module> <mutated file> --main <program> [args]   the copy is the module, the module <program> runs
    python -P -m lane_center.net.tests.mutant selftest"""
import importlib
import os
import runpy
import subprocess
import sys
import tempfile
import types


def module(name, path, run_name=None):
    """the module built from the text of the file `path`, registered in sys.modules under `name` (and, for a dotted name, as its package's
    attribute) before its code runs; run_name: the __name__ its code sees (`__main__`: it runs as the program)"""
    mod = types.ModuleType(run_name or name)
    mod.__file__ = path
    mod.__package__ = name.rpartition(".")[0]
    sys.modules[name] = mod
    if run_name:
        sys.modules[run_name] = mod
    if "." in name:
        parent, _, leaf = name.rpartition(".")
        setattr(importlib.import_module(parent), leaf, mod)
    with open(path, encoding="utf-8") as fh:
        code = compile(fh.read(), path, "exec")
    exec(code, mod.__dict__)
    return mod


def run(argv):
    name, path, rest = argv[0], argv[1], argv[2:]
    if rest[:1] == ["--main"]:
        sys.argv = [rest[1]] + rest[2:]
        module(name, path)
        runpy.run_module(rest[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = [path] + rest
        module(name, path, run_name="__main__")


def selftest():
    """a copy stands in for its module (an importer gets it); an unmutated copy run as the program passes; a mutated one fails"""
    paths = "lane_center.training.bflc_paths"
    real = importlib.import_module(paths)
    with tempfile.TemporaryDirectory() as tmp:
        same, bad = os.path.join(tmp, "same.py"), os.path.join(tmp, "bad.py")
        src = open(real.__file__, encoding="utf-8").read()
        old = "    return os.path.join(ROOT, p)\n"
        assert src.count(old) == 1, old
        open(same, "w", encoding="utf-8").write(src)
        open(bad, "w", encoding="utf-8").write(src.replace(old, "    return p\n"))
        out = [subprocess.run([sys.executable, "-P", "-m", __spec__.name, paths, p], capture_output=True, text=True) for p in (same, bad)]
        assert out[0].returncode == 0 and "bflc_paths ok" in out[0].stdout, out[0].stderr[-800:]
        assert out[1].returncode != 0 and "AssertionError" in out[1].stderr, (out[1].returncode, out[1].stderr[-800:])
        mod = module(paths, bad)
        lane_train = importlib.import_module("lane_center.training.train.lane_train")
        assert lane_train.BP is mod is sys.modules["lane_center.training"].bflc_paths and mod.resolve("x") == "x", "the importer does not get the stand-in"
    print("mutant selftest ok: a copy stands in for its module, an unmutated copy passes as the program, a mutated one fails")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        run(sys.argv[1:])
