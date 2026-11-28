"""The layers of the problem's code (DESIGN.md R13), the style of a problem that is a package and the ground-truth interface; CPU, seconds, no data.

    net/netpath.py  <  net/prep  <  net/model  <  net/loss  <  net/chain.py  <  training/ | evaluation/

Layers (the table of this file): a module imports its own layer and the layers below it. training/ and evaluation/ stand side by side on top: neither imports
the other (the evaluation reads the network through net/chain.py, never through the trainer). Only training/ imports the training tooling
(autoresearch/trainkit, autoresearch/research), and no production module imports tests/. EVERY import counts: inside a function, a self-test or an
`if __name__` block too. Named exceptions: the offline cache builders of the preprocessing run the frozen camera heads (net/model), and the student encoder
reads a trained student with the engine's strict checkpoint loader, imported inside that function.
Package (autoresearch/research/pkgcheck.py): a module of the problem is imported absolutely as bev_fusion.<folders>.<module>, never flat, from the problem
folder, relative or through the repository; every code folder holds an empty __init__.py; nothing works on the module path (sys.path, loading a file as a
module, os.chdir at import time); steps and tests are started as modules (`python -P -m`); every module is loaded once, from this folder.
Ground truth: only `import GtGenerator` and its public names (GtGenerator.check_imports).

    python -P -m bev_fusion.tests.test_net_layering      (from the problem folder, with its module path: README.md)"""
import ast
import json
import os
import shutil
import tempfile

import GtGenerator
from research import pkgcheck

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NAME = os.path.basename(ROOT)
# package prefix -> rank
LAYERS = tuple((f"{NAME}.{prefix}", rank)
               for prefix, rank in (("net.netpath", 0), ("net.prep", 1), ("net.model", 2), ("net.loss", 3), ("net.chain", 4), ("training", 5), ("evaluation", 5)))
FOLDERS = ("net", "net/prep", "net/model", "net/loss", "training", "evaluation")
# preprocessing that runs the frozen camera heads of net/model
BUILDERS = {f"{NAME}.net.prep.build_feature_cache", f"{NAME}.net.prep.cache_bevfusion_features"}
TOOLING = {"trainkit", "research"}                                                   # the training tooling of autoresearch/
LAZY_TOOLING = {(f"{NAME}.net.model.student_encoder", "trainkit")}                   # allowed inside a function only
NOT_LOADED = pkgcheck.SKIP + ("tests",)                                              # the tests run when they are imported: the identity check loads the rest


def layer(module):
    """(layer prefix, rank) of a dotted module name of the problem, None for anything else"""
    return next(((p, r) for p, r in LAYERS if module == p or module.startswith(p + ".")), None)


def modules():
    """{dotted name: path} of the production modules"""
    out = {}
    for folder in FOLDERS:
        for f in sorted(os.listdir(os.path.join(ROOT, folder))):
            if f.endswith(".py") and f != "__init__.py":
                out[f"{NAME}.{folder.replace('/', '.')}.{f[:-3]}"] = os.path.join(ROOT, folder, f)
    return out


def imports(source):
    """(imported dotted module, line, inside a function?) of every absolute import of a source text, at any depth (pkgcheck refuses the relative ones)"""
    found = []

    def walk(node, inside):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Import):
                found.extend((a.name, child.lineno, inside) for a in child.names)
            elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
                found.extend((f"{child.module}.{a.name}", child.lineno, inside) for a in child.names)      # `from <problem>.net import chain` names ...net.chain
            walk(child, inside or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)))
    walk(ast.parse(source), False)
    return found


def problems(name, source):
    """what a module `name` (dotted) with this source does wrong against the layer table"""
    bad, (own, rank) = [], layer(name)
    for target, line, inside in imports(source):
        top = target.split(".")[0]
        where = f"{name}:{line}"
        if top in TOOLING:
            if own != f"{NAME}.training" and not (inside and (name, top) in LAZY_TOOLING):
                bad.append(f"{where} imports the training tooling {top!r}")
        elif target.startswith(f"{NAME}.tests."):
            bad.append(f"{where} imports tests/ ({target})")
        elif layer(target):
            other, other_rank = layer(target)
            if other_rank > rank and not (name in BUILDERS and other == f"{NAME}.net.model"):
                bad.append(f"{where} imports {target} (a higher layer: {other})")
            elif other_rank == rank and other != own:
                bad.append(f"{where} imports {target}: {own} and {other} stand side by side")
    return bad


def style_bites():
    """the package check finds each kind of violation, planted in a small copy of this problem's layout -> the number planted"""
    planted = (("import losses", "is flat"), ("from net.loss import losses", "from the problem folder"), ("from . import radar_bev", "is relative"),
               ("from ..loss import losses", "is relative"), (f"from problems.{NAME}.net.loss import losses", "through the repository"),
               ("import autoresearch.Evaluation", "through the repository"), (f"from {NAME}.net.loss import no_such_module", "no such module"),
               ("import sys\nsys.path.insert(0, 'x')", "sys.path"), ("import os\nos.chdir('/tmp')", "os.chdir at import time"),
               ("import importlib.util\nspec = importlib.util.spec_from_file_location('m', 'p')", "spec_from_file_location"))
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, NAME)
        for folder in ("", "net", "net/model", "net/loss"):
            os.makedirs(os.path.join(p, folder), exist_ok=True)
            open(os.path.join(p, folder, "__init__.py"), "w").close()
        shutil.copy(os.path.join(ROOT, "problem.json"), p)
        for rel in ("net/loss/losses.py", "net/model/radar_bev.py", "net/model/model.py"):
            open(os.path.join(p, rel), "w").write("X = 1\n")
        assert pkgcheck.check(p) == [], pkgcheck.check(p)
        for source, what in planted:
            open(os.path.join(p, "net/model/model.py"), "w").write(source + "\n")
            assert any(what in e for e in pkgcheck.check(p)), f"planted, not found ({what}): {source!r} -> {pkgcheck.check(p)}"
        open(os.path.join(p, "net/model/model.py"), "w").write("X = 1\n")
        with open(os.path.join(p, "problem.json"), encoding="utf-8") as f:
            real = json.load(f)
        layout = ((lambda: open(os.path.join(p, "net/loss/__init__.py"), "w").write("X = 1\n"), "not empty"),
                  (lambda: os.remove(os.path.join(p, "net/loss/__init__.py")), "no __init__.py"),
                  (lambda: json.dump(dict(real, pythonpath=["net"]), open(os.path.join(p, "problem.json"), "w")), '"pythonpath"'),
                  (lambda: json.dump(dict(real, steps={"train": {"run": ["{python}", "training/train.py", "{config}"]}}), open(os.path.join(p, "problem.json"), "w")),
                   "steps.train"),
                  (lambda: json.dump(dict(real, tests=["tests/test_chain.py"]), open(os.path.join(p, "problem.json"), "w")), "problem.json tests"))
        for plant, what in layout:
            plant()
            assert any(what in e for e in pkgcheck.check(p)), f"planted, not found: {what} -> {pkgcheck.check(p)}"
    return len(planted) + len(layout)


def main():
    mods = modules()
    count = {prefix: sum(1 for m in mods if layer(m)[0] == prefix) for prefix, _ in LAYERS}
    assert all(count.values()), count
    bad = [p for name, path in sorted(mods.items()) for p in problems(name, open(path, encoding="utf-8").read())]
    assert not bad, "\n".join(bad)
    # the layer table bites: every kind of violation, planted, is found
    n = NAME
    planted = ((f"{n}.net.prep.geometry", f"from {n}.net.model.model import ModelConfig", "a higher layer"),
               (f"{n}.net.model.model", f"from {n}.net.loss import losses", "a higher layer"),
               (f"{n}.net.loss.losses", f"from {n}.net import chain", "a higher layer"),
               (f"{n}.net.chain", f"from {n}.training.train import TrainConfig", "a higher layer"),
               (f"{n}.net.chain", f"import {n}.evaluation.metrics", "a higher layer"),
               (f"{n}.evaluation.bev_adapter", f"from {n}.training import train as T", "side by side"),
               (f"{n}.training.train", f"from {n}.evaluation.bev_adapter import build", "side by side"),
               (f"{n}.net.model.model", "from trainkit.checkpoint import warm_start", "training tooling"),
               (f"{n}.evaluation.bev_adapter", "import trainkit.ema", "training tooling"),
               (f"{n}.net.model.student_encoder", "from trainkit.checkpoint import warm_start", "training tooling"),      # allowed only inside its function
               (f"{n}.net.model.model", f"def _selftest():\n    from {n}.net.loss.losses import BEVLoss", "a higher layer"),      # a self-test is no exemption
               (f"{n}.net.prep.augment", f"if __name__ == '__main__':\n    from {n}.net.model.model import ModelConfig",
                "a higher layer"),
               (f"{n}.net.loss.losses", f"def _selftest():\n    from {n}.tests.support import synthetic_world", "imports tests/"))
    for name, source, what in planted:
        assert any(what in p for p in problems(name, source)), f"planted in {name}, not found: {source!r}"
    exception = problems(f"{n}.net.model.student_encoder", "def load_trained():\n    from trainkit.checkpoint import warm_start")
    assert not exception, "the named exception holds inside the function"
    builder = problems(f"{n}.net.prep.build_feature_cache", f"from {n}.net.model.camera_encoder import FrontCameraEncoder")
    assert not builder, "the cache builders run the camera heads"
    # the package style (P1 - P3 of pkgcheck), its planted cases, and one identity per module
    assert pkgcheck.check(ROOT) == [], "\n".join(pkgcheck.check(ROOT))
    styled = style_bites()
    twice = pkgcheck.identity_errors(ROOT, NOT_LOADED)
    assert twice == [], "\n".join(twice)
    below = GtGenerator.check_imports([ROOT])
    assert not below, "\n".join(below)
    print("test_net_layering passed: " + ", ".join(f"{c} {p[len(NAME) + 1:]}" for p, c in count.items()) + f" modules; no import goes up a layer or between training/ and "
          f"evaluation/, net/ and evaluation/ read no training tooling ({len(planted)} planted violations are found); the package style holds ({styled} planted violations "
          f"are found) and every module is loaded once; the ground truth is reached through the interface of GtGenerator only")


if __name__ == "__main__":
    main()
