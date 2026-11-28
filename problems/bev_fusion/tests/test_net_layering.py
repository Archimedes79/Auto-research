"""The layers of the problem's code (DESIGN.md R13) and how its modules find each other; CPU, a second, no data.

    net/netpath.py  <  net/prep  <  net/model  <  net/loss  <  net/chain.py  <  training/ | evaluation/

A module imports its own layer and the layers below it. training/ and evaluation/ stand side by side on top: neither imports the other (the evaluation reads the
network through net/chain.py, never through the trainer). Only training/ imports the training tooling (autoresearch/trainkit, autoresearch/research), and no
production module imports tests/. EVERY import counts: inside a function, a self-test or an `if __name__` block too.
Named exceptions: the offline cache builders of the preprocessing run the frozen camera heads (net/model), and the student encoder reads a trained student with
the engine's strict checkpoint loader, imported inside that function.
Modules are imported by their package path from the problem folder (`from net.model.model import ...`): a flat import of a module of the problem, an edit of
sys.path and an os.chdir at import time are refused.
Run (from the problem folder, with its module path): python tests/test_net_layering.py"""
import ast
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# package prefix -> rank
LAYERS = (("net.netpath", 0), ("net.prep", 1), ("net.model", 2), ("net.loss", 3), ("net.chain", 4), ("training", 5), ("evaluation", 5))
FOLDERS = ("net", "net/prep", "net/model", "net/loss", "training", "evaluation")
BUILDERS = {"net.prep.build_feature_cache", "net.prep.cache_bevfusion_features"}     # preprocessing that runs the frozen camera heads of net/model
TOOLING = {"trainkit", "research"}                                                   # the training tooling of autoresearch/
LAZY_TOOLING = {("net.model.student_encoder", "trainkit")}                           # allowed inside a function only


def layer(module):
    """(layer prefix, rank) of a dotted module name of the problem, None for anything else"""
    return next(((p, r) for p, r in LAYERS if module == p or module.startswith(p + ".")), None)


def modules():
    """{dotted name: path} of the production modules"""
    out = {}
    for folder in FOLDERS:
        for f in sorted(os.listdir(os.path.join(ROOT, folder))):
            if f.endswith(".py"):
                out[folder.replace("/", ".") + "." + f[:-3]] = os.path.join(ROOT, folder, f)
    return out


def imports(source):
    """(imported dotted module, line, inside a function?) of every import of a source text, at any depth"""
    found = []

    def walk(node, inside):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Import):
                found.extend((a.name, child.lineno, inside) for a in child.names)
            elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
                found.extend((f"{child.module}.{a.name}", child.lineno, inside) for a in child.names)      # `from net import chain` names net.chain
            walk(child, inside or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)))
    walk(ast.parse(source), False)
    return found


def problems(name, source, flat):
    """what a module `name` (dotted) with this source does wrong"""
    bad, (own, rank) = [], layer(name)
    for target, line, inside in imports(source):
        top = target.split(".")[0]
        where = f"{name}:{line}"
        if top in TOOLING:
            if own != "training" and not (inside and (name, top) in LAZY_TOOLING):
                bad.append(f"{where} imports the training tooling {top!r}")
        elif top == "tests":
            bad.append(f"{where} imports tests/ ({target})")
        elif top in flat:
            bad.append(f"{where} imports {top!r} flat: name it by its package ({flat[top]})")
        elif layer(target):
            other, other_rank = layer(target)
            if other_rank > rank and not (name in BUILDERS and other == "net.model"):
                bad.append(f"{where} imports {target} (a higher layer: {other})")
            elif other_rank == rank and other != own:
                bad.append(f"{where} imports {target}: {own} and {other} stand side by side")
    tree = ast.parse(source)
    if any(isinstance(n, ast.Attribute) and n.attr == "path" and isinstance(n.value, ast.Name) and n.value.id in ("sys", "_sys") for n in ast.walk(tree)):
        bad.append(f"{name} edits or reads sys.path: the module path is the caller's (research puts the problem folder on PYTHONPATH)")
    if any(isinstance(c, ast.Call) and ast.unparse(c.func) == "os.chdir" for stmt in tree.body if not isinstance(stmt, (ast.FunctionDef, ast.ClassDef, ast.If))
           for c in ast.walk(stmt)):
        bad.append(f"{name} changes the working folder at import time")
    return bad


def main():
    mods = modules()
    flat = {name.rsplit(".", 1)[1]: name for name in mods}
    count = {prefix: sum(1 for m in mods if layer(m)[0] == prefix) for prefix, _ in LAYERS}
    assert all(count.values()), count
    bad = [p for name, path in sorted(mods.items()) for p in problems(name, open(path, encoding="utf-8").read(), flat)]
    assert not bad, "\n".join(bad)
    # the check bites: every kind of violation, planted, is found
    planted = (("net.prep.geometry", "from net.model.model import ModelConfig", "a higher layer"),
               ("net.model.model", "from net.loss import losses", "a higher layer"),
               ("net.loss.losses", "from net import chain", "a higher layer"),
               ("net.chain", "from training.train import TrainConfig", "a higher layer"),
               ("net.chain", "import evaluation.metrics", "a higher layer"),
               ("evaluation.bev_adapter", "from training import train as T", "side by side"),
               ("training.train", "from evaluation.bev_adapter import build", "side by side"),
               ("net.model.model", "from trainkit.checkpoint import warm_start", "training tooling"),
               ("evaluation.bev_adapter", "import trainkit.ema", "training tooling"),
               ("net.model.student_encoder", "from trainkit.checkpoint import warm_start", "training tooling"),          # allowed only inside its function
               ("net.model.model", "def _selftest():\n    from net.loss.losses import BEVLoss", "a higher layer"),          # a self-test is no exemption
               ("net.prep.augment", "if __name__ == '__main__':\n    from net.model.model import ModelConfig", "a higher layer"),
               ("net.loss.losses", "def _selftest():\n    from tests.support import synthetic_world", "imports tests/"),
               ("net.model.model", "import losses", "flat"),
               ("evaluation.metrics", "import sys\nsys.path.insert(0, 'x')", "sys.path"),
               ("training.train", "import os\nos.chdir('/tmp')", "at import time"))
    for name, source, what in planted:
        assert any(what in p for p in problems(name, source, flat)), f"planted in {name}, not found: {source!r}"
    assert not problems("net.model.student_encoder", "def load_trained():\n    from trainkit.checkpoint import warm_start", flat), "the named exception holds inside the function"
    assert not problems("net.prep.build_feature_cache", "from net.model.camera_encoder import FrontCameraEncoder", flat), "the cache builders run the camera heads"
    assert not problems("training.train", "import os\ndef main():\n    os.chdir('/tmp')", flat), "a chdir inside a function is the caller's choice"
    print("test_net_layering passed: " + ", ".join(f"{n} {p}" for p, n in count.items()) + f" modules; no import goes up a layer or between training/ and evaluation/, "
          f"net/ and evaluation/ read no training tooling, no flat import, no sys.path edit; {len(planted)} planted violations are found")


if __name__ == "__main__":
    main()
