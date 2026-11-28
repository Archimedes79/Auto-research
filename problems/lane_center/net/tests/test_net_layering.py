"""The layers of the lane problem's code (DESIGN.md R13) and the style of a problem that is a package; CPU, seconds, no data.

    net/prep (preprocessing: ground-truth targets, datasets, caches)  <  net/model (the networks)  <  net/loss (the losses)  <  training/ | evaluation/

A module of net/ imports only its own layer and layers below it, nothing of training/, evaluation/ or the tooling of autoresearch/. training/ and
evaluation/ stand side by side on top: neither imports the other (the evaluation reads the network through net/model/lane_net.py, never through a
trainer). EVERY import counts: inside a function, a self-test or an `if __name__` block too. KNOWN names the exceptions one by one: the command
line of the token cache build (net/prep/lane_tokcache.py `build`) runs the stored network; a new import fails the test, and an entry that does not
occur must be taken out. The paths module `training/bflc_paths.py` (where code, data and runs are) may be imported by every layer; it imports
nothing of the lane. No module reaches a moved loss through its model module (`jtoken.jtoken_loss`: the token test lends jtoken those names, so
only this check sees such a call in a trainer). The network is built, loaded and run from features to tokens in one place (ONE_PLACE:
`JTokenHead`, `QGraph`, `from_args`, `node_stage`, `select_tokens` are named only where they are defined and in net/model/lane_net.py, an alias
included; LOADERS: `torch.load` only in the named files; tests are free). The ground truth is reached through the public interface of GtGenerator
only (`GtGenerator.check_imports`).
Package (autoresearch/research/pkgcheck.py, its planted cases are in `research selftest`): a module of the problem is imported absolutely as
lane_center.<folders>.<module>, never flat, from the problem folder, relative or through the repository; every code folder holds an empty
__init__.py; nothing works on the module path (the one marked exception: the GaussianCaR clone in net/prep/foundation/gcar_features.py); steps
and tests are started as modules (`python -P -m`), and every module a command names (`-m lane_center.<x>` in code, documents and scripts; the
tests of problem.json) exists; every module is loaded once, from this folder.
Run: `research tests lane_center --only test_net_layering` (or python -P -m lane_center.net.tests.test_net_layering with the module path: README.md)."""
import ast
import json
import os
import re
import tempfile

import GtGenerator
from research import pkgcheck

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
NAME = os.path.basename(ROOT)                                         # lane_center: every module is NAME.<folders>.<module>
ORDER = ["prep", "model", "loss"]                                     # the layers of net/, the lowest first
SIDE = ("training", "evaluation")                                     # on top of net/, side by side
# packages of autoresearch/: none for net/, Evaluation for evaluation/, trainkit for training/
TOOLING = ("Evaluation", "trainkit", "research")
TOOL_OF = dict(training=("trainkit",), evaluation=("Evaluation",))
# (module, imported module, the function the import must be inside ("__main__": an `if __name__` block), or None for anywhere)
# the cache build's command line runs the stored network
KNOWN = {(f"{NAME}.net.prep.lane_tokcache", f"{NAME}.net.model.{m}", "__main__") for m in ("jtok", "lane_net")}
PATHS = f"{NAME}.training.bflc_paths"                                  # the paths module: below every layer (it imports nothing of the lane, checked)
NOT_LOADED = pkgcheck.SKIP + ("net/tests", "evaluation/tests")         # the identity check imports the rest (a test sets its environment when it is imported)
# The network is built, loaded and run from features to tokens in one place. A name of the network's construction -> the files that may NAME it (its definition
# and net/model/lane_net.py): any reference counts, so an alias (`H = JK.JTokenHead`, `from ...jtoken import JTokenHead as H`) is found like a call
ONE_PLACE = {"JTokenHead": ("net/model/jtoken.py", "net/model/lane_net.py"), "QGraph": ("net/model/qgraph.py",), "from_args": ("net/model/qgraph.py", "net/model/lane_net.py"),
             "node_stage": ("net/model/qgraph.py", "net/model/lane_net.py"), "select_tokens": ("net/prep/lane_tokcache.py", "net/model/lane_net.py")}
# `torch.load`: the files that may read a stored file with it, each with what it reads
LOADERS = {"net/model/lane_net.py": "the network's files (`load_checkpoint`, `load_b`)", "evaluation/lane_dump.py": "a run's args and states for the checks of a dump; it builds nothing",
           "training/trainer.py": "its own ckpt.pt (resume, roll-back)", "net/model/pyramid.py": "the E271 start weights of the pyramid (stage 0)",
           "net/prep/foundation/gcar_features.py": "the weights of the foundation (GaussianCaR), not of the lane network"}


def modules():
    """dotted module name -> (layer, path) for net/ (but net/tests), training/, evaluation/ and tools/; the tooling of autoresearch/ -> ("autoresearch", "")"""
    out = {name: ("autoresearch", "") for name in TOOLING}
    for top in ("net", "training", "evaluation", "tools"):
        for dp, dn, fn in os.walk(os.path.join(ROOT, top)):
            dn[:] = sorted(d for d in dn if d != "__pycache__" and not (top == "net" and d == "tests"))
            parts = os.path.relpath(dp, ROOT).split(os.sep)
            layer = parts[1] if parts[0] == "net" and len(parts) > 1 else parts[0]
            for f in sorted(fn):
                if f.endswith(".py") and f != "__init__.py":
                    out[".".join([NAME] + parts + [f[:-3]])] = (layer, os.path.join(dp, f))
    return out


def resolve(dotted, mods):
    """the module of `mods` that an imported dotted name names: the name itself or its longest prefix in mods (`Evaluation.stats` -> Evaluation), else None"""
    parts = dotted.split(".")
    return next((".".join(parts[:n]) for n in range(len(parts), 0, -1) if ".".join(parts[:n]) in mods), None)


def imports(source):
    """(imported dotted name, line, the enclosing functions; "__main__" for an `if __name__` block) of EVERY absolute import of a source text;
    `from a.b import c` gives a.b.c (the module c, or a name of the module a.b: `resolve` decides)"""
    found = []

    def walk(node, inside):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Import):
                found.extend((a.name, child.lineno, inside) for a in child.names)
            elif isinstance(child, ast.ImportFrom) and child.module and child.level == 0:
                found.extend((f"{child.module}.{a.name}", child.lineno, inside) for a in child.names)
            here = inside
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                here = inside + (child.name,)
            elif isinstance(child, ast.If) and "__name__" in ast.unparse(child.test):
                here = inside + ("__main__",)
            walk(child, here)
    walk(ast.parse(source), ())
    return found


def allowed(layer, tlayer, target):
    """may a module of `layer` import the module `target` of `tlayer`? -> (yes, the reason of a no)"""
    if layer in ORDER:
        return tlayer in ORDER and ORDER.index(tlayer) <= ORDER.index(layer), "a higher layer, or outside net/"
    if tlayer in SIDE and tlayer != layer:
        return False, f"{layer}/ and {tlayer}/ stand side by side"
    if tlayer == "autoresearch" and target not in TOOL_OF[layer]:
        return False, f"the tooling of {[k for k, v in TOOL_OF.items() if target in v] or ['the core']}"
    return True, ""


def problems(mods, planted=None):
    """the imports against the rules that are not in KNOWN -> (messages, the entries of KNOWN that were found). planted = {module: source text}: read
    instead of the files, and only those modules are checked (the proof that the check bites)"""
    bad, used = [], set()
    for name, (layer, path) in sorted(mods.items()):
        if layer not in ORDER + list(SIDE) or not path or (planted is not None and name not in planted):
            continue
        source = planted[name] if planted is not None else open(path, encoding="utf-8").read()
        for imported, line, inside in imports(source):
            target = resolve(imported, mods)
            if target is None or target == PATHS:
                continue
            ok, why = allowed(layer, mods[target][0], target)
            if ok:
                continue
            hit = [k for k in KNOWN if k[0] == name and k[1] == target and (k[2] is None or k[2] in inside)]
            if hit:
                used.update(hit)
                continue
            bad.append(f"{os.path.relpath(path, ROOT)}:{line} imports {mods[target][0]}/{target} ({why})")
    return bad, used


def defined(path):
    """the names a module binds at its top level by def, class or assignment"""
    out = set()
    for node in ast.parse(open(path, encoding="utf-8").read()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out |= {x.id for t in node.targets for x in ast.walk(t) if isinstance(x, ast.Name)}
    return out


def through_model(paths, moved):
    """`alias.name` in the files `paths` where alias is a model module and name lives in its loss module only. moved = {model module (dotted): names}"""
    bad = []
    for path in paths:
        tree = ast.parse(open(path, encoding="utf-8").read())
        alias = {a.asname or a.name: f"{n.module}.{a.name}" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module for a in n.names
                 if f"{n.module}.{a.name}" in moved}
        alias.update({a.asname: a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names if a.asname and a.name in moved})
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in alias and n.attr in moved[alias[n.value.id]]:
                model = alias[n.value.id]
                bad.append(f"{os.path.relpath(path, ROOT)}:{n.lineno} reads {model}.{n.attr}: it is in loss_{model.rsplit('.', 1)[1]}")
    return bad


def built_elsewhere(paths, root=ROOT):
    """the references to a name of ONE_PLACE (a name, an attribute, a from-import) and the uses of `torch.load` in the files `paths` outside the files that may
    hold them"""
    bad = []
    for path in paths:
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        for n in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
            names = [n.id] if isinstance(n, ast.Name) else [n.attr] if isinstance(n, ast.Attribute) else [a.name for a in n.names] if isinstance(n, ast.ImportFrom) else []
            for name in names:
                if name in ONE_PLACE and rel not in ONE_PLACE[name]:
                    bad.append(f"{rel}:{n.lineno} names {name}: the network is built and run in {ONE_PLACE[name][-1]}")
            loads = (isinstance(n, ast.Attribute) and n.attr == "load" and isinstance(n.value, ast.Name) and n.value.id == "torch") or (isinstance(n, ast.ImportFrom) and n.module == "torch" and "load" in names)
            if loads and rel not in LOADERS:
                bad.append(f"{rel}:{n.lineno} reads a file with torch.load: the loaders are net/model/lane_net.py (`load_checkpoint`, `load_b`)")
    return bad


def started_unknown(root=ROOT):
    """the modules of the problem that a command names and that do not exist: `-m lane_center.<x>` (also `"-m", "lane_center.<x>"`) in the code, the
    documents and the scripts outside the records, and the tests of problem.json -> [(file, module)]"""
    pat = re.compile(r'-m(?:", "| )(' + re.escape(NAME) + r'(?:\.\w+)+)')
    named = [("problem.json", t.split()[0]) for t in json.load(open(os.path.join(root, "problem.json"), encoding="utf-8"))["tests"]]
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d not in pkgcheck.SKIP + pkgcheck.SCRATCH)
        for f in sorted(fn):
            if f.endswith((".py", ".md", ".sh", ".json")):
                text = open(os.path.join(dp, f), encoding="utf-8").read()
                named += [(os.path.relpath(os.path.join(dp, f), root), m) for m in pat.findall(text)]
    exists = lambda m: any(os.path.isfile(os.path.join(root, *m.split(".")[1:]) + end) for end in (".py", os.sep + "__init__.py"))  # noqa: E731
    return [(f, m) for f, m in named if not exists(m)]


def main():
    mods = modules()
    unknown = started_unknown()
    assert not unknown, f"commands that start a module that does not exist: {unknown}"
    with tempfile.TemporaryDirectory() as tmp:                                    # the check bites: a name glued to its subcommand is found
        os.makedirs(os.path.join(tmp, "net"))
        json.dump({"tests": [f"{NAME}.net.no_test"]}, open(os.path.join(tmp, "problem.json"), "w"))
        open(os.path.join(tmp, "net", "x.py"), "w").write(f'"""python -P -m {NAME}.net.xbuild --n 2"""\n')
        assert sorted(m for _, m in started_unknown(tmp)) == [f"{NAME}.net.no_test", f"{NAME}.net.xbuild"], started_unknown(tmp)
    count = {layer: sum(1 for v in mods.values() if v[0] == layer and v[1]) for layer in ORDER + list(SIDE)}
    assert all(count.values()), count
    bad, used = problems(mods)
    assert not bad, "\n".join(bad)
    assert used == KNOWN, f"named exceptions that no longer occur (take them out): {sorted(KNOWN - used, key=str)}"
    own = [t for t, _, _ in imports(open(mods[PATHS][1], encoding="utf-8").read()) if resolve(t, mods)]
    assert not own, f"{PATHS} is the paths module below every layer: it imports nothing of the lane, but imports {own}"
    model = f"{NAME}.net.model."
    moved = {model + n.rsplit(".loss_", 1)[1]: defined(p) - defined(mods[model + n.rsplit(".loss_", 1)[1]][1]) for n,
             (layer, p) in mods.items() if ".net.loss.loss_" in n}
    code = [os.path.join(dp, f) for top in ("net", "training", "evaluation", "tools") for dp, dn, fn in os.walk(os.path.join(ROOT, top))
            if os.path.basename(dp) != "tests" for f in fn if f.endswith(".py")]
    stale = through_model(code, moved)
    assert not stale, "\n".join(stale)
    twice = built_elsewhere(code)
    assert not twice, "\n".join(twice)
    n = NAME
    with tempfile.TemporaryDirectory() as tmp:
        planted_file = os.path.join(tmp, "trainer_like.py")
        with open(planted_file, "w", encoding="utf-8") as fh:
            fh.write(f"from {n}.net.model import jtoken as JK\nloss = JK.jtoken_loss\n")
        assert through_model([planted_file], moved), "a planted loss read through the model module was not found"
        for source, what in ((f"from {n}.net.model import jtoken as JK\nhead = JK.JTokenHead()\n", "names JTokenHead"), (f"from {n}.net.model import qgraph as QG\nb = QG.QGraph()\n", "names QGraph"),
                             # an aliased constructor
                             (f"from {n}.net.model import jtoken as JK\nmake = JK.JTokenHead\nhead = make()\n", "names JTokenHead"),
                             # a from-import alias
                             (f"from {n}.net.model.jtoken import JTokenHead as H\nhead = H()\n", "names JTokenHead"),
                             # a second copy of features -> tokens ...
                             (f"from {n}.net.model import qgraph as QG\nb = QG.from_args({{}})\n", "names from_args"),
                             ("def tokens(b, x):\n    return b.node_stage(b.pyr(x))\n", "names node_stage"),
                             (f"from {n}.net.prep import lane_tokcache as LC\ntk = LC.select_tokens(None, None, 8)\n", "names select_tokens"),
                             ("import torch\nck = torch.load('model.pt')\n", "torch.load"), ("from torch import load\nck = load('model.pt')\n", "torch.load")):
            with open(planted_file, "w", encoding="utf-8") as fh:
                fh.write(source)
            found = built_elsewhere([planted_file], tmp)
            assert len(found) == 1 and what in found[0], f"planted, not found ({what}): {source!r} -> {found}"
    # the check bites: every kind of violation, planted, is found; and a named exception holds only where it is named
    prep, mdl, loss, ev, tr = (f"{n}.net.prep", f"{n}.net.model", f"{n}.net.loss", f"{n}.evaluation", f"{n}.training")
    planted = ((f"{prep}.qtargets", f"from {mdl} import jtoken", "a higher layer"), (f"{mdl}.jtoken", f"from {loss} import loss_jtoken", "a higher layer"),
               (f"{loss}.loss_jtoken", f"from {tr}.train import jtoken_train", "outside net/"), (f"{mdl}.qgraph", f"from {ev} import jtoken_eval", "outside net/"),
               (f"{prep}.map_sim", f"from {tr} import trainer", "outside net/"), (f"{prep}.qtargets", f"from {ev}.sparse_lanes import graph_eval", "outside net/"),
               (f"{mdl}.jtoken", "from Evaluation import stats", "outside net/"), (f"{prep}.lane_gt_data", "import research", "outside net/"),
               (f"{prep}.lane_tok_data", f"from {mdl} import jtok", "a higher layer"),
               # a self-test is no exemption
               (f"{mdl}.qmnms", f"def _selftest():\n    from {ev} import qgraph_eval", "outside net/"),
               (f"{mdl}.sections", f"def _selftest():\n    from {loss}.loss_sections import section_loss", "a higher layer"),
               (f"{prep}.lane_ptok_data", f"if __name__ == '__main__':\n    from {mdl} import lane_net", "a higher layer"),          # nor is a main block
               # the named exception holds inside its main block only
               (f"{prep}.lane_tokcache", f"from {mdl} import lane_net", "a higher layer"),
               (f"{ev}.adapter", f"from {tr}.train import jtoken_train", "side by side"), (f"{ev}.lane_predict", f"def dump():\n    from {tr}.train import jlive_train", "side by side"),
               (f"{ev}.jtoken_eval", f"if __name__ == '__main__':\n    from {tr} import trainer", "side by side"), (f"{tr}.train.jlive_train", f"from {ev} import jtoken_eval", "side by side"),
               (f"{tr}.train.lane_train", f"def main():\n    from {ev} import adapter", "side by side"), (f"{tr}.train.jtoken_train", f"from {ev}.qgraph_eval import score_frames", "side by side"),
               (f"{ev}.adapter", "from trainkit import control", "the tooling of ['training']"), (f"{tr}.train.jlive_train", "from Evaluation import stats", "the tooling of ['evaluation']"))
    for name, source, why in planted:
        assert name in mods and mods[name][1], name                               # a planted case names a module that exists
        found, _ = problems(mods, {name: source})
        assert len(found) == 1 and why in found[0], f"planted in {name}, not found ({why}): {source!r} -> {found}"
    main_block = f"if __name__ == '__main__':\n    from {mdl} import lane_net\n    from {mdl} import jtok"
    assert problems(mods, {f"{prep}.lane_tokcache": main_block}) == ([], KNOWN), "the named exception of the cache build"
    # the package style (P1 - P3 of pkgcheck) and one identity per module
    style = pkgcheck.check(ROOT)
    assert not style, "\n".join(style)
    twice = pkgcheck.identity_errors(ROOT, NOT_LOADED)
    assert not twice, "\n".join(twice)
    below = GtGenerator.check_imports([ROOT])
    assert not below, "\n".join(below)
    print("test_net_layering passed: " + ", ".join(f"{count[k]} {k}" for k in count) + f" modules; no import goes up a layer or out of net/, none between training/ and "
          f"evaluation/ ({len(KNOWN)} named exceptions, {len(planted)} planted violations are found); the network is built in one place; the package style holds "
          f"and every module is loaded once; the ground truth is reached through the interface of GtGenerator only")


if __name__ == "__main__":
    main()
