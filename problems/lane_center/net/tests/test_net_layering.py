"""The layers of the lane problem's code (DESIGN.md R13); CPU, a second, no data.

    net/prep (preprocessing: ground-truth targets, datasets, caches)  <  net/model (the networks)  <  net/loss (the losses)  <  training/ | evaluation/

A module of net/ imports only its own layer and layers below it, nothing of training/, evaluation/ or the tooling of autoresearch/. training/ and
evaluation/ stand side by side on top: neither imports the other (the evaluation reads the network through net/model/lane_net.py, never through a
trainer). EVERY import counts: inside a function, a self-test or an `if __name__` block too. KNOWN names the exceptions one by one: the command
line of the token cache build (net/prep/lane_tokcache.py `build`) runs the stored network; a new import fails the test, and an entry that does not
occur must be taken out. The paths module `training/bflc_paths.py` (where code, data and runs are) may be imported by every layer; it imports
nothing of the lane. A module name of net/ exists only once on the lane's search path (modules are imported flat). No module reaches a moved loss
through its model module (`jtoken.jtoken_loss`: the token test lends jtoken those names, so only this check sees such a call in a trainer). The
network is built, loaded and run from features to tokens in one place (ONE_PLACE: `JTokenHead`, `QGraph`, `from_args`, `node_stage`,
`select_tokens` are named only where they are defined and in net/model/lane_net.py, an alias included; LOADERS: `torch.load` only in the named
files; tests are free). The ground truth is reached through the public interface of GtGenerator only (`GtGenerator.check_imports`).
Run: `research tests lane_center --only test_net_layering` (or python net/tests/test_net_layering.py with the problem's module path)."""
import ast
import os
import tempfile

import GtGenerator

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ORDER = ["prep", "model", "loss"]                                     # the layers of net/, the lowest first
SIDE = ("training", "evaluation")                                     # on top of net/, side by side
TOOLING = ("Evaluation", "trainkit", "research")                       # packages of autoresearch/: none for net/, Evaluation for evaluation/, trainkit for training/
TOOL_OF = dict(training=("trainkit",), evaluation=("Evaluation",))
# (module, imported module, the function the import must be inside ("__main__": an `if __name__` block), or None for anywhere)
KNOWN = {("lane_tokcache", "jtok", "__main__"), ("lane_tokcache", "lane_net", "__main__")}      # the cache build's command line runs the stored network
PATHS = "bflc_paths"                                                  # the paths module: below every layer (it imports nothing of the lane, checked)
# The network is built, loaded and run from features to tokens in one place. A name of the network's construction -> the files that may NAME it (its definition and
# net/model/lane_net.py): any reference counts, so an alias (`H = JK.JTokenHead`, `from jtoken import JTokenHead as H`) is found like a call
ONE_PLACE = {"JTokenHead": ("net/model/jtoken.py", "net/model/lane_net.py"), "QGraph": ("net/model/qgraph.py",), "from_args": ("net/model/qgraph.py", "net/model/lane_net.py"),
             "node_stage": ("net/model/qgraph.py", "net/model/lane_net.py"), "select_tokens": ("net/prep/lane_tokcache.py", "net/model/lane_net.py")}
# `torch.load`: the files that may read a stored file with it, each with what it reads
LOADERS = {"net/model/lane_net.py": "the network's files (`load_checkpoint`, `load_b`)", "evaluation/lane_dump.py": "a run's args and states for the checks of a dump; it builds nothing",
           "training/trainer.py": "its own ckpt.pt (resume, roll-back)", "net/model/pyramid.py": "the E271 start weights of the pyramid (stage 0)",
           "net/prep/foundation/gcar_features.py": "the weights of the foundation (GaussianCaR), not of the lane network"}


def modules():
    """module or package name -> (layer, path) for net/ (but net/tests), training/, evaluation/ and tools/, and the tooling of autoresearch/"""
    out = {name: ("autoresearch", "") for name in TOOLING}
    for top in ("net", "training", "evaluation", "tools"):
        for dp, dn, fn in os.walk(os.path.join(ROOT, top)):
            dn[:] = sorted(d for d in dn if d != "__pycache__" and not (top == "net" and d == "tests"))
            if top != "net" and dp != os.path.join(ROOT, top):               # a folder is importable as a package name too (sparse_lanes, lane_task)
                out.setdefault(os.path.basename(dp), (top, ""))
            for f in sorted(fn):
                if not f.endswith(".py") or f == "__init__.py":
                    continue
                parts = os.path.relpath(dp, ROOT).split(os.sep)
                layer = parts[1] if parts[0] == "net" and len(parts) > 1 else parts[0]
                name = f[:-3]
                if name in out and out[name][1]:
                    assert layer not in ORDER and out[name][0] not in ORDER, f"module name {name!r} twice: {out[name][1]} and {os.path.join(dp, f)}"
                    continue
                out[name] = (layer, os.path.join(dp, f))
    return out


def imports(source):
    """(imported top-level name, line, the enclosing functions; "__main__" for an `if __name__` block) of EVERY absolute import of a source text"""
    found = []

    def walk(node, inside):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Import):
                found.extend((a.name.split(".")[0], child.lineno, inside) for a in child.names)
            elif isinstance(child, ast.ImportFrom) and child.module and child.level == 0:
                found.append((child.module.split(".")[0], child.lineno, inside))
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
        for target, line, inside in imports(source):
            if target not in mods or target == PATHS:
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
    """`alias.name` in the files `paths` where alias is a model module and name lives in its loss module only. moved = {model module: names}"""
    bad = []
    for path in paths:
        tree = ast.parse(open(path, encoding="utf-8").read())
        alias = {a.asname or a.name: a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names if a.name in moved}
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in alias and n.attr in moved[alias[n.value.id]]:
                bad.append(f"{os.path.relpath(path, ROOT)}:{n.lineno} reads {alias[n.value.id]}.{n.attr}: it is in loss_{alias[n.value.id]}")
    return bad


def built_elsewhere(paths, root=ROOT):
    """the references to a name of ONE_PLACE (a name, an attribute, a from-import) and the uses of `torch.load` in the files `paths` outside the files that may hold them"""
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


def main():
    mods = modules()
    count = {layer: sum(1 for v in mods.values() if v[0] == layer and v[1]) for layer in ORDER + list(SIDE)}
    assert all(count.values()), count
    bad, used = problems(mods)
    assert not bad, "\n".join(bad)
    assert used == KNOWN, f"named exceptions that no longer occur (take them out): {sorted(KNOWN - used, key=str)}"
    own = [t for t, _, _ in imports(open(mods[PATHS][1], encoding="utf-8").read()) if t in mods]
    assert not own, f"{PATHS} is the paths module below every layer: it imports nothing of the lane, but imports {own}"
    moved = {n[5:]: defined(p) - defined(mods[n[5:]][1]) for n, (layer, p) in mods.items() if n.startswith("loss_")}
    code = [os.path.join(dp, f) for top in ("net", "training", "evaluation", "tools") for dp, dn, fn in os.walk(os.path.join(ROOT, top))
            if os.path.basename(dp) != "tests" for f in fn if f.endswith(".py")]
    stale = through_model(code, moved)
    assert not stale, "\n".join(stale)
    twice = built_elsewhere(code)
    assert not twice, "\n".join(twice)
    with tempfile.TemporaryDirectory() as tmp:
        planted_file = os.path.join(tmp, "trainer_like.py")
        with open(planted_file, "w", encoding="utf-8") as fh:
            fh.write("import jtoken as JK\nloss = JK.jtoken_loss\n")
        assert through_model([planted_file], moved), "a planted loss read through the model module was not found"
        for source, what in (("import jtoken as JK\nhead = JK.JTokenHead()\n", "names JTokenHead"), ("import qgraph as QG\nb = QG.QGraph()\n", "names QGraph"),
                             ("import jtoken as JK\nmake = JK.JTokenHead\nhead = make()\n", "names JTokenHead"),                              # an aliased constructor
                             ("from jtoken import JTokenHead as H\nhead = H()\n", "names JTokenHead"),                                       # a from-import alias
                             ("import qgraph as QG\nb = QG.from_args({})\n", "names from_args"),                                              # a second copy of features -> tokens ...
                             ("def tokens(b, x):\n    return b.node_stage(b.pyr(x))\n", "names node_stage"),
                             ("import lane_tokcache as LC\ntk = LC.select_tokens(None, None, 8)\n", "names select_tokens"),
                             ("import torch\nck = torch.load('model.pt')\n", "torch.load"), ("from torch import load\nck = load('model.pt')\n", "torch.load")):
            with open(planted_file, "w", encoding="utf-8") as fh:
                fh.write(source)
            found = built_elsewhere([planted_file], tmp)
            assert len(found) == 1 and what in found[0], f"planted, not found ({what}): {source!r} -> {found}"
    # the check bites: every kind of violation, planted, is found; and a named exception holds only where it is named
    planted = (("qtargets", "import jtoken", "a higher layer"), ("jtoken", "import loss_jtoken", "a higher layer"), ("loss_jtoken", "import jtoken_train", "outside net/"),
               ("qgraph", "import jtoken_eval", "outside net/"), ("map_sim", "import trainer", "outside net/"), ("qtargets", "from sparse_lanes import graph_eval", "outside net/"),
               ("jtoken", "from Evaluation import stats", "outside net/"), ("lane_gt_data", "import research", "outside net/"), ("lane_tok_data", "import jtok", "a higher layer"),
               ("qmnms", "def _selftest():\n    import qgraph_eval", "outside net/"),                                  # a self-test is no exemption
               ("sections", "def _selftest():\n    from loss_sections import section_loss", "a higher layer"),
               ("lane_ptok_data", "if __name__ == '__main__':\n    import lane_net", "a higher layer"),                # nor is a main block
               ("lane_tokcache", "import lane_net", "a higher layer"),                                                 # the named exception holds inside its main block only
               ("adapter", "import jtoken_train", "side by side"), ("lane_predict", "def dump():\n    import jlive_train", "side by side"),
               ("jtoken_eval", "if __name__ == '__main__':\n    import trainer", "side by side"), ("jlive_train", "import jtoken_eval", "side by side"),
               ("lane_train", "def main():\n    import adapter", "side by side"), ("jtoken_train", "from qgraph_eval import score_frames", "side by side"),
               ("adapter", "from trainkit import control", "the tooling of ['training']"), ("jlive_train", "from Evaluation import stats", "the tooling of ['evaluation']"))
    for name, source, why in planted:
        assert name in mods and mods[name][1], name                               # a planted case names a module that exists
        found, _ = problems(mods, {name: source})
        assert len(found) == 1 and why in found[0], f"planted in {name}, not found ({why}): {source!r} -> {found}"
    assert problems(mods, {"lane_tokcache": "if __name__ == '__main__':\n    import lane_net\n    import jtok"}) == ([], KNOWN), "the named exception of the cache build"
    below = GtGenerator.check_imports([ROOT])
    assert not below, "\n".join(below)
    print("test_net_layering passed: " + ", ".join(f"{count[k]} {k}" for k in count) + f" modules; no import goes up a layer or out of net/, none between training/ and "
          f"evaluation/ ({len(KNOWN)} named exceptions, {len(planted)} planted violations are found); the network is built in one place; the ground truth is reached "
          f"through the interface of GtGenerator only")


if __name__ == "__main__":
    main()
