"""D152: the lane model is separate from the training system and layered. net/prep (preprocessing: ground-truth targets, datasets, caches) < net/model
(the networks) < net/loss (the losses) < training/ (trainer, train scripts, probes). A module of net/ imports only its own layer and layers below it,
nothing of training/ or evaluation/. Imports inside a self-test (a function with "selftest" in its name, `test_...`, `_t_...`, the named test helpers,
or an `if __name__ == "__main__"` block) are exempt: a command line there hands a preprocessing function the stored model it runs (the token cache
build, the champion's p_none of the map association). The 21 dependencies the code had before D152 that went against this were resolved on
2026-11-26; KNOWN names exceptions one by one and is empty: a new import fails the test, and an entry that does not occur must be taken out.
A module name of net/ exists only once on the lane's search path (modules are imported flat). And no module reaches a moved loss through its
model module (`jtoken.jtoken_loss`): the token test lends jtoken those names, so only this check sees such a call in a trainer. The paths module
`training/bflc_paths.py` (where code, data and runs are) may be imported by every layer; it imports nothing of the lane. CPU, a second, no data.
Run: `research tests lane_center --only test_net_layering` (or python net/tests/test_net_layering.py from the problem folder)."""
import ast
import os
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ORDER = ["prep", "model", "loss"]
TOOLING = ("Evaluation", "trainkit", "research")                       # packages of autoresearch/ that net/ must not read
TEST_HELPERS = set()                                                # (module, function): helpers of a self-test that import upwards
# (module of net/, imported module, the function the import must be inside, or None for anywhere); empty since 2026-11-26: keep it so
KNOWN = set()
PATHS = "bflc_paths"                                                  # the paths module: below every layer (it imports nothing of the lane, checked)


def modules():
    """module or package name -> (layer, path) for net/ (but net/tests), training/, evaluation/ and tools/, and the tooling of autoresearch/"""
    out = {name: ("autoresearch", "") for name in TOOLING}
    for top in ("net", "training", "evaluation", "tools"):
        for dp, dn, fn in os.walk(os.path.join(ROOT, top)):
            dn[:] = sorted(d for d in dn if d != "__pycache__" and not (top == "net" and d == "tests"))
            if top != "net" and dp != os.path.join(ROOT, top):               # a folder is importable as a package name too (sparse_lanes, lane_grid)
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


def imports(name, path):
    """(imported top-level name, line, names of the enclosing functions) of every import outside the self-tests"""
    tree = ast.parse(open(path, encoding="utf-8").read())
    found = []

    def walk(node, inside):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if "selftest" in child.name or child.name.startswith(("test_", "_t_")) or (name, child.name) in TEST_HELPERS:
                    continue
                walk(child, inside + (child.name,))
            elif isinstance(child, ast.If) and "__name__" in ast.unparse(child.test):
                continue
            else:
                if isinstance(child, ast.Import):
                    found.extend((a.name.split(".")[0], child.lineno, inside) for a in child.names)
                elif isinstance(child, ast.ImportFrom) and child.module:
                    found.append((child.module.split(".")[0], child.lineno, inside))
                walk(child, inside)
    walk(tree, ())
    return found


def problems(mods, extra=()):
    """-> (the imports against the layers that are not in KNOWN, the entries of KNOWN that were found)"""
    bad, used = [], set()
    for name, (layer, path) in sorted(mods.items()):
        if layer not in ORDER:
            continue
        for target, line, inside in imports(name, path) + [e[1:] for e in extra if e[0] == name]:
            if target not in mods or target == PATHS:
                continue
            tlayer = mods[target][0]
            if tlayer in ORDER and ORDER.index(tlayer) <= ORDER.index(layer):
                continue
            hit = [k for k in KNOWN if k[0] == name and k[1] == target and (k[2] is None or k[2] in inside)]
            if hit:
                used.update(hit)
                continue
            bad.append(f"net/{layer}/{name}.py:{line} imports {tlayer}/{target} (a higher layer, or outside net/)")
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


def main():
    mods = modules()
    count = {layer: sum(1 for v in mods.values() if v[0] == layer) for layer in ORDER}
    assert all(count[layer] > 0 for layer in ORDER), count
    bad, used = problems(mods)
    assert not bad, "\n".join(bad)
    assert used == KNOWN, f"named exceptions that no longer occur (take them out): {sorted(KNOWN - used, key=str)}"
    own = [t for t, _, _ in imports(PATHS, mods[PATHS][1]) if t in mods]
    assert not own, f"{PATHS} is the paths module below every layer: it imports nothing of the lane, but imports {own}"
    moved = {n[5:]: defined(p) - defined(mods[n[5:]][1]) for n, (layer, p) in mods.items() if n.startswith("loss_")}
    code = [os.path.join(dp, f) for top in ("net", "training", "evaluation", "tools") for dp, dn, fn in os.walk(os.path.join(ROOT, top))
            if os.path.relpath(dp, ROOT) != os.path.join("net", "tests") for f in fn if f.endswith(".py")]
    stale = through_model(code, moved)
    assert not stale, "\n".join(stale)
    with tempfile.TemporaryDirectory() as tmp:
        planted_file = os.path.join(tmp, "trainer_like.py")
        with open(planted_file, "w", encoding="utf-8") as fh:
            fh.write("import jtoken as JK\nloss = JK.jtoken_loss\n")
        assert through_model([planted_file], moved), "a planted loss read through the model module was not found"
    # the check bites: planted imports against the layers are found
    for name, target in (("qtargets", "jtoken"), ("jtoken", "loss_jtoken"), ("loss_jtoken", "jtoken_train"), ("qgraph", "jtoken_eval"), ("map_sim", "trainer"), ("qtargets", "sparse_lanes"),
                         ("jtoken", "Evaluation"), ("lane_gt_data", "research"), ("lane_tok_data", "jtok")):
        assert name in mods and target in mods, (name, target)                   # a planted pair names modules that exist
        planted, _ = problems(mods, extra=[(name, target, 1, ())])
        assert any(f"{name}.py:1" in p for p in planted), f"planted import {name} -> {target} not found"
    print(f"test_net_layering passed: {count['prep']} preprocessing, {count['model']} network and {count['loss']} loss modules; no import goes up a layer or out of net/ "
          f"({len(KNOWN)} named exceptions); planted imports are found")


if __name__ == "__main__":
    main()
