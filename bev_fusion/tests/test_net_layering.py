"""D152: the model is separate from the training system and layered. net/prep (preprocessing) < net/model (the network) < net/loss (losses and targets)
< training/ (the training system). A module imports only its own layer and layers below it; nothing of net/ imports training/ or the training tooling
(autoresearch/trainkit, autoresearch/Training). Imports inside a self-test (a `_selftest...` function or an `if __name__ == "__main__"` block) are exempt.
Two named exceptions: the offline cache builders of the preprocessing run the frozen camera heads (net/model), and feature_cache.CachedBEVFusionModel
forwards to net/model/cached_model inside the function (the evaluation reads that name, also from old pinned run snapshots).
Also: module names are unique over the four folders (they are imported flat, and a run's snapshot is one flat folder).
CPU, about a second, no data. Run (from the bev_fusion folder): python tests/test_net_layering.py"""
import ast
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LAYERS = (("prep", "net/prep"), ("model", "net/model"), ("loss", "net/loss"), ("training", "training"))
ORDER = [name for name, _ in LAYERS]
BUILDERS = {"build_feature_cache", "cache_bevfusion_features"}          # preprocessing that runs the frozen camera heads
FORWARDS = {("feature_cache", "cached_model", "CachedBEVFusionModel")}  # (module, imported module, inside this function)
TOOLING = {"trainkit", "Training", "run", "jobrunner"}                  # the training tooling of autoresearch/
LAZY_TOOLING = {("student_encoder", "trainkit")}                        # the engine's strict checkpoint loader, imported inside the function that warm-starts the student


def modules():
    out = {}
    for layer, folder in LAYERS:
        d = os.path.join(ROOT, folder)
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                name = f[:-3]
                assert name not in out, f"module name {name!r} twice: {out[name][1]} and {folder}/{f}"
                out[name] = (layer, os.path.join(d, f))
    return out


def imports(path):
    """(imported top-level name, line, names of the enclosing functions) of every import outside the self-tests"""
    tree = ast.parse(open(path, encoding="utf-8").read())
    found = []

    def walk(node, inside):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if "selftest" in child.name or child.name.startswith("test_"):
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
    bad = []
    for name, (layer, path) in sorted(mods.items()):
        for target, line, inside in list(imports(path)) + [e[1:] for e in extra if e[0] == name]:
            if layer != "training" and target in TOOLING:
                if not (inside and (name, target) in LAZY_TOOLING):
                    bad.append(f"{layer}/{name}.py:{line} imports the training tooling {target!r}")
                continue
            if target not in mods or ORDER.index(mods[target][0]) <= ORDER.index(layer):
                continue
            if name in BUILDERS and mods[target][0] == "model":
                continue
            if any((name, target, f) in FORWARDS for f in inside):
                continue
            bad.append(f"{layer}/{name}.py:{line} imports {mods[target][0]}/{target} (a higher layer)")
    return bad


def main():
    mods = modules()
    count = {layer: sum(1 for v in mods.values() if v[0] == layer) for layer in ORDER}
    assert all(count[layer] > 0 for layer in ORDER), count
    bad = problems(mods)
    assert not bad, "\n".join(bad)
    # the check bites: a planted upward import in each net layer is found
    for name, target in (("geometry", "model"), ("model", "losses"), ("losses", "train"), ("precompute", "train")):
        planted = problems(mods, extra=[(name, target, 1, ())])
        assert any(f"{name}.py:1" in p for p in planted), f"planted import {name} -> {target} not found"
    assert any("tooling" in p for p in problems(mods, extra=[("model", "trainkit", 1, ())])), "planted tooling import not found"
    print(f"test_net_layering passed: {count['prep']} preprocessing, {count['model']} network, {count['loss']} loss and {count['training']} training modules; "
          f"no import goes up a layer, net/ reads no training system; planted imports are found")


if __name__ == "__main__":
    main()
