"""The evaluator hash covers the scored path (stored dump -> prediction -> numbers); CPU, a second, no data.

Every module of this problem that evaluation/adapter.py imports, directly or through another module (EVERY import: inside a function, a self-test or
an `if __name__` block too), must be in `adapter.EVALUATOR_FILES`: an edit of such a file can move a scored number, and only a changed hash keeps the
judge from accepting a `from` control across it. NOT_SCORED names the modules where the walk stops, each with the reason why no scored number passes
through it; an entry the walk never reaches must be taken out. A file in the list that the path does not import is refused too. An import runs the
__init__.py of every package above the module: those are on the path as well (empty today; a line added to one changes the hash).

    python -P -m lane_center.evaluation.tests.test_evaluator_hash        (with the module path: README.md; `research tests lane_center` runs it)"""
import ast
import os

from lane_center.evaluation import adapter

ROOT = adapter.ROOT
PKG = os.path.basename(ROOT)                                       # lane_center: a module is <PKG>.<folders>.<module>
NOT_SCORED = {
    f"{PKG}.training.bflc_paths": "where code, data and runs are: no number",
    f"{PKG}.net.model.jtoken": "the token head: `lane_readout.batch_dump` reads its `probs` / `decode` when the predict step WRITES a dump; reading and decoding a stored dump calls neither",
    f"{PKG}.net.prep.lane_tok_data": "the GT tables of the token cache build and of the label step (`lane_tokcache.label_frame`, `build`): a stored dump is decoded without them",
    f"{PKG}.net.model.jtok": "the pair gate of the token cache build's command line (`lane_tokcache.py build`)",
    f"{PKG}.net.model.lane_net": "the network of the token cache build's command line (`lane_tokcache.py build`)",
}


def modules():
    """dotted module name -> file, for every module of the package (its records and documents left out)"""
    out = {}
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = sorted(d for d in dn if d not in ("__pycache__", "experiments", "literature"))
        for f in sorted(fn):
            if f.endswith(".py") and f != "__init__.py":
                out[".".join([PKG] + os.path.relpath(os.path.join(dp, f[:-3]), ROOT).split(os.sep))] = os.path.join(dp, f)
    return out


def imported(path, mods):
    """the files of this problem that the file `path` imports (every import statement; `from <package> import <module>` names the module)"""
    found = set()
    for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
        if isinstance(node, ast.Import):
            found |= {mods[a.name] for a in node.names if a.name in mods}
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                name = f"{node.module}.{a.name}"
                if name in mods or node.module in mods:
                    found.add(mods.get(name) or mods[node.module])
    return found


def packages(path):
    """the __init__.py files that an import of the module file `path` runs: of its folder and every folder above it up to the problem folder"""
    d, out = os.path.dirname(path), []
    while os.path.normpath(d).startswith(os.path.normpath(ROOT)):
        out.append(os.path.join(d, "__init__.py"))
        d = os.path.dirname(d)
    return out


def scored_path(start, mods, extra=None):
    """the files reached from `start` through the imports (with the __init__.py files they run), the walk stopped at NOT_SCORED -> (files, the
    NOT_SCORED modules that were reached). extra = {file: [more files it imports]}: planted imports (the proof that the walk bites)"""
    stop = {os.path.normpath(mods[n]): n for n in NOT_SCORED if n in mods}
    seen, hit, todo = set(), set(), [os.path.normpath(start)]
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        if f in stop:
            hit.add(stop[f])
            continue
        seen.add(f)
        todo += [os.path.normpath(p) for p in sorted(imported(f, mods)) + packages(f) + list((extra or {}).get(f, []))]
    return seen, hit


def main():
    mods = modules()
    rel = lambda fs: sorted(os.path.relpath(f, ROOT).replace(os.sep, "/") for f in fs)  # noqa: E731
    listed = {os.path.normpath(f) for f in adapter.EVALUATOR_FILES}
    assert len(listed) == len(adapter.EVALUATOR_FILES) and all(os.path.isfile(f) for f in listed), "EVALUATOR_FILES names a file twice or a file that does not exist"
    path, hit = scored_path(mods[f"{PKG}.evaluation.adapter"], mods)
    assert not path - listed, f"on the scored path and not in the evaluator hash (adapter.EVALUATOR_FILES): {rel(path - listed)}"
    assert not listed - path, f"in the evaluator hash and not on the scored path: {rel(listed - path)}"
    assert hit == set(NOT_SCORED), f"NOT_SCORED names modules the scored path does not reach (take them out): {sorted(set(NOT_SCORED) - hit)}"
    # the walk bites: a module that a file of the path starts to import is reported; an import inside a function and a module taken from its package count
    readout = os.path.normpath(mods[f"{PKG}.net.model.lane_readout"])
    planted, _ = scored_path(mods[f"{PKG}.evaluation.adapter"], mods, extra={readout: [mods[f"{PKG}.net.model.qnodes"]]})
    assert rel(planted - listed) == ["net/model/qnodes.py"], rel(planted - listed)
    assert os.path.normpath(mods[f"{PKG}.evaluation.lane_dump"]) in imported(mods[f"{PKG}.evaluation.adapter"], mods), "an import inside a function counts (adapter reads the dump's reader there)"
    task = f"{PKG}.evaluation.lane_task"
    assert mods[f"{task}.measures"] in imported(mods[f"{PKG}.evaluation.qgraph_eval"], mods) and mods[f"{task}.geometry"] in imported(mods[f"{task}.measures"], mods)
    print(f"test_evaluator_hash passed: the {len(listed)} files of the scored path are the evaluator hash {adapter.evaluator_now()} ({', '.join(rel(listed))}); "
          f"the walk stops at {sorted(NOT_SCORED)}")


if __name__ == "__main__":
    main()
