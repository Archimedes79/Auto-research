"""The evaluator hash covers the scored path (stored dump -> prediction -> numbers); CPU, a second, no data.

Every module of this problem that evaluation/adapter.py imports, directly or through another module (EVERY import: inside a function, a self-test or
an `if __name__` block too), must be in `adapter.EVALUATOR_FILES`: an edit of such a file can move a scored number, and only a changed hash keeps the
judge from accepting a `from` control across it. NOT_SCORED names the modules where the walk stops, each with the reason why no scored number passes
through it; an entry the walk never reaches must be taken out. A file in the list that the path does not import is refused too.

    python evaluation/tests/test_evaluator_hash.py        (with the problem's module path; `research tests lane_center` runs it)"""
import ast
import json
import os

import adapter

ROOT = adapter.ROOT
NOT_SCORED = {
    "bflc_paths": "where code, data and runs are: no number",
    "jtoken": "the token head: `lane_readout.batch_dump` reads its `probs` / `decode` when the predict step WRITES a dump; reading and decoding a stored dump calls neither",
    "lane_tok_data": "the GT tables of the token cache build and of the label step (`lane_tokcache.label_frame`, `build`): a stored dump is decoded without them",
    "jtok": "the pair gate of the token cache build's command line (`lane_tokcache.py build`)",
    "lane_net": "the network of the token cache build's command line (`lane_tokcache.py build`)",
}


def modules():
    """module name -> file for the flat module path of problem.json (`pythonpath`), package name -> its folder"""
    out = {}
    for folder in json.load(open(os.path.join(ROOT, "problem.json"), encoding="utf-8"))["pythonpath"]:
        d = os.path.join(ROOT, *folder.split("/"))
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if f.endswith(".py"):
                out.setdefault(f[:-3], p)
            elif os.path.isfile(os.path.join(p, "__init__.py")):
                out.setdefault(f, p)
    return out


def imported(path, mods):
    """the files of this problem that the file `path` imports (every import statement; a package gives its __init__.py and the submodules named)"""
    found, here = set(), os.path.dirname(path)
    in_package = os.path.isfile(os.path.join(here, "__init__.py"))

    def add(base, names=()):
        """a module file, or a package folder with the names taken from it"""
        if os.path.isdir(base):
            found.add(os.path.join(base, "__init__.py"))
            found.update(os.path.join(base, n + ".py") for n in names if os.path.isfile(os.path.join(base, n + ".py")))
        elif os.path.isfile(base):
            found.add(base)
    for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in mods:
                    add(mods[a.name.split(".")[0]], a.name.split(".")[1:2])
        elif isinstance(node, ast.ImportFrom):
            names = [a.name for a in node.names]
            if node.level and in_package:                                  # from .geometry import X / from . import measures
                add(os.path.join(here, node.module + ".py") if node.module else here, names)
            elif not node.level and node.module and node.module.split(".")[0] in mods:
                base = mods[node.module.split(".")[0]]
                sub = node.module.split(".")[1:2]
                add(os.path.join(base, sub[0] + ".py") if sub and os.path.isdir(base) else base, names)
                if sub and os.path.isdir(base):
                    found.add(os.path.join(base, "__init__.py"))
    return found


def scored_path(start, mods, extra=None):
    """the files reached from `start` through the imports, the walk stopped at NOT_SCORED -> (files, the NOT_SCORED modules that were reached).
    extra = {file: [more files it imports]}: planted imports (the proof that the walk bites)"""
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
        todo += [os.path.normpath(p) for p in sorted(imported(f, mods)) + list((extra or {}).get(f, []))]
    return seen, hit


def main():
    mods = modules()
    rel = lambda fs: sorted(os.path.relpath(f, ROOT).replace(os.sep, "/") for f in fs)  # noqa: E731
    listed = {os.path.normpath(f) for f in adapter.EVALUATOR_FILES}
    assert len(listed) == len(adapter.EVALUATOR_FILES) and all(os.path.isfile(f) for f in listed), "EVALUATOR_FILES names a file twice or a file that does not exist"
    path, hit = scored_path(os.path.join(ROOT, "evaluation", "adapter.py"), mods)
    assert not path - listed, f"on the scored path and not in the evaluator hash (adapter.EVALUATOR_FILES): {rel(path - listed)}"
    assert not listed - path, f"in the evaluator hash and not on the scored path: {rel(listed - path)}"
    assert hit == set(NOT_SCORED), f"NOT_SCORED names modules the scored path does not reach (take them out): {sorted(set(NOT_SCORED) - hit)}"
    # the walk bites: a module that a file of the path starts to import is reported; an import inside a function and inside a package count
    readout = os.path.normpath(mods["lane_readout"])
    planted, _ = scored_path(os.path.join(ROOT, "evaluation", "adapter.py"), mods, extra={readout: [mods["qnodes"]]})
    assert rel(planted - listed) == ["net/model/qnodes.py"], rel(planted - listed)
    assert os.path.normpath(mods["lane_dump"]) in imported(os.path.join(ROOT, "evaluation", "adapter.py"), mods), "an import inside a function counts (adapter reads the dump's reader there)"
    assert os.path.join(mods["lane_task"], "measures.py") in imported(mods["qgraph_eval"], mods) and os.path.join(mods["lane_task"], "geometry.py") in imported(os.path.join(mods["lane_task"], "measures.py"), mods)
    print(f"test_evaluator_hash passed: the {len(listed)} files of the scored path are the evaluator hash {adapter.evaluator_now()} ({', '.join(rel(listed))}); "
          f"the walk stops at {sorted(NOT_SCORED)}")


if __name__ == "__main__":
    main()
