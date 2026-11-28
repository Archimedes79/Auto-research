"""The style of a problem that is a package, checked: `check(problem_dir)` lists what breaks it ([] = clean). A problem's own layering test
calls it and adds its layer table (`imports` gives the edges) and the check of what is loaded (`identity_errors`). Standard library only.

P1 packages      every folder from a module up to the problem folder holds an __init__.py, and it is empty; problem.json has no
                 "pythonpath"; a step is `{python} -P -m <problem>.<module> ...`, a test `<problem>.<module> [args]`
P2 one name      a module of the problem is imported absolutely, as <problem>.<folders>.<module>, also through importlib.import_module
                 with a fixed name: not flat (`import qgraph`), not from the problem folder (`from net.model import x`), not relative,
                 not through the repository (`problems.<p>`, `autoresearch.<x>`); it exists; and no other problem's package is imported
                 (a snapshot holds only its own problem). "A module of the problem" is decided by its tree: a name whose whole dotted
                 path ends a module of the tree is one, whatever else is called so (the standard library, an installed package)
P3 no path work  no sys.path (under any alias, `os.sys.path`), site.addsitedir, runpy.run_path, spec_from_file_location,
                 SourceFileLoader, imp or a package's __path__; no environment name "PYTHONPATH" or "BFLC_ROOT" (as a key, a keyword
                 or `NAME=`; a text that only mentions it passes); no os.chdir at import time. A line that must (the path of a foreign
                 clone, a helper that runs a mutated copy) carries the comment `# path work: <why>`

Why: a regular package is bound to ONE folder, so a second root on the module path (an install, the working tree behind a snapshot) can
neither add nor shadow a module of a run, and -P keeps the working folder off the path (METHOD.md, hard rule 3). The module path is the
roots `paths.module_path` gives, per process."""
from __future__ import annotations

import ast
import importlib
import json
import os
import sys
from typing import Dict, Iterator, List, Sequence

SKIP = ("experiments", "literature")                            # folders of the problem that are records and documents, not its package
SCRATCH = ("__pycache__", "temp", "_to_delete")                 # never code, at any depth
MARK = "# path work:"                                           # on a line, with the reason: this line may work on the module path
LOADERS = ("addsitedir", "spec_from_file_location", "SourceFileLoader", "__path__")
ENV_NAMES = ("PYTHONPATH", "BFLC_ROOT")
WHY = "a module of the problem is imported as {name}.<folders>.<module>"


def files(problem_dir: str, skip: Sequence[str] = SKIP) -> List[str]:
    """the python files of the problem, relative to its folder with /"""
    out = []
    for dirpath, dirs, names in os.walk(problem_dir):
        rel = os.path.relpath(dirpath, problem_dir).replace(os.sep, "/")
        below = lambda x: x if rel == "." else f"{rel}/{x}"                                   # noqa: E731
        dirs[:] = sorted(d for d in dirs if d not in SCRATCH and below(d) not in skip)
        out += [below(f) for f in sorted(names) if f.endswith(".py")]
    return out


def dotted(name: str, rel: str) -> str:
    """net/model/net.py -> <name>.net.model.net; an __init__.py is its folder"""
    parts = rel[:-3].split("/")
    return ".".join([name] + (parts[:-1] if parts[-1] == "__init__" else parts))


def _layout(problem_dir: str, name: str, py: List[str]) -> List[str]:
    errs, folders = [], set()
    for f in py:
        d = os.path.dirname(f)
        while d not in folders:
            folders.add(d)
            d = os.path.dirname(d)
    errs += [f"{d or '.'}/: modules and no __init__.py (a folder without one is open to a second root on the module path)"
             for d in sorted(folders) if not os.path.isfile(os.path.join(problem_dir, d, "__init__.py"))]
    errs += [f"{f}: not empty (an __init__.py holds nothing: a module is imported by its own name)"
             for f in py if os.path.basename(f) == "__init__.py" and os.path.getsize(os.path.join(problem_dir, f)) > 0]
    with open(os.path.join(problem_dir, "problem.json"), encoding="utf-8") as fh:
        p = json.load(fh)
    if "pythonpath" in p:
        errs.append('problem.json: the key "pythonpath" (a package needs no search path)')
    for step, s in p.get("steps", {}).items():
        run = s.get("run") if isinstance(s, dict) else None
        if isinstance(run, list) and not (run[:3] == ["{python}", "-P", "-m"] and str(run[3:4] and run[3]).startswith(name + ".")):
            errs.append(f"problem.json steps.{step}: `{' '.join(map(str, run[:4]))}` is not `{{python}} -P -m {name}.<module>`")
    errs += [f"problem.json tests: {t!r} is not `{name}.<module> [args]`" for t in p.get("tests", []) if not str(t).startswith(name + ".")]
    return errs


def _imported(tree: ast.AST) -> Iterator[tuple]:
    """(line, module as written, the names taken from it or None for `import x`, dots of a relative import); a dynamic import of a fixed
    name counts as `import x`"""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            yield node.lineno, node.module or "", [a.name for a in node.names], node.level
        elif isinstance(node, ast.Import):
            for a in node.names:
                yield node.lineno, a.name, None, 0
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str) \
                and ast.unparse(node.func).rsplit(".", 1)[-1] in ("import_module", "__import__"):
            text = node.args[0].value
            yield node.lineno, text.lstrip("."), None, len(text) - len(text.lstrip("."))


def _at_import(tree: ast.Module) -> Iterator[ast.AST]:
    """the nodes that run when the module is imported: not the bodies of functions, not the `if __name__ == "__main__":` block"""
    todo = list(tree.body)
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) or (isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)):
            continue
        yield node
        todo.extend(ast.iter_child_nodes(node))


def _bound(tree: ast.AST) -> Dict[str, str]:
    """local name -> the dotted name an import binds it to: `import sys as s` s -> sys, `from os import chdir` chdir -> os.chdir"""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update({a.asname or a.name.split(".")[0]: a.name if a.asname else a.name.split(".")[0] for a in node.names})
        elif isinstance(node, ast.ImportFrom) and not node.level:
            out.update({a.asname or a.name: f"{node.module}.{a.name}" for a in node.names})
    return out


def _meant(node: ast.AST, bound: Dict[str, str]) -> str:
    """the dotted name an expression names, through the import aliases: s.path -> sys.path, chdir -> os.chdir"""
    head, dot, rest = ast.unparse(node).partition(".")
    return bound.get(head, head) + dot + rest


def _path_work(tree: ast.Module) -> Iterator[tuple]:
    """(line, what) of everything that works on the module path or loads code by its file"""
    bound = _bound(tree)
    doc = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
           and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            yield from ((node.lineno, "the module imp") for m in mods if m.split(".")[0] == "imp")
        elif isinstance(node, (ast.Attribute, ast.Name)):
            full = _meant(node, bound)
            if full in ("sys.path", "runpy.run_path") or full.endswith(".sys.path") or full.rsplit(".", 1)[-1] in LOADERS:
                yield node.lineno, full
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc:
            yield from ((node.lineno, f"the environment name {word}") for word in ENV_NAMES if node.value == word or f"{word}=" in node.value)
        elif isinstance(node, ast.keyword) and node.arg in ENV_NAMES:
            yield node.value.lineno, f"the environment name {node.arg} (a keyword)"
    for node in _at_import(tree):
        if isinstance(node, ast.Call) and _meant(node.func, bound) == "os.chdir":
            yield node.lineno, "os.chdir at import time"


def _scan(problem_dir: str, skip: Sequence[str]):
    """-> (the violations of P2 and P3, file -> the problem's modules it imports)"""
    name = os.path.basename(os.path.abspath(problem_dir))
    py = files(problem_dir, skip)
    modules = {dotted(name, f) for f in py if os.path.basename(f) != "__init__.py"}
    packages = {k.rsplit(".", n)[0] for k in modules for n in range(1, k.count(".") + 1)}      # the folders above them (P1 says which lack an __init__.py)
    known = modules | packages
    base = os.path.dirname(os.path.abspath(problem_dir))
    others = {d for d in os.listdir(base) if d != name and os.path.isfile(os.path.join(base, d, "problem.json"))}
    errs, uses = [], {}
    for f in py:
        with open(os.path.join(problem_dir, f), encoding="utf-8") as fh:
            text = fh.read()
        tree, lines = ast.parse(text, f), text.split("\n")
        mine = set()
        for line, module, taken, level in _imported(tree):
            top = module.split(".")[0]
            if level:
                why = "relative"
            elif top == name:
                lost = [f"{module}.{t}" for t in (taken or []) if module in packages and t != "*" and f"{module}.{t}" not in known]
                why = "no such module" if module not in known or lost else ""       # a name taken from a package (its __init__.py is empty) is a module
                mine |= {f"{module}.{t}" if f"{module}.{t}" in known else module for t in (taken or [""])} if not why else set()
            elif top in ("problems", "autoresearch"):
                why = "through the repository (the roots are problems/ and autoresearch/ themselves)"
            elif top in others:
                why = "another problem's package (a snapshot holds only its own problem)"
            elif f"{name}.{module}" in known:
                why = "from the problem folder"
            elif any(k.endswith("." + module) for k in known):
                why = "flat"
            else:
                continue                                                             # no module of this tree
            if why:
                errs.append(f"{f}:{line}: import of {'.' * level}{module} is {why}: {WHY.format(name=name)}")
        uses[f] = sorted(mine)
        errs += [f"{f}:{line}: {what}: the module path is the roots `research` gives, a module is loaded by its name (a line that must: `{MARK} <why>`)"
                 for line, what in _path_work(tree) if MARK not in lines[line - 1]]
    return errs, uses


def check(problem_dir: str, skip: Sequence[str] = SKIP) -> List[str]:
    """the violations of P1 - P3 as texts; `skip`: folders of the problem that are no code of the package"""
    name = os.path.basename(os.path.abspath(problem_dir))
    return _layout(problem_dir, name, files(problem_dir, skip)) + _scan(problem_dir, skip)[0]


def imports(problem_dir: str, skip: Sequence[str] = SKIP) -> Dict[str, List[str]]:
    """file -> the modules of the problem it imports, as dotted names (the edges a layer table judges)"""
    return {f: mods for f, mods in _scan(problem_dir, skip)[1].items() if os.path.basename(f) != "__init__.py"}


def identity_errors(problem_dir: str, skip: Sequence[str] = SKIP) -> List[str]:
    """imports every module of the problem by its dotted name (the caller was started with -P -m; its own file is left out), then lists
    what has more than one identity: a file loaded as two modules, a module of the problem's folder under another name than <problem>.<...>,
    a module named <problem>.<...> that lies elsewhere"""
    here = os.path.realpath(problem_dir)
    name = os.path.basename(here)
    main = os.path.realpath(getattr(sys.modules.get("__main__"), "__file__", None) or os.devnull)
    for f in files(problem_dir, skip):
        if os.path.basename(f) != "__init__.py" and os.path.join(here, f) != main:
            importlib.import_module(dotted(name, f))
    loaded: Dict[str, Dict[int, str]] = {}                                                    # file -> the module objects made from it
    errs = []
    for mod_name, mod in sorted(sys.modules.items()):
        file = os.path.realpath(getattr(mod, "__dict__", {}).get("__file__") or os.devnull)   # read, not asked: a lazy module imports on a question
        inside, named = file.startswith(here + os.sep), mod_name.split(".")[0] == name
        if inside:
            loaded.setdefault(file, {})[id(mod)] = mod_name
        if inside and not named and mod is not sys.modules.get("__main__"):                    # multiprocessing names the started module __mp_main__ as well
            errs.append(f"{file} is loaded as {mod_name}, not as {name}.<...>")
        elif named and not inside and file != os.devnull:
            errs.append(f"{mod_name} is loaded from {file}, outside {here}")
    return errs + [f"{file} is loaded as {' and '.join(sorted(names.values()))}" for file, names in sorted(loaded.items()) if len(names) > 1]
