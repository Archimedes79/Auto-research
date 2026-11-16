import os
P = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "jtest_eval.py")
s = open(P, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


a = s.index("def _imports(path):")
b = s.index("def code_dirs():")
s = s[:a] + '''def _executed(node):
    """the nodes that run when a module is imported: not the bodies of functions and lambdas"""
    for child in ast.iter_child_nodes(node):
        yield child
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            yield from _executed(child)


def _imports(path, lazy=True):
    """(level, module name) of every import statement of a source file, with `lazy` also those inside functions (they run when the function is called); `from m import n` also as m.n (n may be a submodule)"""
    key = _stamp(path) + (lazy,)
    if key not in _IMP:
        try:
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except (OSError, SyntaxError, UnicodeDecodeError):
            _IMP[key] = []
            return []
        out = []
        for n in (ast.walk(tree) if lazy else _executed(tree)):
            if isinstance(n, ast.Import):
                out += [(0, a.name) for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                base = n.module or ""
                out += [(n.level, base)] + [(n.level, f"{base}.{a.name}" if base else a.name) for a in n.names]
        _IMP[key] = out
    return _IMP[key]


''' + s[b:]
rep('''    """sha256 of every file of the lane code this module runs, keyed by the path from the repository root: the static closure of the `import` statements (also those inside functions) of `entry` (default: this file), each looked up in every folder of
    `code_dirs()` (a name found in several folders counts for all of them), with the packages above a dotted name; library modules are not followed, nor those of CODE_SKIP. A superset of what a run loads"""''',
    '''    """sha256 of every file of the lane code this module runs, keyed by the path from the repository root: the static closure of the `import` statements of `entry` (default: this file), each looked up in every folder of `code_dirs()` (a name found in
    several folders counts for all of them), with the packages above a dotted name; library modules are not followed, nor those of CODE_SKIP. The imports inside functions count for the lane's own code, for the shared library `autoresearch/` (the
    Evaluation package, whose lazy parts a run does not call) only those that run at import. A superset of what a run loads"""''')
rep('''    found, todo = set(), [os.path.realpath(entry or os.path.abspath(__file__))]
    while todo:
        p = todo.pop()
        if p in found:
            continue
        found.add(p)
        for level, name in _imports(p):''', '''    repo = os.path.realpath(os.path.join(ROOT, ".."))
    found, todo = set(), [os.path.realpath(entry or os.path.abspath(__file__))]
    while todo:
        p = todo.pop()
        if p in found:
            continue
        found.add(p)
        for level, name in _imports(p, lazy=os.path.relpath(p, repo).split(os.sep)[0] != "autoresearch"):''')
rep('''            todo += hits                                                                               # the folders are real paths already (code_dirs): no realpath per hit, it is a lstat per path component
    repo = os.path.realpath(os.path.join(ROOT, ".."))
    return {''', '''            todo += hits                                                                               # the folders are real paths already (code_dirs): no realpath per hit, it is a lstat per path component
    return {''')
open(P, "w", encoding="utf-8").write(s)
print("ok")
