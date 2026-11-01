TEST_SCOPES = ("_selftest", "selftest", "_cli", "cli", "main", "_main")   # functions beside the network (tests, command-line tools): their imports are listed, not followed


def _imports(tree):
    """[(import node, in_test_scope)] of every import of a module, function level included. In a test scope = inside a function named in TEST_SCOPES
    or an `if __name__ == "__main__"` block"""
    import ast
    out = []

    def walk(n, scoped):
        for c in ast.iter_child_nodes(n):
            s = scoped
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)) and c.name in TEST_SCOPES:
                s = True
            elif isinstance(c, ast.If) and isinstance(c.test, ast.Compare) and isinstance(c.test.left, ast.Name) and c.test.left.id == "__name__":
                s = True
            if isinstance(c, (ast.Import, ast.ImportFrom)):
                out.append((c, s))
            walk(c, s)
    walk(tree, False)
    return out


def import_closure(code_dir, roots=BUNDLE_ROOTS, exclude=(), info=None):
    """the files (relative, sorted) of code_dir that the root modules import, transitively: the network and its losses. Only modules found under
    code_dir count; a parent package of an imported module counts too (python runs its __init__). A module in `exclude` (the training loop, the
    data loaders) is not followed and not in the result. Imports inside a test scope (a `_selftest` or `_cli` function, an `if __name__` block) are
    not followed. `info` (a dict, optional) collects what the closure cannot say by itself: "skipped" ("importer -> excluded module"), "test_imports"
    ("importer -> module" found only in a test scope), "unparsed" (files that do not parse: their imports are unknown), "external" (top-level names
    of absolute imports that are neither in code_dir nor in the standard library), "external_mods" (their dotted names), "dynamic" (files that
    import by name at run time)"""
    import ast
    info = info if info is not None else {}
    for k in ("skipped", "test_imports", "unparsed", "external", "external_mods", "dynamic"):
        info.setdefault(k, set())
    mods = {}
    for dp, dn, fn in os.walk(code_dir):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn:
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(dp, f), code_dir).replace("\\", "/")
                name = rel[:-3].replace("/", ".")
                mods[name[:-9] if name.endswith(".__init__") else name] = rel
    byfile = {v: k for k, v in mods.items()}
    todo, seen = [r for r in roots if r in byfile], set()
    std = getattr(sys, "stdlib_module_names", ())
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        try:
            tree = ast.parse(open(os.path.join(code_dir, rel), encoding="utf-8-sig").read())   # utf-8-sig: a BOM (Notepad) is no syntax error
        except (OSError, SyntaxError, ValueError):
            info["unparsed"].add(rel)
            continue
        name = byfile[rel]
        pkg = name if rel.endswith("__init__.py") else (name.rsplit(".", 1)[0] if "." in name else "")
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and ((isinstance(n.func, ast.Name) and n.func.id == "__import__") or (isinstance(n.func, ast.Attribute) and n.func.attr == "import_module")):
                info["dynamic"].add(rel)
        for n, scoped in _imports(tree):
            cands, absolute = [], False
            if isinstance(n, ast.Import):
                cands, absolute = [a.name for a in n.names], True
            else:
                base = n.module or ""
                absolute = not n.level
                if n.level:
                    up = pkg.split(".") if pkg else []
                    up = up[:len(up) - (n.level - 1)] if n.level > 1 else up
                    base = ".".join(up + ([n.module] if n.module else []))
                cands = [base] + [f"{base}.{a.name}" if base else a.name for a in n.names]
            for c in cands:
                parts = c.split(".") if c else []
                prefixes = [".".join(parts[:i]) for i in range(1, len(parts) + 1)]
                hit = [p for p in prefixes if p in mods]
                if scoped:
                    for p in hit:
                        info["test_imports"].add(f"{rel} -> {mods[p]}")
                    continue
                if absolute and not hit and parts and parts[0] not in std:
                    info["external"].add(parts[0])
                    info["external_mods"].add(c)
                for p in hit:
                    if mods[p] in exclude:
                        info["skipped"].add(f"{rel} -> {mods[p]}")
                    elif mods[p] not in seen:
                        todo.append(mods[p])
    return sorted(seen)


def _hash_files(items):
    """12 hex over (label, absolute path) pairs: the label and the text of each file, line endings normalised, in the order given"""
    h = hashlib.sha256()
    for label, full in items:
        h.update(label.encode() + b"\0" + open(full, "rb").read().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:12]


def repo_modules(dotted_names, repo):
    """the files of the repository that absolute imports like `autogt.layers` name: for each dotted name the longest prefix that is a module file or a
    package under the repository root, plus the __init__.py of its parent packages; relative paths, sorted. Names that are not repository code
    (torch.nn, numpy) find nothing"""
    out = set()
    for dotted in dotted_names:
        parts = dotted.split(".")
        for i in range(len(parts), 0, -1):
            base = os.path.join(repo, *parts[:i])
            f = base + ".py" if os.path.isfile(base + ".py") else (os.path.join(base, "__init__.py") if os.path.isfile(os.path.join(base, "__init__.py")) else None)
            if f:
                out.add(os.path.relpath(f, repo).replace("\\", "/"))
                for j in range(1, i):
                    init = os.path.join(repo, *parts[:j], "__init__.py")
                    if os.path.isfile(init):
                        out.add(os.path.relpath(init, repo).replace("\\", "/"))
                break
    return sorted(out)


def bundle(code_dir, roots=BUNDLE_ROOTS, into=None, config=None, exclude=BUNDLE_EXCLUDE):
    """The model bundle of a run: the network and loss modules (import_closure of `roots` under code_dir), the repository modules they import
    (`repo_files`: autogt/layers.py, the label layout the losses need) with a hash over all of them (line endings normalised), and the run's config.
    into = a folder to copy them to (results/<id>/bundle/, repository files under _repo/), so the model can be rebuilt from that folder and the
    checkpoint alone; it is made next to the target, checked against the hash and moved into place, and an existing non-empty folder is left alone.
    What it still does NOT carry is listed, never hidden: missing_roots, unparsed (their imports are unknown), imports_excluded (the network imports
    the loop or a loader at module level), test_imports (only a selftest or a command-line tool imports them), external_local (repository folders
    named by an import that no file was found for), dynamic_imports; the imports of the repository files themselves are not followed.
    Nothing found reads "unknown", never an empty hash"""
    if not code_dir or not os.path.isdir(code_dir):
        return {"code_hash": UNKNOWN, "files": [], "source": UNKNOWN}
    info = {}
    files = import_closure(code_dir, roots, exclude, info)
    if not files:
        return {"code_hash": UNKNOWN, "files": [], "source": code_dir, "note": f"none of {list(roots)} found"}
    repo = repo_root()
    repo_files = repo_modules(info["external_mods"], repo)
    local = sorted(t for t in info["external"] if (os.path.isdir(os.path.join(repo, t)) or os.path.isdir(os.path.join(P.root(), t))) and not any(f.split("/")[0] == t for f in repo_files))
    items = [(rel, os.path.join(code_dir, rel)) for rel in files] + [("_repo/" + rel, os.path.join(repo, rel)) for rel in repo_files]
    out = {"code_hash": _hash_files(items), "files": files, "repo_files": repo_files, "roots": list(roots), "missing_roots": [r for r in roots if r not in files],
           "excluded": list(exclude), "imports_excluded": sorted(info["skipped"]), "test_imports": sorted(info["test_imports"]), "unparsed": sorted(info["unparsed"]),
           "external_local": local, "dynamic_imports": sorted(info["dynamic"]),
           "source": os.path.basename(os.path.dirname(code_dir.rstrip("/\\"))) + "/" + os.path.basename(code_dir.rstrip("/\\"))}
    if config and os.path.isfile(config):
        out["config_hash"] = file_hash(config)
    if into:
        if os.path.isdir(into) and os.listdir(into):
            out["copy_skipped"] = f"{os.path.basename(into)}/ exists and is not empty: left as it is"
        else:
            tmp = into.rstrip("/\\") + ".tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            for label, full in items:
                dst = os.path.join(tmp, label)
                os.makedirs(os.path.dirname(dst) or tmp, exist_ok=True)
                shutil.copy2(full, dst)
            if config and os.path.isfile(config):
                shutil.copy2(config, os.path.join(tmp, "config.json"))
            if _hash_files([(label, os.path.join(tmp, label)) for label, _ in items]) != out["code_hash"]:
                shutil.rmtree(tmp, ignore_errors=True)
                out["copy_skipped"] = "the copy did not read back equal"
            else:
                if os.path.isdir(into):
                    os.rmdir(into)   # empty (checked above)
                os.replace(tmp, into)
                out["copied_to"] = os.path.basename(into)
    return out


