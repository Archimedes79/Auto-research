"""one-off: the fixes of the second critic pass on provenance.py (bundle scopes, repository files, repro, trainer interpreter)"""
p = "autoresearch/tools/provenance.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


# 6 + 7: scopes: `== "__main__"` only (body), module-level functions only
i, j = s.index("def _imports(tree):"), s.index("def import_closure(")
s = s[:i] + '''def _is_main_test(test):
    import ast
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__" and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value == "__main__")


def _imports(tree):
    """[(import node, in_test_scope)] of every import of a module, function level included. In a test scope = inside a MODULE-LEVEL function named
    in TEST_SCOPES (a method or a nested function of that name is part of the network) or in the body of an `if __name__ == "__main__"` block"""
    import ast
    out = []

    def visit(c, scoped, top):
        if isinstance(c, (ast.Import, ast.ImportFrom)):
            out.append((c, scoped))
            return
        if top and isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)) and c.name in TEST_SCOPES:
            scoped = True
        if isinstance(c, ast.If) and _is_main_test(c.test):
            for st in c.body:
                visit(st, True, False)
            for st in c.orelse:
                visit(st, scoped, False)
            return
        for ch in ast.iter_child_nodes(c):
            visit(ch, scoped, False)
    for st in tree.body:
        visit(st, False, True)
    return out


def _names(n, pkg):
    """(candidate dotted names, absolute?) of an import node found in a module of package `pkg`"""
    import ast
    if isinstance(n, ast.Import):
        return [a.name for a in n.names], True
    base = n.module or ""
    if n.level:
        up = pkg.split(".") if pkg else []
        up = up[:len(up) - (n.level - 1)] if n.level > 1 else up
        base = ".".join(up + ([n.module] if n.module else []))
    return [base] + [f"{base}.{a.name}" if base else a.name for a in n.names], not n.level


''' + s[j:]

# 2: one listing of the tag instead of two git calls per candidate
sub('''def repo_modules(dotted_names, repo, ref=None):''', '''_TREES = {}


def tree_names(repo, ref):
    """the set of file paths of a tag (one `git ls-tree -r`), cached; empty when git cannot say"""
    key = (repo, ref)
    if key not in _TREES:
        try:
            r = subprocess.run(["git", "--no-optional-locks", "-C", repo, "ls-tree", "-r", "--name-only", "-z", ref], capture_output=True, timeout=120)
            _TREES[key] = {x.decode("utf-8", "replace") for x in r.stdout.split(b"\\0") if x} if r.returncode == 0 else set()
        except (OSError, subprocess.SubprocessError):
            _TREES[key] = set()
    return _TREES[key]


def repo_modules(dotted_names, repo, ref=None):''')
sub('''        return git_bytes(repo, ref, rel.replace("\\\\", "/")) is not None if ref else os.path.isfile(os.path.join(repo, rel))''', '''        return rel in tree_names(repo, ref) if ref else os.path.isfile(os.path.join(repo, rel))''')

# 1 + 10 + 11: the imports of the repository files are listed, the temp folder is always removed, the tag reading is honest
sub('''    keep_tmp = tempfile.mkdtemp() if repo_ref else None      # the files of the tag, written out so that they can be hashed and copied like the others
    repo_items = []
    for rel in repo_files:
        if repo_ref:
            full = os.path.join(keep_tmp, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            open(full, "wb").write(git_bytes(repo, repo_ref, rel) or b"")
        else:
            full = os.path.join(repo, rel)
        repo_items.append(("_repo/" + rel, full))
    items = [(rel, os.path.join(code_dir, rel)) for rel in files] + repo_items
''', '''    keep_tmp = tempfile.mkdtemp() if repo_ref else None      # the files of the tag, written out so that they can be hashed and copied like the others
    try:
        return _bundle_rest(code_dir, roots, into, config, exclude, repo_ref, repo, files, info, repo_files, local, keep_tmp)
    finally:
        if keep_tmp:
            shutil.rmtree(keep_tmp, ignore_errors=True)


def _bundle_rest(code_dir, roots, into, config, exclude, repo_ref, repo, files, info, repo_files, local, keep_tmp):
    import ast
    repo_items = []
    for rel in repo_files:
        if repo_ref:
            data = git_bytes(repo, repo_ref, rel)
            if data is None:
                continue
            full = os.path.join(keep_tmp, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            open(full, "wb").write(data)
        else:
            full = os.path.join(repo, rel)
        repo_items.append(("_repo/" + rel, full))
    got = [label[len("_repo/"):] for label, _ in repo_items]
    std = getattr(sys, "stdlib_module_names", ())
    unfollowed = set()                                       # what the repository files import and the bundle does not carry (autogt/__init__.py pulls in half of autogt)
    for label, full in repo_items:
        rel = label[len("_repo/"):]
        mod = rel[:-3].replace("/", ".")
        pkg = mod[:-9] if mod.endswith(".__init__") else (mod.rsplit(".", 1)[0] if "." in mod else "")
        try:
            tree = ast.parse(open(full, encoding="utf-8-sig").read())
        except (OSError, SyntaxError, ValueError):
            unfollowed.add(f"{rel} (does not parse)")
            continue
        names = set()
        for n, scoped in _imports(tree):
            if not scoped:
                names.update(c for c in _names(n, pkg)[0] if c and c.split(".")[0] not in std)
        for f in repo_modules(sorted(names), repo, repo_ref):
            if f not in got:
                unfollowed.add(f"{rel} -> {f}")
    items = [(rel, os.path.join(code_dir, rel)) for rel in files] + repo_items
''')
sub('''"repo_files": repo_files, "roots": list(roots),''', '''"repo_files": got, "repo_imports_unfollowed": sorted(unfollowed), "roots": list(roots),''')
sub('''"repo_files_from": repo_ref or "the working tree"}''', '''"repo_files_from": (repo_ref if got or not repo_files else f"{repo_ref} (the files named are not in that tag)") if repo_ref else "the working tree"}''')
sub('''                out["copied_to"] = os.path.basename(into)
    if keep_tmp:
        shutil.rmtree(keep_tmp, ignore_errors=True)
    return out''', '''                out["copied_to"] = os.path.basename(into)
    return out''')
sub('''    into = a folder to copy them to (results/<id>/bundle/, repository files under _repo/), so the model can be rebuilt from that folder and the
    checkpoint alone; it is made next to the target''', '''    into = a folder to copy them to (results/<id>/bundle/, repository files under _repo/): the files that define the network and its losses, to read and
    to hash, not an installable package (see repo_imports_unfollowed); it is made next to the target''')
sub('''dynamic_imports; the imports of the repository files themselves are not followed. repo_ref''', '''dynamic_imports; the imports of the repository files themselves are not followed but listed (repo_imports_unfollowed: `autogt/__init__.py` imports
    more of autogt, which the bundle does not carry: its version is the run's GT version and code tag). repo_ref''')

# 3 + 4 + 5: repro_prepare
sub('''    dest = os.path.abspath(dest or os.path.join(repo, "temp", "repro", run_id))
    if not os.path.isdir(dest):''', '''    dest = os.path.abspath(dest or os.path.join(repo, "temp", "repro", run_id))
    if os.path.isdir(dest):                                          # a leftover folder is reused only when it is exactly this tag
        have = (git(dest, "rev-parse", "HEAD") or "").strip()
        want = (git(repo, "rev-parse", f"{tag}^{{commit}}") or "").strip()
        if not have or have != want:
            sys.exit(f"{dest} exists and is not a checkout of {tag} (HEAD {have[:10] or 'none'}, wanted {want[:10]}): remove it with `git worktree remove --force` or pass another --dir")
    if not os.path.isdir(dest):''')
sub('''    for rel in ("data", os.path.join(os.path.relpath(P.root(), repo), "data")):
        src, dst = os.path.join(repo, rel), os.path.join(dest, rel)
        if os.path.isdir(src) and not os.path.lexists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst)
    cwd = r.get("cwd") or P.root()
    rel = os.path.relpath(cwd, repo) if os.path.commonpath([os.path.abspath(cwd), repo]) == repo else os.path.relpath(P.root(), repo)''', '''    notes = []
    for rel in ("data", os.path.join(os.path.relpath(P.root(), repo), "data")):
        src, dst = os.path.join(repo, rel), os.path.join(dest, rel)
        if not os.path.isdir(src):
            continue
        try:
            if not os.path.lexists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.symlink(src, dst)
            elif os.path.isdir(dst) and not os.path.islink(dst):     # a partly tracked folder (README, split files): link what is not tracked into it
                for child in os.listdir(src):
                    if not os.path.lexists(os.path.join(dst, child)):
                        os.symlink(os.path.join(src, child), os.path.join(dst, child))
        except OSError as e:
            notes.append(f"could not link {rel}: {e}")
    cwd = r.get("cwd") or P.root()
    try:
        inside_repo = os.path.normcase(os.path.commonpath([os.path.abspath(cwd), repo])) == os.path.normcase(repo)
    except ValueError:
        inside_repo = False
    if not inside_repo:
        notes.append(f"the recorded cwd {cwd} is not inside {repo}: the problem folder is used")
    rel = os.path.relpath(cwd, repo) if inside_repo else os.path.relpath(P.root(), repo)''')
sub('''    return dest, cmd


def find_run_dir''', '''    if notes:
        print("WARNING: " + "; ".join(notes), file=sys.stderr)
    return dest, cmd


def find_run_dir''')

# 8 + 9: the trainer interpreter
sub('''        out = subprocess.run([py, "-c", PROBE], capture_output=True, text=True, timeout=30)
        d = json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}''', '''        out = subprocess.run([py, "-I", "-c", PROBE], capture_output=True, text=True, timeout=30, cwd=tempfile.gettempdir())
        d = json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}
    if not isinstance(d, dict):
        return {}''')
sub('''                cands += re.findall(r"(~?/[\\w./-]*/bin/python[\\d.]*)", open(f, encoding="utf-8", errors="replace").read())''', '''                text = open(f, encoding="utf-8", errors="replace").read().replace("${HOME}", "~").replace("$HOME", "~")
                code = "\\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))      # a commented-out old interpreter does not count
                cands += re.findall(r"(~?/[\\w./-]*/bin/python[\\d.]*)", code)''')

# selftest: one case each
sub('''        # the model bundle (network + losses = the import closure of model.py and losses.py) and the manifest of a training run''', '''        # scopes: `!=` and else are no test scope, a method or nested function named main is part of the network, a module-level _selftest is a test scope
        import ast as _ast
        tr_ = _ast.parse("if __name__ != '__main__':\\n    import a1\\nif __name__ == '__main__':\\n    import a2\\nelse:\\n    import a3\\nclass C:\\n    def main(self):\\n        import a4\\n"
                         "def forward():\\n    def main():\\n        import a5\\ndef main():\\n    import a6\\n")
        got_ = {n.names[0].name: sc for n, sc in _imports(tr_)}
        assert got_ == {"a1": False, "a2": True, "a3": False, "a4": False, "a5": False, "a6": True}, got_
        # the interpreter probe ignores a commented-out interpreter, expands $HOME, and takes only a dict answer
        fakehome = os.path.join(tmp, "fh")
        os.makedirs(os.path.join(fakehome, "env", "bin"))
        open(os.path.join(fakehome, "env", "bin", "python"), "w").write("#!/bin/sh\\necho '[1]'\\n")
        os.chmod(os.path.join(fakehome, "env", "bin", "python"), 0o755)
        open(os.path.join(tmp, "tr.sh"), "w").write("#!/bin/bash\\n# old: " + os.path.join(fakehome, "gone", "bin", "python") + "\\n" + os.path.join(fakehome, "env", "bin", "python") + " -u t.py\\n")
        assert trainer_python({"command": "bash tr.sh", "cwd": tmp}) == os.path.join(fakehome, "env", "bin", "python"), "the live line, not the comment"
        assert trainer_env({"command": "bash tr.sh", "cwd": tmp}) == {}, "an answer that is not a dict reads unknown"
        # the model bundle (network + losses = the import closure of model.py and losses.py) and the manifest of a training run''')
sub('''            run("add", "autogt")
            run("commit", "-q", "-m", "autogt")
            run("tag", "tg/layers")''', '''            put(tmp, "autogt/__init__.py", "from .layers import AREAS\\nfrom . import sets\\n")
            put(tmp, "autogt/sets.py", "S = 1\\n")
            run("add", "autogt")
            run("commit", "-q", "-m", "autogt")
            run("tag", "tg/layers")''')
sub('''            assert bundle(s3, repo_ref="tg/layers")["code_hash"] == bt["code_hash"] and bundle(s3, repo_ref="tg/nonexistent")["repo_files"] == [], "a ref without the files finds none"''', '''            assert bundle(s3, repo_ref="tg/layers")["code_hash"] == bt["code_hash"] and bundle(s3, repo_ref="tg/nonexistent")["repo_files"] == [], "a ref without the files finds none"
            assert bt["repo_imports_unfollowed"] == ["autogt/__init__.py -> autogt/sets.py"], bt["repo_imports_unfollowed"]   # the package __init__ imports more than the bundle carries
            assert bundle(s3, repo_ref="tg/nonexistent")["repo_files_from"] == "tg/nonexistent" or "not in that tag" in bundle(s3, repo_ref="tg/nonexistent")["repo_files_from"]
            leaked = [d for d in os.listdir(tempfile.gettempdir()) if d.startswith("tmp") and os.path.isdir(os.path.join(tempfile.gettempdir(), d, "autogt"))]
            assert not leaked, "the temp folder of the tag's files is removed"''')
sub('''            assert repro_prepare("r1")[0] == wt, "a second call reuses the worktree"''', '''            assert repro_prepare("r1")[0] == wt, "a second call reuses the worktree"
            os.makedirs(os.path.join(tmp, "elsewhere_wt"))
            try:
                repro_prepare("r1", os.path.join(tmp, "elsewhere_wt"))
                raise AssertionError("a leftover folder that is not a checkout of the tag must be refused")
            except SystemExit as e:
                assert "is not a checkout of" in str(e), e''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
