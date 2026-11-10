"""Clean up the scratch of a finished loop (phase P4). Standard library only.

    python autoresearch/tools/clean_temp.py loop19             # dry run: what would go, what live code still imports from it
    python autoresearch/tools/clean_temp.py loop19 --move      # commit + tag the scratch (<prefix>/loop19-scratch), then move it to <trash>/temp_loop19/
    python autoresearch/tools/clean_temp.py loop19 --delete    # commit + tag, then delete (the tag keeps the scripts only: refused when the folder holds any other file, or when no tag holds
                                                               # a file with its CURRENT content (every file is compared by blob id; an old <prefix>/loop19-scratch of an earlier
                                                               # snapshot gets a new tag <prefix>/loop19-scratch.attempt2 for the new one); use --move then)
    python autoresearch/tools/clean_temp.py loop19 --move --ignore-refs   # go on although a document or config still names a file of it (history); --force: only the 6-hour guard
    python autoresearch/tools/clean_temp.py --selftest

Convention: scratch of a loop lives in <scratch>/loopNN/ (also found: <scratch>/analysis/loopNN, <scratch>/launchers/loopNN).
Refuses when the live code folders (project.json live_code, and the other code_dirs such as bev_fusion's tests) import a python module that exists only in
the scratch (promote it first), when a file there (and the project's project.json / README.md) still names a scratch file or folder by path (a shell launcher, a
config, a test, a README: change the reference; a mention of a script that already left is history and does not block, nor does a bare folder in a document or a
comment; --ignore-refs lets the rest go), or when a file changed in the last 6 hours (a job may still use it; --force overrides that only).
"""
import os
import posixpath
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402

IMP = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)


def find(name, temp):
    return [p for p in (os.path.join(temp, name), os.path.join(temp, "analysis", name), os.path.join(temp, "launchers", name)) if os.path.isdir(p)]


def kept_by_snapshot(cfg):
    """is the scratch folder inside one of the code_dirs that git_snapshot commits? (otherwise the tag does not hold the scripts)"""
    norm = lambda x: posixpath.normpath(x.replace(chr(92), "/")).strip("/")
    s = norm(cfg["scratch"])
    return any(norm(d) in (".", "") or s == norm(d) or s.startswith(norm(d) + "/") for d in cfg["code_dirs"])


def tree_blobs(top, ref):
    """{path: blob id} of the files that the tag or commit `ref` holds (-z: a name with a space or a quote is one name); {} when git cannot read it"""
    from run import git
    out = {}
    for ent in git(top, "ls-tree", "-r", "-z", ref).stdout.split("\0"):
        meta, _, path = ent.partition("\t")                      # "<mode> <type> <blob id>" TAB name
        if path:
            out[path] = meta.split()[2]
    return out


def blob_ids(top, rels):
    """the blob id git would store for each working-tree file (the filters of the repository applied, as `git add` does), None where git cannot hash it"""
    from run import git
    out = []
    for i in range(0, len(rels), 100):
        chunk = rels[i:i + 100]
        r = git(top, "hash-object", "--", *chunk)
        ids = r.stdout.split()
        out += ids if r.returncode == 0 and len(ids) == len(chunk) else [None] * len(chunk)
    return out


def not_held(root, ref, folders):
    """the files of the scratch folders whose CURRENT content the tag or commit `ref` does not hold: not in it (a .gitignore rule, or a file that is not a script: the snapshot
    adds the .py / .sh files of the scratch only, a result.json or notes.md is held by no tag), or in it with another blob id (an old tag of an earlier snapshot: the names
    agree and the content does not)"""
    from run import git
    top = git(root, "rev-parse", "--show-toplevel").stdout.strip()
    have = tree_blobs(top, ref)
    rels = [os.path.relpath(os.path.join(dp, fn), top).replace(chr(92), "/") for f in folders for dp, _, fns in os.walk(f) if "__pycache__" not in dp for fn in fns]
    named = [r for r in rels if r in have]
    return [r for r in rels if r not in have] + [r for r, h in zip(named, blob_ids(top, named)) if h != have[r]]


def tag_for_delete(root, commit, tag, folders, note=""):
    """-> (tag, lost): a tag that holds the current content of every file of the folders (lost == []), else (None, the files that no snapshot holds). git_snapshot reuses an
    existing <prefix>/<name>-scratch and leaves the new snapshot commit on no ref ('tag existed'): that tag holds the earlier state, so a delete would lose what changed since.
    Then the new snapshot is tagged <tag>.attempt2 (.attempt3 ...) and compared again; nothing is tagged when the snapshot itself does not hold the files"""
    from run import git
    lost = not_held(root, tag, folders)
    if not lost:
        return tag, []
    lost = not_held(root, commit, folders)
    if lost:
        return None, lost
    top = git(root, "rev-parse", "--show-toplevel").stdout.strip()
    for n in range(2, 100):
        new = f"{tag}.attempt{n}"
        if git(top, "rev-parse", "-q", "--verify", f"refs/tags/{new}").returncode == 0:
            if not not_held(root, new, folders):                  # an earlier --delete that stopped after the tag: it holds this state
                return new, []
            continue
        r = git(top, "tag", "-a", new, "-m", (f"{note or tag}: the snapshot of the scratch as it was before the clean-up (the tag {tag} holds an earlier state)")[:200], commit)
        if r.returncode != 0:
            return None, [f"(the tag {new} could not be made: {r.stderr.strip()[:100]})"]
        lost = not_held(root, new, folders)
        return (new, []) if not lost else (None, lost)
    return None, [f"(no free name {tag}.attemptN)"]


def modules_in(folders):
    mods = {}
    for f in folders:
        for dp, _, fn in os.walk(f):
            for x in fn:
                if x.endswith(".py"):
                    mods.setdefault(x[:-3], os.path.join(dp, x))
    return mods


def tool_modules():
    """the modules of autoresearch/tools: live code puts that folder on sys.path (evaluation/bev_selftest.py imports card, decide, experiment from it), so a copy of
    one of them in a session's scratch is not what the import finds"""
    d = os.path.join(P.AR, "tools")
    return {x[:-3] for x in os.listdir(d) if x.endswith(".py")} if os.path.isdir(d) else set()


def live_imports(mods, root, live, also=()):
    pyfiles = [os.path.join(dp, x) for d in live for dp, dn, fn in os.walk(os.path.join(root, d)) for x in fn if x.endswith(".py") and "__pycache__" not in dp]
    other = {os.path.basename(p)[:-3] for p in pyfiles} | set(also)
    hits = {}
    for p in pyfiles:
        for m in set(IMP.findall(open(p, encoding="utf-8", errors="ignore").read())):
            if m in mods and m not in other:
                hits.setdefault(m, []).append(os.path.relpath(p, root))
    return hits


REF_CODE = (".py", ".sh", ".ps1", ".bat", ".cmd", ".json", ".yml", ".yaml", ".cfg", ".toml", ".ini", ".ipynb")      # files that run or configure
REF_DOC = (".md", ".txt")
REF_MAX_BYTES = 2_000_000
COMMENT = ("#", "//", '"_', "*", ";", "::", "REM ", "<!--", "'''", '"""')      # a line that explains and does not run (also the "_about" keys of the json configs)
KEY_COMMENT = re.compile(r"""^\s*\{?\s*"_\w*"\s*:""")          # {"_about": ...} of a json config, also on the first line of the file
TAIL = r"""(/[^\s'"`<>()\[\]{},;|*?]*)?"""


def live_dirs(cfg):
    """the folders whose files must not depend on the scratch: live_code plus the other code_dirs (bev_fusion: tests, which import its radar check and read its json),
    never the scratch itself ('.' and the scratch folder are skipped, so a project that lists its scratch among the code_dirs keeps working)"""
    norm = lambda x: posixpath.normpath(x.replace(chr(92), "/")).strip("/")
    s = norm(cfg["scratch"])
    extra = [d for d in cfg["code_dirs"] if norm(d) not in (".", "", s) and not norm(d).startswith(s + "/")]
    return list(dict.fromkeys(list(cfg["live_code"]) + extra))


def path_refs(folders, root, live, extra=()):
    """[(file, line number, line, blocking)] of the lines that spell a scratch folder, below the live folders and in the `extra` files (the project's own project.json /
    README.md). A shell launcher, a config, a test or a README that spells `temp/loop19/x.py` or `temp/train/run_ar.sh` breaks when the folder moves, and no python import
    shows it. Blocking is a reference that still has something to break: the file it names is still there (a mention of a script that was promoted and left is history),
    or the folder itself is named in a line of code (.sh .py .json ...) that is no comment; a bare folder in a README or in a comment is listed and does not block.
    Read too: both slash spellings (a run of backslashes is one separator: json escapes it), upper case, and the python spellings "temp", "loop19" and "temp" / "loop19".
    Not read: scratch folders (the one cleaned and its siblings), symlinks, names built at run time (f"temp/loop{n}", $SCRATCH/loop19), files over 2 MB"""
    rels = sorted({os.path.relpath(f, root).replace(chr(92), "/").rstrip("/") for f in folders})
    rx = re.compile(r"(?<![\w.-])(" + "|".join(re.escape(r) for r in rels) + r")" + TAIL + r"(?![\w*-]|\.\w)", re.I)
    canon = {r.lower(): r for r in rels}
    q = r"""["']"""
    joined = []
    for r in rels:
        parts = [re.escape(x) for x in r.split("/")]
        joined.append(re.compile(q + (q + r"\s*,\s*" + q).join(parts) + q, re.I))          # os.path.join(HERE, "temp", "loop19")
        joined.append(re.compile(q + (q + r"\s*/\s*" + q).join(parts) + q, re.I))          # Path(HERE) / "temp" / "loop19"
    skip = {os.path.abspath(f) for f in folders}
    files = {}
    for d in live:
        for dp, dn, fn in os.walk(os.path.join(root, d)):
            dn[:] = [x for x in dn if x != "__pycache__" and os.path.abspath(os.path.join(dp, x)) not in skip and not os.path.islink(os.path.join(dp, x))]
            for x in fn:
                files[os.path.join(dp, x)] = 1
    for p in extra:
        files[p] = 1
    out = []
    for p in sorted(files):
        ext = os.path.splitext(p)[1].lower()
        if ext not in REF_CODE + REF_DOC:
            continue
        try:
            if os.path.islink(p) or os.path.getsize(p) > REF_MAX_BYTES:
                continue
            with open(p, encoding="utf-8", errors="ignore") as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            m = rx.search(re.sub(r"\\+", "/", line))
            code = ext in REF_CODE and not (line.lstrip().startswith(COMMENT) or KEY_COMMENT.match(line))
            if m:
                inside = (m.group(2) or "").rstrip(".:").strip("/") != ""
                named = canon[m.group(1).lower()] + (m.group(2) or "").rstrip(".:")                     # the spelling of the file system, not of the line
                blocking = os.path.exists(os.path.join(root, named)) if inside else code
            elif any(j.search(line) for j in joined):
                blocking = code
            else:
                continue
            out.append((os.path.relpath(p, root).replace(chr(92), "/"), i, line.strip(), blocking))
    return out


def newest(folders):
    return max((os.path.getmtime(os.path.join(dp, x)) for f in folders for dp, _, fn in os.walk(f) for x in fn), default=0)


def run(name, move=False, delete=False, force=False, root=None, cfg=None, git=True, ignore_refs=False):
    root, cfg = root or P.root(), cfg or P.cfg()
    temp = os.path.join(root, cfg["scratch"])
    folders = find(name, temp)
    if not folders:
        print(f"nothing named {name} under {cfg['scratch']}/")
        return 0
    n = sum(len(fn) for f in folders for _, _, fn in os.walk(f))
    print(f"{name}: {n} files in " + ", ".join(os.path.relpath(f, root) for f in folders))
    live = live_dirs(cfg)
    hits = live_imports(modules_in(folders), root, live, tool_modules())
    for m, users in sorted(hits.items()):
        print(f"  BLOCKED: module {m} is imported by {users[:3]} - copy it to a code folder first")
    own = [os.path.join(root, f) for f in ("project.json", "README.md")] + [os.path.join(os.path.dirname(os.path.join(root, cfg["ledger"])), "README.md")]
    refs = path_refs(folders, root, live, [p for p in dict.fromkeys(own) if os.path.isfile(p)])
    block, info = [r for r in refs if r[3]], [r for r in refs if not r[3]]
    for f, ln, text, _ in block[:8]:
        print(f"  {'IGNORED' if ignore_refs else 'BLOCKED'}: {f}:{ln} {text[:110]}")
    if len(block) > 8:
        print(f"  ... and {len(block) - 8} more line(s) that name it")
    if block and not ignore_refs:
        print("  change the reference to the new place (a script that moved to a code folder), or --ignore-refs when the mention is history")
    for f, ln, text, _ in info[:3]:
        print(f"  mentioned, not blocking (a script that already left, a bare folder in a document or a comment): {f}:{ln} {text[:90]}")
    age_h = (time.time() - newest(folders)) / 3600
    if age_h < 6 and not force:
        print(f"  BLOCKED: newest file is {age_h:.1f} h old (a job may use it); --force overrides")
        return 1
    if hits or (block and not ignore_refs):
        return 1
    if not (move or delete):
        print("dry run: nothing changed (--move or --delete)")
        return 0
    if delete and git and not kept_by_snapshot(cfg):   # decided BEFORE the snapshot: a refusal must leave no tag behind
        print(f"  REFUSED: --delete needs '{cfg['scratch']}' in project.json code_dirs (the snapshot commits code_dirs only); use --move")
        return 1
    held = None
    if git:
        from run import git_snapshot
        commit, tag, note = git_snapshot(root, cfg, cfg["tag_prefix"], f"{name}-scratch", f"scratch of {name} before clean-up")
        print(f"  code kept in git: {commit} {tag} {note}")
        if delete and not tag:                     # no repository, or a git step failed: nothing holds the scripts
            print(f"  REFUSED: --delete needs the scratch in git, but the snapshot gave no tag ({commit} {note}); use --move or fix git")
            return 1
        if delete:                                 # an old tag of an earlier snapshot ("tag existed") holds the earlier content: compare every file by blob id, tag the new snapshot if need be
            held, lost = tag_for_delete(root, commit, tag, folders, f"scratch of {name}")
            if lost:
                print(f"  REFUSED: no tag holds {len(lost)} file(s) as they are now, e.g. {lost[:3]} (only scripts are tagged); use --move (or a new scratch name)")
                return 1
            if held != tag:
                print(f"  tag {tag} holds an earlier state of the scratch: the new snapshot {commit} is tagged {held}")
    if delete and not held:                        # git=False (a test setting) or no check: nothing proves the files are kept
        print("  REFUSED: --delete needs a tag that holds the files; use --move")
        return 1
    for f in folders:
        if delete:
            lost = not_held(root, held, [f])       # again right before the delete: a file written since the snapshot is not in the tag
            if lost:
                print(f"  REFUSED: {len(lost)} file(s) changed since the snapshot, e.g. {lost[:3]}; nothing of {os.path.relpath(f, root)} was deleted; run it again or use --move")
                return 1
            shutil.rmtree(f)
            print("deleted", os.path.relpath(f, root))
        else:
            dst = os.path.join(root, cfg["trash"], f"temp_{name}", os.path.relpath(f, temp))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(f, dst)
            print("moved", os.path.relpath(f, root), "->", os.path.relpath(dst, root), f"(add a line to {cfg['trash']}/README.md)")
    return 0


def selftest():
    import tempfile
    root = tempfile.mkdtemp()
    cfg = dict(P.DEFAULTS, live_code=["training"], scratch="temp", trash="_to_delete")
    for d in ("training", "temp/loop99", "_to_delete"):
        os.makedirs(os.path.join(root, d))
    open(os.path.join(root, "temp/loop99/probe.py"), "w").write("x = 1\n")
    open(os.path.join(root, "training/user.py"), "w").write("import probe\n")
    old = time.time() - 48 * 3600
    os.utime(os.path.join(root, "temp/loop99/probe.py"), (old, old))
    assert run("loop99", move=True, root=root, cfg=cfg, git=False) == 1, "must refuse while live code imports it"
    os.remove(os.path.join(root, "training/user.py"))
    assert run("loop99", root=root, cfg=cfg, git=False) == 0 and os.path.exists(os.path.join(root, "temp/loop99"))
    assert run("loop99", move=True, root=root, cfg=cfg, git=False) == 0
    assert os.path.exists(os.path.join(root, "_to_delete/temp_loop99/loop99/probe.py"))
    # --delete refuses when nothing keeps the scripts: no repository here (git_snapshot gives no tag); --move still works
    os.makedirs(os.path.join(root, "temp/loop98"))
    open(os.path.join(root, "temp/loop98/a.py"), "w").write("x = 2\n")
    os.utime(os.path.join(root, "temp/loop98/a.py"), (old, old))
    cfg98 = dict(cfg, code_dirs=["temp"])
    assert run("loop98", delete=True, root=root, cfg=cfg98, git=True) == 1 and os.path.exists(os.path.join(root, "temp/loop98/a.py")), \
        "--delete without a tag must refuse and keep the scratch"
    assert run("loop98", move=True, root=root, cfg=cfg98, git=True) == 0 and not os.path.exists(os.path.join(root, "temp/loop98")), "--move needs no tag"
    # path references: what blocks (the named file is still there, a folder in a line of code, any spelling of it) and what does not (history, documents, comments, other names)
    ref = tempfile.mkdtemp()
    for d in ("training", "tests", "temp/loop94", "temp/loop941", "_to_delete"):
        os.makedirs(os.path.join(ref, d))
    for fn in ("temp/loop94/go.sh", "temp/loop94/helper.py", "temp/loop94/card.py", "temp/loop941/x.py"):
        open(os.path.join(ref, fn), "w").write("x = 1\n")
        os.utime(os.path.join(ref, fn), (old, old))
    cfgp = dict(P.DEFAULTS, live_code=["training"], code_dirs=["training", "tests", "temp"], scratch="temp", trash="_to_delete")
    assert live_dirs(cfgp) == ["training", "tests"] and live_dirs(dict(cfgp, code_dirs=[".", "./temp", "../autogt", "temp/sub"])) == ["training", "../autogt"], live_dirs(cfgp)

    imp = tempfile.mkdtemp()                                                    # a module that live code finds elsewhere (autoresearch/tools on sys.path) is no dependency on a scratch copy
    os.makedirs(os.path.join(imp, "training"))
    open(os.path.join(imp, "training/u.py"), "w").write("import card\nimport helper\n")
    assert set(live_imports({"card": "x", "helper": "y"}, imp, ["training"])) == {"card", "helper"}
    assert set(live_imports({"card": "x", "helper": "y"}, imp, ["training"], {"card"})) == {"helper"}
    assert {"card", "decide", "run"} <= tool_modules(), "the tools of autoresearch are found"
    shutil.rmtree(imp)

    def case(where, text, blocks, **kw):
        path = os.path.join(ref, where)
        with open(path, "w") as f:
            f.write(text)
        try:
            got = run("loop94", root=ref, cfg=kw.pop("cfg", cfgp), git=False, **kw)
            assert got == (1 if blocks else 0), (where, text, got)
            assert os.path.exists(os.path.join(ref, "temp/loop94/go.sh")), "a dry run moves nothing"
        finally:
            os.remove(path)
    for where, text in (("training/launch.sh", "bash temp/loop94/go.sh --fast\n"), ("training/cfg.json", '{"cmd": "temp\\\\loop94\\\\go.sh"}\n'),
                        ("training/run.cmd", "call TEMP\\LOOP94\\go.sh\n"),                                               # Windows spelling, upper case
                        ("tests/README.md", "start it with `./temp/loop94/go.sh`.\n"),                                      # a document naming a file that is still there
                        ("tests/test_a.py", "import helper\n"), ("tests/test_b.py", 'p = os.path.join(HERE, "..", "temp", "loop94", "go.sh")\n'),
                        ("tests/test_c.py", 'p = HERE / "temp" / "loop94" / "go.sh"\n'), ("training/b.sh", "ls temp/loop94\n"),
                        ("project.json", '{"fingerprint_extra": ["temp/loop94/go.sh"]}\n'), ("README.md", "run temp/loop94/go.sh\n")):
        case(where, text, True)
    for ext in (".py", ".ps1", ".yml", ".yaml", ".cfg", ".txt", ".bat", ".toml", ".ini"):                           # every extension that is read
        case("tests/x" + ext, "cmd = temp/loop94/go.sh\n", True)
    for where, text in (("training/launch.sh", "bash temp/loop941/x.py\n"), ("training/a.sh", "cd mytemp/loop94 && ./x-temp/loop94/go.sh\n"),          # other names with the same letters
                        ("training/b.sh", "tail temp/loop94.log\n"),
                        ("tests/README.md", "moved from temp/loop94/old.py (promoted)\n"),                               # a script that left: history
                        ("tests/NOTES.md", "the scratch of the loop lived in temp/loop94.\n"),                           # a bare folder in a document
                        ("training/c.py", "# see temp/loop94\n"), ("training/d.json", '{"_about": "made in temp/loop94"}\n')):     # a comment
        case(where, text, False)
    case("tests/test_card.py", "import card\n", False)                                                                 # card is a tool of autoresearch on sys.path, not the scratch copy
    case("training/big.txt", "temp/loop94/go.sh\n" + "x" * (REF_MAX_BYTES + 10), False)                               # over the size limit: not read
    open(os.path.join(ref, "temp/loop94/self.sh"), "w").write("bash temp/loop94/go.sh\n")                           # the scratch folder names itself: not scanned, also when it lies in a live folder
    os.utime(os.path.join(ref, "temp/loop94/self.sh"), (old, old))
    case("training/ok.sh", "echo\n", False, cfg=dict(cfgp, live_code=["training", "temp"]))
    os.remove(os.path.join(ref, "temp/loop94/self.sh"))
    # --ignore-refs lets a mention go and nothing else: not an import, not the age guard; --force overrides the age guard only
    case("tests/README.md", "start it with temp/loop94/go.sh\n", True, force=True)
    case("tests/README.md", "start it with temp/loop94/go.sh\n", False, ignore_refs=True)
    case("tests/test_a.py", "import helper\n", True, ignore_refs=True, force=True)
    os.utime(os.path.join(ref, "temp/loop94/go.sh"))                                                                  # a file of today: a job may use it
    case("training/ok.sh", "echo\n", True)
    case("training/ok.sh", "echo\n", False, force=True)
    os.utime(os.path.join(ref, "temp/loop94/go.sh"), (old, old))
    open(os.path.join(ref, "tests/README.md"), "w").write("history: temp/loop94/go.sh\n")
    assert run("loop94", move=True, root=ref, cfg=cfgp, git=False) == 1
    assert run("loop94", move=True, ignore_refs=True, root=ref, cfg=cfgp, git=False) == 0 and os.path.exists(os.path.join(ref, "_to_delete/temp_loop94/loop94/go.sh"))
    shutil.rmtree(ref)
    # real throw-away repository
    import subprocess
    repo = tempfile.mkdtemp()

    def g(*args):
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout
    g("init", "-q")
    g("config", "user.name", "selftest")
    g("config", "user.email", "selftest@example.invalid")
    os.makedirs(os.path.join(repo, "code"))
    open(os.path.join(repo, "code/a.py"), "w").write("x = 1\n")
    g("add", "-A")
    g("commit", "-qm", "base")
    cfgr = dict(P.DEFAULTS, live_code=[], scratch="temp", trash="_to_delete", tag_prefix="t", code_dirs=["code"])
    for n in ("loop97", "loop96"):
        os.makedirs(os.path.join(repo, "temp", n))
        open(os.path.join(repo, "temp", n, "s.py"), "w").write("y = 1\n")
        os.utime(os.path.join(repo, "temp", n, "s.py"), (old, old))
    open(os.path.join(repo, "temp", "loop97", "my script.py"), "w").write("z = 1\n")        # a name with a space is one name for the tag check
    os.utime(os.path.join(repo, "temp", "loop97", "my script.py"), (old, old))
    # the scratch is not in code_dirs: refused before the snapshot, so no tag is left behind
    assert run("loop97", delete=True, root=repo, cfg=cfgr, git=True) == 1 and os.path.exists(os.path.join(repo, "temp/loop97/s.py"))
    assert g("tag").strip() == "", "a refused --delete must not create a tag"
    # an old tag without today's scripts (a resumed snapshot): the tag is not reused, the new snapshot gets <tag>.attempt2 and holds the script; only then it goes
    g("tag", "-a", "t/loop96-scratch", "-m", "old attempt")
    cfgr2 = dict(cfgr, code_dirs=["code", "./temp"])
    assert run("loop96", delete=True, root=repo, cfg=cfgr2, git=True) == 0 and not os.path.exists(os.path.join(repo, "temp/loop96"))
    assert g("show", "t/loop96-scratch.attempt2:temp/loop96/s.py") == "y = 1\n" and "temp/loop96/s.py" not in g("ls-tree", "-r", "--name-only", "t/loop96-scratch").split(), "the old tag stays as it was"
    # scratch in code_dirs (a "./temp" spelling too), fresh tag: the snapshot holds the script and --delete goes through
    assert run("loop97", delete=True, root=repo, cfg=cfgr2, git=True) == 0 and not os.path.exists(os.path.join(repo, "temp/loop97"))
    assert "temp/loop97/s.py" in g("ls-tree", "-r", "--name-only", "t/loop97-scratch").split(), "the tag must hold the deleted script"

    def scratch(n, text, **files):
        os.makedirs(os.path.join(repo, "temp", n), exist_ok=True)
        for fn, body in dict({"s.py": text}, **files).items():
            open(os.path.join(repo, "temp", n, fn), "w").write(body)
            os.utime(os.path.join(repo, "temp", n, fn), (old, old))
    # the finding of the deep review: <prefix>/loop93-scratch exists from an earlier --move (old content); the same file names come back with NEW content. Names agree, content does not:
    # the old tag must not pass for the new files (the snapshot commit would hang on no ref and the folder be deleted)
    scratch("loop93", "y = 1\n")
    assert run("loop93", move=True, root=repo, cfg=cfgr2, git=True) == 0
    shutil.move(os.path.join(repo, "_to_delete/temp_loop93/loop93"), os.path.join(repo, "temp/loop93"))
    scratch("loop93", "y = 2\n")
    assert run("loop93", delete=True, root=repo, cfg=cfgr2, git=True) == 0 and not os.path.exists(os.path.join(repo, "temp/loop93"))
    assert g("show", "t/loop93-scratch:temp/loop93/s.py") == "y = 1\n", "the old tag is untouched"
    assert g("show", "t/loop93-scratch.attempt2:temp/loop93/s.py") == "y = 2\n", "the new content is held by a tag of its own"
    scratch("loop93", "y = 3\n")                                               # and once more: attempt2 holds y = 2, so attempt3 is made
    assert run("loop93", delete=True, root=repo, cfg=cfgr2, git=True) == 0
    assert g("show", "t/loop93-scratch.attempt3:temp/loop93/s.py") == "y = 3\n" and g("show", "t/loop93-scratch.attempt2:temp/loop93/s.py") == "y = 2\n"
    scratch("loop93", "y = 3\n")                                               # the same state again: attempt3 already holds it, no attempt4
    assert run("loop93", delete=True, root=repo, cfg=cfgr2, git=True) == 0 and "t/loop93-scratch.attempt4" not in g("tag").split()
    # a tag that cannot be made: refused, the scripts stay
    g("tag", "-a", "t/loop91-scratch", "-m", "old attempt")
    g("tag", "-a", "t/loop91-scratch.attempt2/blocker", "-m", "a ref that blocks the name attempt2")
    scratch("loop91", "y = 1\n")
    assert run("loop91", delete=True, root=repo, cfg=cfgr2, git=True) == 1 and os.path.exists(os.path.join(repo, "temp/loop91/s.py")), "no tag could be made: refuse"
    # a file written after the snapshot is in no tag: the check right before the delete refuses it
    scratch("loop90", "y = 1\n")
    real_tag_for_delete = tag_for_delete

    def writes_after_snapshot(*a, **k):
        res = real_tag_for_delete(*a, **k)
        open(os.path.join(repo, "temp/loop90/s.py"), "w").write("y = 99\n")      # a job writes while the clean-up runs
        os.utime(os.path.join(repo, "temp/loop90/s.py"), (old, old))
        return res
    globals()["tag_for_delete"] = writes_after_snapshot
    try:
        assert run("loop90", delete=True, root=repo, cfg=cfgr2, git=True) == 1 and open(os.path.join(repo, "temp/loop90/s.py")).read() == "y = 99\n"
    finally:
        globals()["tag_for_delete"] = real_tag_for_delete
    assert run("loop90", delete=True, root=repo, cfg=cfgr2, git=False) == 1 and os.path.exists(os.path.join(repo, "temp/loop90/s.py")), "git=False proves nothing: --delete refuses"
    # a result file next to the script is held by no tag: --delete refuses (hard rule 5), --move keeps it
    open(os.path.join(repo, ".gitignore"), "w").write("temp/**\n")                       # the repository's rule: the snapshot then adds the scripts only
    os.makedirs(os.path.join(repo, "temp", "loop95"))
    for fn in ("s.py", "result.json"):
        open(os.path.join(repo, "temp", "loop95", fn), "w").write("1\n")
        os.utime(os.path.join(repo, "temp", "loop95", fn), (old, old))
    assert run("loop95", delete=True, root=repo, cfg=cfgr2, git=True) == 1 and os.path.exists(os.path.join(repo, "temp/loop95/result.json")), \
        "--delete must refuse a folder with a file that no tag holds"
    assert run("loop95", move=True, root=repo, cfg=cfgr2, git=True) == 0 and os.path.exists(os.path.join(repo, "_to_delete/temp_loop95/loop95/result.json"))
    shutil.rmtree(repo)
    shutil.rmtree(root)
    print("clean_temp selftest passed")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a:
        selftest()
    elif not a or a[0].startswith("-"):
        print(__doc__)
    else:
        sys.exit(run(a[0], "--move" in a, "--delete" in a, "--force" in a, ignore_refs="--ignore-refs" in a))
