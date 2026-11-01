"""D22 (a), second round: the findings of the ar-reviewer pass over the first round."""
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
T = f"{ROOT}/autoresearch/tools"


def read(p):
    return open(p, encoding="utf-8").read()


def write(p, s):
    open(p, "w", encoding="utf-8", newline="").write(s)
    print("edited", os.path.relpath(p, ROOT))


def edit(path, old, new, count=1):
    s = read(path)
    assert s.count(old) == count, (path, old[:70], s.count(old))
    write(path, s.replace(old, new))


# ---------------------------------------------------------------- clean_temp.py
p = f"{T}/clean_temp.py"
s = read(p)
a, b = s.index("def kept_by_snapshot(cfg):"), s.index("def modules_in(folders):")
s = s[:a] + '''def kept_by_snapshot(cfg):
    """is the scratch folder inside one of the code_dirs that git_snapshot commits? (otherwise the tag does not hold the scripts)"""
    norm = lambda x: posixpath.normpath(x.replace(chr(92), "/")).strip("/")
    s = norm(cfg["scratch"])
    return any(norm(d) in (".", "") or s == norm(d) or s.startswith(norm(d) + "/") for d in cfg["code_dirs"])


def missing_from_tag(root, tag, folders):
    """the scripts (.py / .sh) of the scratch folders that the tag does NOT contain (an existing tag of an earlier snapshot, a .gitignore rule ...)"""
    from run import git
    top = git(root, "rev-parse", "--show-toplevel").stdout.strip()
    have = set(git(top, "ls-tree", "-r", "--name-only", tag).stdout.split())
    out = []
    for f in folders:
        for dp, _, fns in os.walk(f):
            for fn in fns:
                if fn.endswith((".py", ".sh")):
                    rel = os.path.relpath(os.path.join(dp, fn), top).replace(chr(92), "/")
                    if rel not in have:
                        out.append(rel)
    return out


''' + s[b:]
s = s.replace("import os\nimport re\nimport shutil\n", "import os\nimport posixpath\nimport re\nimport shutil\n", 1)
old = s[s.index("    if git:\n        from run import git_snapshot"):s.index("    for f in folders:\n        if delete:")]
new = '''    if delete and git and not kept_by_snapshot(cfg):   # decided BEFORE the snapshot: a refusal must leave no tag behind
        print(f"  REFUSED: --delete needs '{cfg['scratch']}' in project.json code_dirs (the snapshot commits code_dirs only); use --move")
        return 1
    if git:
        from run import git_snapshot
        commit, tag, note = git_snapshot(root, cfg, cfg["tag_prefix"], f"{name}-scratch", f"scratch of {name} before clean-up")
        print(f"  code kept in git: {commit} {tag} {note}")
        if delete and not tag:                     # no repository, or a git step failed: nothing holds the scripts
            print(f"  REFUSED: --delete needs the scratch in git, but the snapshot gave no tag ({commit} {note}); use --move or fix git")
            return 1
        if delete:                                 # an old tag of an earlier snapshot ("tag existed") may not hold today's scripts
            lost = missing_from_tag(root, tag, folders)
            if lost:
                print(f"  REFUSED: tag {tag} does not contain {len(lost)} script(s), e.g. {lost[:3]}; use --move (or a new scratch name)")
                return 1
'''
s = s.replace(old, new)
# selftest: real throw-away repositories instead of a stand-in for git_snapshot
a, b = s.index("    # --delete refuses when nothing keeps the scripts"), s.index("    shutil.rmtree(root)\n    print(\"clean_temp selftest passed\")")
s = s[:a] + '''    # --delete refuses when nothing keeps the scripts: no repository here (git_snapshot gives no tag); --move still works
    os.makedirs(os.path.join(root, "temp/loop98"))
    open(os.path.join(root, "temp/loop98/a.py"), "w").write("x = 2\\n")
    os.utime(os.path.join(root, "temp/loop98/a.py"), (old, old))
    cfg98 = dict(cfg, code_dirs=["temp"])
    assert run("loop98", delete=True, root=root, cfg=cfg98, git=True) == 1 and os.path.exists(os.path.join(root, "temp/loop98/a.py")), \\
        "--delete without a tag must refuse and keep the scratch"
    assert run("loop98", move=True, root=root, cfg=cfg98, git=True) == 0 and not os.path.exists(os.path.join(root, "temp/loop98")), "--move needs no tag"
    # real throw-away repository
    import subprocess
    repo = tempfile.mkdtemp()

    def g(*args):
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout
    g("init", "-q")
    g("config", "user.name", "selftest")
    g("config", "user.email", "selftest@example.invalid")
    os.makedirs(os.path.join(repo, "code"))
    open(os.path.join(repo, "code/a.py"), "w").write("x = 1\\n")
    g("add", "-A")
    g("commit", "-qm", "base")
    cfgr = dict(P.DEFAULTS, live_code=[], scratch="temp", trash="_to_delete", tag_prefix="t", code_dirs=["code"])
    for n in ("loop97", "loop96"):
        os.makedirs(os.path.join(repo, "temp", n))
        open(os.path.join(repo, "temp", n, "s.py"), "w").write("y = 1\\n")
        os.utime(os.path.join(repo, "temp", n, "s.py"), (old, old))
    # the scratch is not in code_dirs: refused before the snapshot, so no tag is left behind
    assert run("loop97", delete=True, root=repo, cfg=cfgr, git=True) == 1 and os.path.exists(os.path.join(repo, "temp/loop97/s.py"))
    assert g("tag").strip() == "", "a refused --delete must not create a tag"
    # an old tag without today's scripts (a resumed snapshot) is refused too; the scripts stay
    g("tag", "-a", "t/loop96-scratch", "-m", "old attempt")
    cfgr2 = dict(cfgr, code_dirs=["code", "./temp"])
    assert run("loop96", delete=True, root=repo, cfg=cfgr2, git=True) == 1 and os.path.exists(os.path.join(repo, "temp/loop96/s.py"))
    # scratch in code_dirs (a "./temp" spelling too), fresh tag: the snapshot holds the script and --delete goes through
    assert run("loop97", delete=True, root=repo, cfg=cfgr2, git=True) == 0 and not os.path.exists(os.path.join(repo, "temp/loop97"))
    assert "temp/loop97/s.py" in g("ls-tree", "-r", "--name-only", "t/loop97-scratch").split(), "the tag must hold the deleted script"
    shutil.rmtree(repo)
''' + s[b:]
write(p, s)

# ---------------------------------------------------------------- run.py
p = f"{T}/run.py"
edit(p, '''def record(d):
    try:
        return json.load(open(os.path.join(d, "run.json")))
    except Exception:
        return {}
''', '''def record(d):
    """the run record of a directory, {} when there is none. A file that exists but does not parse, or a write in progress (a tmp file next to it),
    is retried a few times: on /mnt/c (9p) an atomic replace is not atomic for a concurrent reader (about 2 % torn / missing reads measured)"""
    path = os.path.join(d, "run.json")
    for i in range(4):
        try:
            return json.load(open(path))
        except Exception:
            try:
                writing = os.path.exists(path) or any(n.startswith("run.json.tmp.") for n in os.listdir(d))
            except OSError:
                writing = False
            if not writing or i == 3:
                return {}
            time.sleep(0.05)
''')
edit(p, '''    P.atomic_write(path, json.dumps(rec, indent=indent))
''', '''    text = json.dumps(rec, indent=indent)
    for i in range(3):
        try:
            return P.atomic_write(path, text)
        except PermissionError:      # native Windows: os.replace over a file that a reader has open
            time.sleep(0.1)
    P.atomic_write(path, text)
''')
edit(p, '''        o = record(s)
        owner = o.get("run_dir", "")''', '''        o = record(s)
        if not o:                    # owner file there but unreadable right now: a taker that is writing it, not a dead one
            try:
                return time.time() - os.path.getmtime(s) > 10
            except OSError:
                return False
        owner = o.get("run_dir", "")''')
edit(p, '''notes=f"GPU memory {u} MB in use or all {a.gpu_slots} slots taken for {a.wait_gpu} min")''',
     '''notes=("nvidia-smi unreadable" if (a.gpu_max_mb and u is None) else f"GPU memory {u} MB in use") + f" or all {a.gpu_slots} slots taken for {a.wait_gpu} min")''')
edit(p, '''    rec.update(status=status, exit=rc, duration_s=dur, attempts=attempt, ended=now(), last_lines=tail.strip().splitlines()[-12:])
    save_rec(os.path.join(d, "run.json"), rec)
''', '''    rec.update(status=status, exit=rc, duration_s=dur, attempts=attempt, ended=now(), last_lines=tail.strip().splitlines()[-12:])
    try:
        save_rec(os.path.join(d, "run.json"), rec)
    except OSError as e:             # the command has finished: the done flag and the ledger row matter more than this file
        print(f"[{a.id}] warning: run record not written ({e!r})", flush=True)
''')

# ---------------------------------------------------------------- project.py: children must see the folder that was resolved via the cwd
edit(f"{T}/project.py", '''        if os.path.isfile(os.path.join(alt, "project.json")):
            r = alt
''', '''        if os.path.isfile(os.path.join(alt, "project.json")):
            r = alt
            os.environ["AR_PROJECT"] = r                        # a child started from another folder must find the same problem
''')

# ---------------------------------------------------------------- selftest.py
edit(f"{T}/selftest.py", '''    def writer():
        i = 0
        while not stop.is_set():
            RUN.save_rec(os.path.join(rd, "run.json"), dict(id="S.atomic", pid=1, notes="x" * 20000, i=i))
            i += 1
''', '''    n_writes = [0]

    def writer():
        while not stop.is_set():
            RUN.save_rec(os.path.join(rd, "run.json"), dict(id="S.atomic", pid=1, notes="x" * 20000, i=n_writes[0]))
            n_writes[0] += 1
''')
edit(f"{T}/selftest.py", '''    assert torn == 0, f"{torn} torn reads of a run record"''',
     '''    assert torn == 0 and n_writes[0] > 20, f"{torn} torn reads of a run record, {n_writes[0]} writes (a dead writer proves nothing)"''')
edit(f"{T}/selftest.py", '''    sch_before = open(f"{T}/schedule.csv", "rb").read()
    r5 = subprocess.run([sys.executable, f"{HERE}/schedule.py", "add", "S.race", "--kind", "train"], capture_output=True, text=True, env=env)''',
     '''    SC.save([dict(item="S.dup1", kind="train", state="planned", resource="cpu", priority="1", command="true")], f"{T}/schedule.csv")
    sch_before = open(f"{T}/schedule.csv", "rb").read()
    r5 = subprocess.run([sys.executable, f"{HERE}/schedule.py", "add", "S.dup1", "--kind", "train"], capture_output=True, text=True, env=env)''')
