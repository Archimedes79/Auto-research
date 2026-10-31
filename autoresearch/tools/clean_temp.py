"""Clean up the scratch of a finished loop (phase P4). Standard library only.

    python autoresearch/tools/clean_temp.py loop19             # dry run: what would go, what live code still imports from it
    python autoresearch/tools/clean_temp.py loop19 --move      # commit + tag the scratch (<prefix>/loop19-scratch), then move it to <trash>/temp_loop19/
    python autoresearch/tools/clean_temp.py loop19 --delete    # commit + tag, then delete (the tag keeps the scripts)
    python autoresearch/tools/clean_temp.py --selftest

Convention: scratch of a loop lives in <scratch>/loopNN/ (also found: <scratch>/analysis/loopNN, <scratch>/launchers/loopNN).
Refuses when the live code folders (project.json live_code) import a python module that exists only in the scratch (promote it first),
or when a file changed in the last 6 hours (a job may still use it; --force overrides).
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


def modules_in(folders):
    mods = {}
    for f in folders:
        for dp, _, fn in os.walk(f):
            for x in fn:
                if x.endswith(".py"):
                    mods.setdefault(x[:-3], os.path.join(dp, x))
    return mods


def live_imports(mods, root, live):
    pyfiles = [os.path.join(dp, x) for d in live for dp, dn, fn in os.walk(os.path.join(root, d)) for x in fn if x.endswith(".py") and "__pycache__" not in dp]
    other = {os.path.basename(p)[:-3] for p in pyfiles}
    hits = {}
    for p in pyfiles:
        for m in set(IMP.findall(open(p, encoding="utf-8", errors="ignore").read())):
            if m in mods and m not in other:
                hits.setdefault(m, []).append(os.path.relpath(p, root))
    return hits


def newest(folders):
    return max((os.path.getmtime(os.path.join(dp, x)) for f in folders for dp, _, fn in os.walk(f) for x in fn), default=0)


def run(name, move=False, delete=False, force=False, root=None, cfg=None, git=True):
    root, cfg = root or P.root(), cfg or P.cfg()
    temp = os.path.join(root, cfg["scratch"])
    folders = find(name, temp)
    if not folders:
        print(f"nothing named {name} under {cfg['scratch']}/")
        return 0
    n = sum(len(fn) for f in folders for _, _, fn in os.walk(f))
    print(f"{name}: {n} files in " + ", ".join(os.path.relpath(f, root) for f in folders))
    hits = live_imports(modules_in(folders), root, cfg["live_code"])
    for m, users in sorted(hits.items()):
        print(f"  BLOCKED: module {m} is imported by {users[:3]} - copy it to a code folder first")
    age_h = (time.time() - newest(folders)) / 3600
    if age_h < 6 and not force:
        print(f"  BLOCKED: newest file is {age_h:.1f} h old (a job may use it); --force overrides")
        return 1
    if hits:
        return 1
    if not (move or delete):
        print("dry run: nothing changed (--move or --delete)")
        return 0
    if delete and git and not kept_by_snapshot(cfg):   # decided BEFORE the snapshot: a refusal must leave no tag behind
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
    for f in folders:
        if delete:
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
    shutil.rmtree(root)
    print("clean_temp selftest passed")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a:
        selftest()
    elif not a or a[0].startswith("-"):
        print(__doc__)
    else:
        sys.exit(run(a[0], "--move" in a, "--delete" in a, "--force" in a))
