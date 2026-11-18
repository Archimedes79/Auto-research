"""Read-only git calls through the Windows git.exe when the repository lies on a Windows drive and this runs in WSL. Standard library only.

    import gitfast
    gitfast.git(top, "status", "--porcelain", check=True, inp=None)   # stdout text; SystemExit with git's message on a failure when check
    gitfast.run(top, *args, inp=None, timeout=None, env=None)         # the CompletedProcess (bytes); the caller decides what a failure means
    gitfast.exe_prefix(top)                                           # [git.exe, -C, C:/..., -c core.autocrlf=false, -c core.quotepath=off] or None
    gitfast.windows_git(top)                                          # [git.exe, -C, <windows path>] for any repository (credential helper), or None
    python3 gitfast.py --selftest

Why: from WSL, git on /mnt/<drive>/ reads every file through the 9p mount; measured 2026-11-18 on this repository: status 41-232 s, ls-files --others 22 s,
diff 20 s, grep 5.4 s, tag 1.4-7.6 s, against 0.2-1 s for git.exe on the same repository. check_git.py, provenance.py and check_structure.py
call this module, so the rule lives in one place.

The rule: a call goes through git.exe when ALL hold
  - the repository path is /mnt/<drive letter>/... and git.exe is on PATH, and AR_GIT is not `wsl` (AR_GIT=wsl forces the WSL git: tests, debugging);
  - the command only reads (READ_ONLY, and `tag` / `remote` in their listing forms): every other command (add, commit, update-ref, tag creation, read-tree,
    worktree, push ...) runs in the WSL git, as before; so does a rev-parse that answers with a path (--show-toplevel, --git-dir ...: git.exe would
    print C:/... instead of /mnt/c/...);
  - the call names no repository through the environment (GIT_DIR, GIT_INDEX_FILE, GIT_WORK_TREE ...: git.exe does not see WSL variables or WSL paths) and `.git`
    of the folder is not a file (a linked worktree holds a WSL path that git.exe cannot follow);
  - every path after `--` can be written relative to the repository (-C makes both gits read it the same way); any other absolute path stays with the WSL git.
git.exe runs with `-c core.autocrlf=false` (Git for Windows defaults to true: a file's reported state must not change) and `-c core.quotepath=off`; the output
paths are repo-relative with forward slashes, the same strings as in the WSL git.
Fallback: when git.exe cannot run (OSError) or fails for a reason about the environment (stderr says "not a git repository", "dubious ownership" ...), the call is
retried once with the WSL git and git.exe is not used for that folder again in this process; any other non-zero exit is git's own answer and is returned.
"""
import functools
import os
import re
import shutil
import subprocess
import sys
import tempfile

READ_ONLY = {"status", "ls-files", "diff", "diff-tree", "diff-index", "rev-list", "rev-parse", "log", "show", "cat-file", "grep", "ls-remote", "for-each-ref", "ls-tree",
             "merge-base"}
LISTING = {"tag": ("-l", "--list", "--points-at", "--contains", "--no-contains", "--merged", "--no-merged"), "remote": ("-v", "--verbose")}   # the list forms of commands that also write
GLOBAL_OPTIONS = ("--no-optional-locks", "--no-pager")                  # the global options a caller may put before the command (and `-c key=value`)
PATH_ANSWERS = ("--show-", "--git-", "--absolute-git-dir", "--resolve-git-dir", "--path-format", "--local-env-vars")      # rev-parse options that answer with a path
ENV_ERROR = re.compile(rb"not a git repository|dubious ownership|safe\.directory|cannot change to|cannot chdir|unable to access|Invalid argument", re.I)
HARMLESS_ENV = ("GIT_AUTHOR_", "GIT_COMMITTER_", "GIT_TERMINAL_PROMPT")
_BROKEN = set()                                                         # folders where git.exe failed for an environment reason


@functools.lru_cache(maxsize=None)
def _exe():
    return shutil.which("git.exe")


def drive_path(top):
    """/mnt/c/dir -> C:/dir (git.exe takes forward slashes); None for a path that is not on a Windows drive"""
    m = re.fullmatch(r"/mnt/([A-Za-z])(/.*)?", os.path.abspath(top))
    return f"{m.group(1).upper()}:{m.group(2) or '/'}" if m else None


def exe_prefix(top):
    """the command prefix that runs a read-only git call for the repository in `top` through git.exe, or None where the WSL git is the right one"""
    exe, w = _exe(), drive_path(top)
    if os.environ.get("AR_GIT", "").lower() == "wsl" or not exe or not w or top in _BROKEN or os.path.isfile(os.path.join(top, ".git")):
        return None
    return [exe, "-C", w, "-c", "core.autocrlf=false", "-c", "core.quotepath=off"]


def windows_git(top):
    """the Windows git (git.exe, reachable from WSL, with the Windows credential helper that the WSL git lacks) as a command prefix with the repository path in
    Windows form, or None where there is none"""
    exe = _exe()
    if not exe:
        return None
    try:
        w = subprocess.run(["wslpath", "-w", top], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return [exe, "-C", w] if w else None


def _command(args):
    """(index of the command, True when the global options before it are all known)"""
    i = 0
    while i < len(args) and args[i].startswith("-"):
        if args[i] == "-c":
            i += 2
        elif args[i] in GLOBAL_OPTIONS:
            i += 1
        else:
            return i, False
    return i, True


def reads_only(args):
    """True when the call is one of the read-only commands (a listing form for `tag` and `remote`) and answers with no path in Windows form"""
    i, known = _command(args)
    if not known or i >= len(args):
        return False
    cmd, rest = args[i], list(args[i + 1:])
    if cmd in LISTING:
        return not rest or rest[0] in LISTING[cmd]
    if cmd == "rev-parse" and any(a.startswith(PATH_ANSWERS) for a in rest):
        return False
    return cmd in READ_ONLY


def exe_args(top, args):
    """the arguments for git.exe: a path after `--` that lies in the repository becomes repo-relative; None when an argument is an absolute path that git.exe would
    not read the same way (the WSL git then runs the call)"""
    top, out, tail = os.path.abspath(top), [], False
    for a in args:
        if a == "--":
            tail = True
        elif a.startswith("/") and not (tail and os.path.commonpath([top, a]) == top):
            return None
        elif tail and os.path.isabs(a):
            a = os.path.relpath(a, top).replace("\\", "/")
        out.append(a)
    return out


def run(top, *args, inp=None, timeout=None, env=None):
    """subprocess.CompletedProcess (bytes) of `git -C top <args>`: read-only commands through git.exe where the rule above allows it, else the WSL git"""
    data = inp.encode("utf-8") if isinstance(inp, str) else inp
    prefix = exe_prefix(top) if reads_only(args) and not any(k.startswith("GIT_") and not k.startswith(HARMLESS_ENV) for k in (os.environ if env is None else env)) else None
    xargs = exe_args(top, args) if prefix else None
    if xargs is not None:
        try:
            r = subprocess.run([*prefix, *xargs], capture_output=True, input=data, timeout=timeout, env=env)
            if not (r.returncode and ENV_ERROR.search(r.stderr)):
                return r
        except OSError:
            pass
        _BROKEN.add(top)
    return subprocess.run(["git", "-C", top, "-c", "core.quotepath=off", *args], capture_output=True, input=data, timeout=timeout, env=env)


def git(top, *args, check=True, inp=None):
    """stdout of a git call as text; check: a failure ends the tool with git's message (SystemExit)"""
    r = run(top, *args, inp=inp)
    if check and r.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr.decode('utf-8', 'replace').strip()[:200]}")
    return r.stdout.decode("utf-8", errors="replace")


def _selftest():
    real_exe, real_prefix = _exe(), exe_prefix
    stub = lambda f: globals().update(exe_prefix=f)                      # noqa: E731  (the stand-in for git.exe: a prefix function)
    # the rule: which calls read only, which answer with a Windows path, which arguments git.exe cannot take
    for ok in (("status", "--porcelain=v1", "-z"), ("--no-optional-locks", "-c", "a=b", "ls-files", "-z"), ("tag",), ("tag", "-l", "x/*"), ("tag", "--points-at", "abc"),
               ("remote",), ("rev-parse", "--verify", "-q", "HEAD^{commit}"), ("cat-file", "--batch-check"), ("ls-remote", "--tags", "origin"), ("grep", "-l", "-F", "x"),
               ("rev-list", "--left-right", "--count", "HEAD...origin/main"), ("log", "--format=%h %an %s"), ("ls-tree", "-r", "--name-only", "-z", "t"), ("merge-base", "--is-ancestor", "a", "b")):
        assert reads_only(ok), ok
    for bad in (("add", "-A"), ("commit", "-m", "x"), ("tag", "v1"), ("tag", "-d", "v1"), ("tag", "-a", "v1", "-m", "x"), ("remote", "add", "o", "u"), ("update-ref", "x", "y"),
                ("read-tree", "HEAD"), ("worktree", "add", "d", "t"), ("push", "origin"), ("rev-parse", "--show-toplevel"), ("rev-parse", "--git-dir"), ("rev-parse", "--absolute-git-dir"),
                ("--git-dir=x", "status"), ("-C", "x", "status"), (), ("config", "--get", "a"), ("branch", "-D", "x"), ("checkout", "x"), ("reset", "-q")):
        assert not reads_only(bad), bad
    assert drive_path("/mnt/c/Auto-research/x") == "C:/Auto-research/x" and drive_path("/mnt/d") == "D:/" and drive_path("/tmp/x") is None and drive_path("/mnt/wsl/x") is None
    assert exe_args("/mnt/c/r", ["ls-files", "--", "/mnt/c/r/a b/c.py", "rel/x"]) == ["ls-files", "--", "a b/c.py", "rel/x"]
    assert exe_args("/mnt/c/r", ["ls-files", "--", "/mnt/c/other/c.py"]) is None and exe_args("/mnt/c/r", ["grep", "-e", "/mnt/c/r/x"]) is None
    assert exe_args("/mnt/c/r", ["ls-files", "--", "/mnt/c/rr/x"]) is None, "a sibling folder with the same prefix is not inside the repository"
    saved = {k: os.environ.pop(k) for k in [k for k in os.environ if k.startswith("GIT_") and not k.startswith(HARMLESS_ENV)] + ["AR_GIT"] if k in os.environ}
    asked = []

    def fake(script):                                                    # a prefix that stands in for git.exe: `sh -c script _ <args>` answers whatever the script says
        def prefix(top):
            asked.append(top)
            return ["sh", "-c", script, "_"]
        return prefix
    try:
        with tempfile.TemporaryDirectory() as tmp:                       # a repository in a Linux folder: the WSL git, unless the stand-in is asked for
            def g(*a):
                subprocess.run(["git", "-C", tmp, "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True)
            g("init", "-q", "-b", "main")
            for name, body in (("plain.txt", "a\n"), ("a b.txt", "b\n"), ("\u00e4\u00f6 \u00fc.txt", "c\n"), ("crlf.txt", "d\r\ne\r\n")):
                with open(os.path.join(tmp, name), "wb") as f:
                    f.write(body.encode("utf-8"))
            g("add", "-A")
            g("commit", "-qm", "m\u00e4 base")
            with open(os.path.join(tmp, "new.txt"), "w") as f:
                f.write("x")
            assert exe_prefix(tmp) is None, "a Linux folder is no Windows drive"
            assert "\u00e4\u00f6 \u00fc.txt" in git(tmp, "ls-files", "-z") and "?? new.txt" in git(tmp, "status", "--porcelain=v1", "-z"), "raw non-ASCII names (quotepath off)"
            assert git(tmp, "cat-file", "--batch-check", inp="HEAD^{commit}\nnope^{commit}\n").splitlines()[1] == "nope^{commit} missing"
            stub(fake("echo FAKE"))                                      # a read-only call goes to the stand-in, a writing one never does
            assert git(tmp, "tag").strip() == "FAKE" and git(tmp, "status").strip() == "FAKE" and asked == [tmp, tmp]
            git(tmp, "tag", "v1")
            assert asked == [tmp, tmp] and git(tmp, "tag").strip() == "FAKE", "a tag is created by the WSL git"
            os.environ["AR_GIT"] = "wsl"                                 # forced WSL git: the real prefix function says None, whatever git.exe is there
            stub(real_prefix)
            assert git(tmp, "tag").split() == ["v1"] and exe_prefix(tmp) is None
            del os.environ["AR_GIT"]
            # fallbacks: git.exe says "not a git repository" / "dubious ownership" (retry with the WSL git, git.exe not used again for the folder) / fails for another reason (git's own answer)
            for script, retried in (("echo 'fatal: not a git repository: x' >&2; exit 128", True), ("echo 'fatal: detected dubious ownership' >&2; exit 128", True),
                                    ("echo 'fatal: bad revision' >&2; exit 128", False)):
                _BROKEN.clear()
                stub(fake(script))
                r = run(tmp, "rev-list", "--max-count=1", "HEAD")
                assert (r.returncode, len(r.stdout.split()), tmp in _BROKEN) == ((0, 1, True) if retried else (128, 0, False)), (script, r)
            _BROKEN.clear()
            stub(lambda top: ["/nonexistent/git.exe", "-C", top])
            assert git(tmp, "tag").split() == ["v1"] and tmp in _BROKEN, "git.exe cannot start (OSError): the WSL git answers"
            _BROKEN.clear()
            stub(fake("echo FAKE"))                                      # an environment that names a repository: no stand-in
            n = len(asked)
            os.environ["GIT_INDEX_FILE"] = os.path.join(tmp, ".git", "index")
            try:
                assert git(tmp, "tag").split() == ["v1"] and len(asked) == n
            finally:
                del os.environ["GIT_INDEX_FILE"]
            assert run(tmp, "tag", env=dict(os.environ, GIT_DIR=os.path.join(tmp, ".git"))).stdout.split() == [b"v1"] and len(asked) == n
            assert git(tmp, "cat-file", "-e", "nope^{commit}", check=False) == "FAKE\n" and len(asked) == n + 1     # check=False: the failure is the caller's to read
            stub(real_prefix)
            try:
                git(tmp, "rev-parse", "--verify", "nope")
                raise AssertionError("check=True must raise on a failure")
            except SystemExit as e:
                assert "git rev-parse --verify nope failed" in str(e)
    finally:
        stub(real_prefix)
        _BROKEN.clear()
        os.environ.pop("AR_GIT", None)
        os.environ.update(saved)
    # a repository on a Windows drive, when git.exe exists here: the same answers as the WSL git, also for names with a space, non-ASCII letters and CRLF content
    if real_exe and os.path.isdir("/mnt/c/Auto-research"):
        base = f"/mnt/c/Auto-research/temp/gitfast_test_{os.getpid()}"
        os.makedirs(base)
        try:
            def w(*a):                                                    # setup through the WSL git (writes)
                subprocess.run(["git", "-C", base, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false", *a], check=True, capture_output=True)
            w("init", "-q", "-b", "main")
            w("config", "core.filemode", "false")
            for name, body in (("plain.txt", b"a\n"), ("a b.txt", b"b\n"), ("\u00e4\u00f6 \u00fc.txt", b"c\n"), ("crlf.txt", b"d\ne\n"), ("raw_crlf.txt", b"d\r\ne\r\n"), ("sub/in.txt", b"f\n")):
                os.makedirs(os.path.dirname(os.path.join(base, name)), exist_ok=True)
                with open(os.path.join(base, name), "wb") as f:
                    f.write(body)
            w("add", "-A")
            w("commit", "-qm", "m\u00e4 base")
            w("tag", "v1")
            w("tag", "-a", "v2", "-m", "x")
            with open(os.path.join(base, "plain.txt"), "w") as f:
                f.write("changed\n")
            with open(os.path.join(base, "crlf.txt"), "wb") as f:           # LF in the index, CRLF in the work tree: Git for Windows (autocrlf=true) would call it unchanged
                f.write(b"d\r\ne\r\n")
            with open(os.path.join(base, "untracked \u00df.txt"), "w") as f:
                f.write("u")
            assert exe_prefix(base) is not None
            calls = [("status", "--porcelain=v1", "-z", "--untracked-files=all"), ("ls-files", "-z"), ("ls-files", "-z", "--others", "--exclude-standard"),
                     ("ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", os.path.join(base, "sub")), ("rev-list", "--all", "--count"), ("tag",), ("tag", "-l", "v*"),
                     ("tag", "--points-at", "HEAD"), ("log", "--format=%h %an %s"), ("diff", "--numstat"), ("diff-tree", "--root", "--no-commit-id", "-r", "-z", "--raw", "--no-abbrev", "HEAD"),
                     ("rev-parse", "--short=10", "HEAD"), ("rev-parse", "--verify", "-q", "v2^{commit}"), ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"),
                     ("cat-file", "-e", "HEAD^{commit}"), ("show", "HEAD:raw_crlf.txt"), ("ls-tree", "-r", "--name-only", "-z", "v1"), ("grep", "-l", "-F", "-z", "-e", "changed"),
                     ("for-each-ref", "--format=%(refname)"), ("merge-base", "--is-ancestor", "v1", "HEAD"), ("remote",), ("status", "--short")]
            for a in calls:
                os.environ["AR_GIT"] = "wsl"
                try:
                    slow = run(base, *a)
                finally:
                    del os.environ["AR_GIT"]
                fast = run(base, *a)
                assert (slow.returncode == 0) == (fast.returncode == 0) and slow.stdout == fast.stdout, (a, slow, fast)
            assert git(base, "diff", "--numstat") == "2\t2\tcrlf.txt\n1\t1\tplain.txt\n", "CRLF: the WSL answer (autocrlf=false), not Git for Windows' (the line change of crlf.txt hidden)"
            assert git(base, "status", "--porcelain=v1").count("\n") == 3, "no other noise: raw_crlf.txt is clean, crlf.txt and plain.txt changed, one untracked"
            assert git(base, "cat-file", "--batch-check", inp="HEAD^{commit}\nnope^{commit}\n").splitlines()[1] == "nope^{commit} missing"
            assert git(base, "show", "HEAD:raw_crlf.txt") == "d\r\ne\r\n", "bytes come out untouched"
            assert not _BROKEN, "git.exe fell back for an environment reason"
        finally:
            shutil.rmtree(base, ignore_errors=True)
            os.environ.pop("AR_GIT", None)
        print("gitfast selftest ok: rule, stand-in routing, fallbacks, Windows-drive repository identical to the WSL git")
    else:
        print("gitfast selftest ok: rule, stand-in routing, fallbacks (no git.exe / Windows drive here: the real comparison was skipped)")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        raise SystemExit(_selftest())
    print(__doc__)
