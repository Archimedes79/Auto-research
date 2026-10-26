#!/usr/bin/env python3
"""Would things work if the working folder were deleted? (decision D11)

    wsl python3 tools/maintenance/reproducible.py              # all checks (about 1 - 5 min, CPU only)
    wsl python3 tools/maintenance/reproducible.py --no-clone   # static checks only (seconds)
    wsl python3 tools/maintenance/reproducible.py --keep       # keep the clone for a look
    wsl python3 tools/maintenance/reproducible.py --as-if-committed   # test what git WOULD hold if everything not ignored were committed now
    wsl python3 tools/maintenance/reproducible.py selftest

Reads `data/rebuild.json` (git policy, clone checks, external prerequisites, the rebuild manifest of derived data) and checks four things:
  1. policy    files that are tracked but must not be (logs, heartbeats, binaries, caches, anything above max_kb) - the rule of CLAUDE.md "Git"
  2. needed    python modules that tracked code imports and that exist on disk but are NOT in git (they would be gone after a clone)
  3. manifest  every derived-data artifact names the script that makes it; the script must be tracked and exist; unscripted artifacts are listed
  4. clone     (checks of kind "info" are reported but do not fail the run) `git clone` of the committed HEAD into a temporary folder, then the cheap checks of `clone_checks` run there
               (compile all code, structure, literature, golden numbers ...): this is "delete the working folder and start from git"
Also reported: uncommitted changes (they are not in the clone) and the external prerequisites (raw nuScenes, the conda env, clones of foundation code).
Exit code 0 only if nothing failed. Standard library only; run it with the WSL python3. The clone is a temporary folder that the tool removes.
"""
import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROBLEM = os.path.normpath(os.path.join(HERE, "..", ".."))


def git(repo, *args, check=True):
    r = subprocess.run(["git", "-C", repo] + list(args), capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def repo_root(problem):
    return os.path.normpath(git(problem, "rev-parse", "--show-toplevel").strip())


def load(problem):
    return json.load(open(os.path.join(problem, "data", "rebuild.json"), encoding="utf-8"))


def tracked(repo):
    return [f for f in git(repo, "ls-files", "-z").split("\0") if f]


def others(repo):
    """files that are changed or new and not ignored: they would be in git after `git add -A`"""
    return [f for f in git(repo, "ls-files", "-m", "-o", "--exclude-standard", "-z").split("\0") if f]


def rel_to_repo(repo, problem, p):
    return os.path.relpath(os.path.normpath(os.path.join(problem, p)), repo).replace("\\", "/")


# ------------------------------------------------------------------ 1 policy
def policy(repo, files, cfg):
    pol = cfg.get("git_policy", {})
    names, exts, dirs = set(pol.get("forbid_names", [])), tuple(pol.get("forbid_ext", [])), set(pol.get("forbid_dirs", []))
    allow, max_b = pol.get("allow", []), int(pol.get("max_kb", 1024)) * 1024
    bad = []
    for f in files:
        if any(fnmatch.fnmatch(f, a) for a in allow):
            continue
        base = os.path.basename(f)
        why = None
        if any(fnmatch.fnmatch(base, n) for n in names):
            why = "per-run temporary file"
        elif base.lower().endswith(exts):
            why = "forbidden type (log, binary, cache, checkpoint)"
        elif dirs & set(f.split("/")):
            why = "forbidden folder"
        else:
            try:
                if os.path.getsize(os.path.join(repo, f)) > max_b:
                    why = f"larger than {max_b // 1024} KB"
            except OSError:
                pass
        if why:
            bad.append((f, why))
    return bad


# ------------------------------------------------------------------ 2 needed but untracked
IMPORT = re.compile(r"^\s*(?:from\s+([A-Za-z_]\w*)[\w.]*\s+import|import\s+([A-Za-z_]\w*))", re.M)


def needed_untracked(repo, problem, files, cfg):
    tset = set(files)
    code_dirs = [os.path.join(problem, d) for d in cfg.get("code_dirs", [])]
    on_disk = {}
    for top in code_dirs:
        for dp, dn, fn in os.walk(top):
            dn[:] = [d for d in dn if d != "__pycache__"]
            for f in fn:
                if f.endswith(".py"):
                    on_disk.setdefault(f[:-3], []).append(os.path.relpath(os.path.join(dp, f), repo).replace("\\", "/"))
    out = []
    for f in files:
        if not f.endswith(".py") or not any(os.path.normpath(os.path.join(repo, f)).startswith(d + os.sep) for d in code_dirs):
            continue
        try:
            src = open(os.path.join(repo, f), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for m in IMPORT.finditer(src):
            mod = m.group(1) or m.group(2)
            cands = on_disk.get(mod)
            if cands and not any(c in tset for c in cands):
                out.append((f, mod, cands[0]))
    return sorted(set(out))


# ------------------------------------------------------------------ 3 manifest
def manifest(repo, problem, files, cfg):
    tset = set(files)
    rows, fails = [], []
    for a in cfg.get("artifacts", []):
        scripts = a.get("scripts", [])
        missing = [s for s in scripts if rel_to_repo(repo, problem, s) not in tset]
        gone = [s for s in scripts if rel_to_repo(repo, problem, s) in tset and not os.path.exists(os.path.join(repo, rel_to_repo(repo, problem, s)))]
        status = a.get("status", "scripted")
        if status == "missing":
            fails.append(f"{a['id']}: no recipe yet (missing): {a.get('note', '')[:120]}")
        if status == "scripted" and not scripts:
            fails.append(f"{a['id']}: status scripted but no script named")
        for s in missing:
            fails.append(f"{a['id']}: producer {s} is not tracked in git")
        for s in gone:
            fails.append(f"{a['id']}: producer {s} does not exist")
        rows.append((a["id"], status, a.get("where", ""), len(scripts), a.get("note", "")))
    return rows, fails


# ------------------------------------------------------------------ 4 clone test
def clone_test(repo, problem, cfg, keep=False, overlay=False):
    tmp = tempfile.mkdtemp(prefix="ar_clone_")
    results = []
    try:
        subprocess.run(["git", "clone", "-q", "--no-hardlinks", repo, tmp], check=True, capture_output=True, text=True)
        head = git(tmp, "rev-parse", "--short", "HEAD").strip()
        if overlay:
            for f in others(repo):
                src = os.path.join(repo, f)
                if os.path.isfile(src):
                    os.makedirs(os.path.dirname(os.path.join(tmp, f)), exist_ok=True)
                    shutil.copy2(src, os.path.join(tmp, f))
        cproblem = os.path.join(tmp, os.path.relpath(problem, repo))
        env = {k: v for k, v in os.environ.items() if k not in ("BFLC_ROOT", "AR_PROJECT")}
        for c in cfg.get("clone_checks", []):
            t0 = time.time()
            try:
                r = subprocess.run(["bash", "-c", c["cmd"]], cwd=cproblem, env=env, capture_output=True, text=True, timeout=int(c.get("timeout", 600)))
                ok, out = r.returncode == 0, (r.stdout + r.stderr).strip().split("\n")[-1][:110]
            except subprocess.TimeoutExpired:
                ok, out = False, "timeout"
            results.append((c["name"], ok, out, time.time() - t0, c.get("kind", "required"), c.get("why", "")))
        return head, results
    finally:
        if keep:
            print("clone kept at", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ report
def run(problem, no_clone=False, keep=False, quiet=False, overlay=False):
    repo = repo_root(problem)
    cfg = load(problem)
    files = tracked(repo)
    if overlay:
        files = sorted(set(files) | set(others(repo)))
    fail = 0
    say = (lambda *a: None) if quiet else print
    say(f"repository {os.path.basename(repo)} | {len(files)} tracked files | HEAD {git(repo, 'rev-parse', '--short', 'HEAD').strip()}")

    bad = policy(repo, files, cfg)
    say(f"\n1 policy: {len(bad)} tracked files break the git rules")
    why = {}
    for f, w in bad:
        why.setdefault(w, []).append(f)
    for w, fs in why.items():
        say(f"   {len(fs):5d}  {w}: e.g. {fs[0]}")
    fail += bool(bad)

    nu = needed_untracked(repo, problem, files, cfg)
    say(f"\n2 needed: {len(nu)} imports of modules that exist on disk but are not in git")
    for f, mod, c in nu[:12]:
        say(f"   {f} imports {mod} ({c})")
    fail += bool(nu)

    rows, mfails = manifest(repo, problem, files, cfg)
    cnt = {}
    for r in rows:
        cnt[r[1]] = cnt.get(r[1], 0) + 1
    say(f"\n3 manifest: {len(rows)} derived-data artifacts: " + ", ".join(f"{v} {k}" for k, v in sorted(cnt.items())))
    for r in rows:
        if r[1] != "scripted":
            say(f"   {r[1]:8s} {r[0]}: {r[4]}")
    for m in mfails:
        say("   FAIL", m)
    fail += bool(mfails)

    dirty = [l for l in git(repo, "status", "--porcelain").split("\n") if l.strip()]
    say(f"\n  uncommitted: {len(dirty)} paths (" + ("simulated as committed: --as-if-committed" if overlay else "they are NOT in the clone below") + ")")
    for l in dirty[:6]:
        say("   ", l)

    if no_clone:
        say("\n4 clone test skipped (--no-clone)")
    else:
        say("\n4 clone test: git clone of HEAD into a temporary folder, then the checks run there")
        head, res = clone_test(repo, problem, cfg, keep, overlay)
        for name, ok, out, sec, kind, why in res:
            tag = "PASS" if ok else ("FAIL" if kind == "required" else "INFO")
            say(f"   {tag}  {name} ({sec:.0f} s)  | {out}")
            if not ok and why:
                say(f"         why: {why}")
        fail += any(not r[1] and r[4] == "required" for r in res)

    say("\nexternal prerequisites (outside git, by design):")
    for e in cfg.get("external", []):
        say(f"   - {e['what']}: {e['where']}  [{e.get('recipe', '')}]")
    say("\nVERDICT: " + ("everything needed is in git or rebuildable by a listed recipe" if not fail else f"NOT clean: {fail} of 4 checks failed (see above)"))
    return fail


# ------------------------------------------------------------------ selftest
def selftest():
    root = tempfile.mkdtemp(prefix="ar_repro_selftest_")
    try:
        prob = os.path.join(root, "repo", "proj")
        os.makedirs(os.path.join(prob, "data"))
        os.makedirs(os.path.join(prob, "code"))
        repo = os.path.join(root, "repo")
        subprocess.run(["git", "init", "-q", repo], check=True)
        subprocess.run(["git", "-C", repo, "config", "user.email", "t@example.invalid"], check=True)
        subprocess.run(["git", "-C", repo, "config", "user.name", "t"], check=True)
        open(os.path.join(prob, "code", "main.py"), "w").write("import helper\nimport os\n")
        open(os.path.join(prob, "code", "helper.py"), "w").write("X = 1\n")
        open(os.path.join(prob, "code", "run.log"), "w").write("temp\n")
        open(os.path.join(prob, "code", "ok.sh"), "w").write("echo ok\n")
        cfg = {"code_dirs": ["code"], "git_policy": {"forbid_names": ["run.log"], "forbid_ext": [".pt"], "max_kb": 1},
               "clone_checks": [{"name": "good", "cmd": "test -f code/main.py"}, {"name": "bad", "cmd": "test -f code/helper.py"}, {"name": "info", "cmd": "false", "kind": "info", "why": "x"}],
               "artifacts": [{"id": "a1", "scripts": ["code/ok.sh"], "status": "scripted"}, {"id": "a2", "scripts": ["code/nope.sh"], "status": "scripted"},
                             {"id": "a3", "scripts": [], "status": "missing", "note": "not scripted yet"}]}
        json.dump(cfg, open(os.path.join(prob, "data", "rebuild.json"), "w"))
        subprocess.run(["git", "-C", repo, "add", "proj/code/main.py", "proj/code/run.log", "proj/code/ok.sh", "proj/data/rebuild.json"], check=True)
        subprocess.run(["git", "-C", repo, "commit", "-q", "-m", "t"], check=True)
        files = tracked(repo)
        pb = policy(repo, files, cfg)
        assert [f for f, _ in pb] == ["proj/code/run.log"], pb
        nu = needed_untracked(repo, prob, files, cfg)
        assert nu == [("proj/code/main.py", "helper", "proj/code/helper.py")], nu
        rows, fails = manifest(repo, prob, files, cfg)
        assert len(rows) == 3 and len(fails) == 2 and "nope.sh" in fails[0] and "no recipe" in fails[1], (rows, fails)
        head, res = clone_test(repo, prob, cfg)
        assert [(r[0], r[1]) for r in res] == [("good", True), ("bad", False), ("info", False)], res
        assert run(prob, quiet=True) == 4  # policy, needed, manifest and clone all fail in this toy repository
        print("reproducible selftest passed")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", default="check", choices=["check", "selftest"])
    ap.add_argument("--no-clone", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--as-if-committed", action="store_true")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    else:
        sys.exit(1 if run(PROBLEM, a.no_clone, a.keep, overlay=a.as_if_committed) else 0)
