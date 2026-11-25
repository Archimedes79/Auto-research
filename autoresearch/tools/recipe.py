#!/usr/bin/env python3
"""recipe.py - repeat one run from the repository alone: the brief for an AI (or a person) and the little that has to be stored for it.

    python3 autoresearch/tools/recipe.py show <run id>                 the brief: question and rule, the code (base commit + stored files), command, inputs, expected result, the steps, the gaps
    python3 autoresearch/tools/recipe.py repeat <run id> [--dir D]     the worktree of the base commit with the stored files laid over it; prints the run.py command that repeats the run (starts nothing)
    python3 autoresearch/tools/recipe.py check [--id E346] [--list]    which runs can be repeated from a fresh clone (exit 1 when a selected run has a gap)
    python3 autoresearch/tools/recipe.py store <id or prefix> ... | --all [--apply]    write recipe.json + code/ into the experiment folders (a dry run without --apply)
    python3 autoresearch/tools/recipe.py --selftest

Problem folder: AR_PROJECT as for every tool. Standard library only; git is read through gitfast.py.

What the git history of the lane problem showed (2026-11-24, 1215 ledger runs, 1058 with a code tag) and what follows from it:
  - the command of 87 % of the runs names only scripts that main holds at the commit the run was started on (the parent of the run's snapshot); 24 scripts (137 KB) of the
    other runs exist only in a snapshot
  - the working tree at the start differs from that commit in tracked code in 94 % of the runs (uncommitted edits: median 78 KB, up to 41 files), and a snapshot adds
    about 1700 scratch files, nearly all of them unrelated to the run. A tag is therefore a poor record: 450 tags are local only (they hold scratch and names), 157 runs have
    none, and a clone cannot read them
  - so a run needs three things in git: the BASE (the main commit it started on: 40 characters), the command (ledger) and the OVERLAY: only the files its command can execute
    (the script and, to the end of the chain, what it imports or runs) that differ from the base; a version that main's history holds (most dirty edits were committed
    later) is named by its blob id, the rest is copied once under the experiment folder as code/<blob id>_<name>
  - everything else is derived at read time from what git holds anyway (ledger row, card, tracked result files, rebuild recipes), never copied
The record is <experiment folder>/recipe.json: {"schema": 2, "overlays": [...each distinct overlay once...], "runs": {<run id>: {"base", "how", "snapshot", "overlay": {<repo path>: {"blob", "file"?}}, "truncated"?}}}.
A run without a local tag gets its base from the ledger's start time (the last first-parent commit of origin/main before it): marked "by time", exact in about 96 % of the
tagged runs (the others waited in the queue after the snapshot); its overlay is empty, so `check` also tests that the scripts of its command exist at that base.
`store` never replaces an exact entry (tag parent) by an approximate one; it needs the tag, so run it on the machine that ran the experiment, then commit recipe.json and code/.
Not covered: the data and checkpoints a command reads (`show` lists the paths it names and where they come from), the environment's package versions (the env files are named),
and files that are not code (configs, registry json) that differed from the base: `repeat` runs those at the base's version.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gitfast  # noqa: E402
import project as P  # noqa: E402

MAX_FILES, MAX_BLOB = 600, 1_000_000
NAMED = re.compile(r"[\w./~${}-]+\.(?:py|sh)\b")
IMPORT = re.compile(r"^[ \t]*(?:from[ \t]+([\w.]+)[ \t]+import[ \t]+(\([^)]*\)|[^\n]+)|import[ \t]+([\w., \t]+))", re.M)
DATA_EXT = (".pt", ".pth", ".ckpt", ".pkl", ".npy", ".npz", ".json", ".csv", ".yml", ".yaml")
EXACT = ("snapshot parent", "pinned commit")     # how a base was found when it is not a guess


def blob_id(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def load_private(top):
    """the patterns that copied code must not carry: the file `.private_patterns` at the top of the repository (never in git: .gitignore ignores it), one `regex => replacement`
    per line, # starts a comment. Without the file nothing is changed. It holds the names and addresses of the people and machines of this checkout"""
    p = os.path.join(top, ".private_patterns")
    out = []
    if os.path.isfile(p):
        for line in open(p, encoding="utf-8"):
            line = line.rstrip("\n")
            if line.strip() and not line.lstrip().startswith("#") and " => " in line:
                rx, rep = line.split(" => ", 1)
                out.append((re.compile(rx), rep))
    return out


def apply_private(data, patterns):
    """the bytes of a text file with the private patterns applied (a file that is not UTF-8 is returned as it is)"""
    try:
        s = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    for rx, rep in patterns:
        s = rx.sub(rep, s)
    return s.encode("utf-8")


class Repo:
    def __init__(self, problem_root=None):
        self.root = os.path.abspath(problem_root or P.root())
        self.top = gitfast.git(self.root, "rev-parse", "--show-toplevel").strip()
        self.pf = os.path.relpath(self.root, self.top).replace("\\", "/")
        self.pf = "" if self.pf == "." else self.pf
        self.prefix = P.cfg_of(self.root)["tag_prefix"]
        pre = (re.escape(self.pf) + "/") if self.pf else ""
        # folders that hold no code a run executes: the data / records / literature folders at the top of the repository or of the problem folder, and the stored overlay itself
        self.skip = re.compile(rf"^(?:{pre})?(?:literature|data|results|runs|models|reports/source|winner|_to_delete|\.claude)/|^data/|(?:^|/)winner/|(?:^|/)_to_delete/|(?:^|/)experiments/[^/]+/code/")
        self._trees, self._tags, self._tracked, self._refs, self._dirty, self._anc, self._have, self._hist, self._main = {}, None, None, {}, {}, {}, {}, None, {}

    def g(self, *a):
        return gitfast.git(self.top, *a)                 # a failing git call ends the tool: a half-read repository must not look like an exact record

    def rel(self, p):
        """a path inside the problem folder as a repository path"""
        return posixpath.normpath(posixpath.join(self.pf, p)) if self.pf else posixpath.normpath(p)

    def tags(self):
        if self._tags is None:
            self._tags = set(self.g("tag", "-l", self.prefix + "/*").split())
        return self._tags

    def exists(self, commit):
        if not commit:
            return False
        if commit not in self._have:
            self._have[commit] = gitfast.run(self.top, "cat-file", "-e", commit + "^{commit}").returncode == 0
        return self._have[commit]

    def top_names(self):
        """the names of the top-level entries of the repository (for the absolute paths of a command)"""
        return {n for n in self.g("ls-tree", "--name-only", "HEAD").split("\n") if n}

    def index(self, paths):
        byname = {}
        for p in paths:
            if p.endswith((".py", ".sh")) and not self.skip.search(p):
                byname.setdefault(posixpath.basename(p), []).append(p)
                if p.endswith("/__init__.py"):
                    byname.setdefault(posixpath.basename(posixpath.dirname(p)) + "/", []).append(p)
        return byname

    def tree(self, commit):
        """(path -> blob id, file name -> code paths) of a commit; cached by tree id"""
        t = self.g("rev-parse", commit + "^{tree}").strip()
        if t not in self._trees:
            paths = {}
            for line in self.g("ls-tree", "-r", t).split("\n"):
                if "\t" in line:
                    meta, p = line.split("\t", 1)
                    paths[p] = meta.split()[2]
            self._trees[t] = (paths, self.index(paths))
        return self._trees[t]

    def snap_tree(self, base, snap):
        """the tree of a snapshot as the tree of its base plus the difference (a snapshot adds about 1700 scratch files: listing it whole costs seconds)"""
        bp, bn = self.tree(base)
        paths, byname = dict(bp), dict(bn)
        for line in self.g("diff-tree", "-r", "--no-renames", "--raw", base, snap).split("\n"):
            if "\t" not in line:
                continue
            meta, p = line.split("\t", 1)
            m = meta.split()
            if m[4] == "D":
                paths.pop(p, None)
                n = posixpath.basename(p)
                if p in byname.get(n, []):
                    byname[n] = [c for c in byname[n] if c != p]
                continue
            paths[p] = m[3]
            n = posixpath.basename(p)
            if m[4] == "A" and p.endswith((".py", ".sh")) and not self.skip.search(p):
                byname[n] = byname.get(n, []) + [p]
                if p.endswith("/__init__.py"):
                    k = posixpath.basename(posixpath.dirname(p)) + "/"
                    byname[k] = byname.get(k, []) + [p]
        return paths, byname

    def blobs(self, oids):
        """blob id -> bytes, read in batches (the stdin that git.exe takes from WSL is limited)"""
        out, oids = {}, list(dict.fromkeys(oids))
        for i in range(0, len(oids), 300):
            data = gitfast.run(self.top, "cat-file", "--batch", inp="\n".join(oids[i:i + 300]) + "\n").stdout
            pos = 0
            while pos < len(data):
                nl = data.index(b"\n", pos)
                head, pos = data[pos:nl].split(), nl + 1
                if len(head) >= 3 and head[1] == b"blob":
                    n = int(head[2])
                    out[head[0].decode()] = data[pos:pos + n]
                    pos += n + 1
        return out

    def refs_of(self, items):
        """[(path, blob id)] -> {blob id: the code names the file can run}, read once per blob id"""
        need = [o for _, o in items if o not in self._refs]
        data = self.blobs(need)
        for p, o in items:
            if o not in self._refs:
                self._refs[o] = refs(p, data.get(o, b"").decode("utf-8", "replace")) if len(data.get(o, b"")) <= MAX_BLOB else []
        return {o: self._refs[o] for _, o in items}

    def in_main(self, commit):
        """True when the commit is in the history of origin/main (HEAD without a remote): a pinned run's commit is, a snapshot is not"""
        if commit not in self._main:
            ref = "origin/main" if self.exists("origin/main") else "HEAD"
            self._main[commit] = gitfast.run(self.top, "merge-base", "--is-ancestor", commit, ref).returncode == 0
        return self._main[commit]

    def base_of(self, row):
        """(base commit, how, snapshot commit). The snapshot is the commit of the LAST attempt that the ledger records (run.py rewrites it on every resume), else the run's tag (attempt 1);
        its first parent is the base, the commit the working tree was snapshotted on. A commit that is in main's history is a pinned run (the code of a named commit): its own base.
        Without a snapshot: the last first-parent commit of origin/main (HEAD) before the ledger's start time, a guess"""
        snap = (row.get("commit") or "").strip()
        if not self.exists(snap):
            tag = self.prefix + "/" + row["id"]
            snap = self.g("rev-parse", tag + "^{commit}").strip() if tag in self.tags() else ""
        if snap:
            c = self.g("rev-list", "--parents", "-n", "1", snap).split()
            if self.in_main(c[0]):
                return c[0], "pinned commit", c[0]
            if len(c) >= 2:
                return c[1], "snapshot parent", c[0]
        started = (row.get("started") or "").replace("T", " ")
        ref = "origin/main" if self.exists("origin/main") else "HEAD"
        c = self.g("rev-list", "-n", "1", "--first-parent", f"--before={started}", ref).strip() if started else ""
        return c, "by time (approximate)" if c else "unknown", (row.get("commit") or "").strip()

    def history_blobs(self):
        """the ids of every object in the history of origin/main (HEAD without a remote): a file version that is in it needs no copy, and a clone has it"""
        if self._hist is None:
            ref = "origin/main" if self.exists("origin/main") else "HEAD"
            self._hist = {l.split()[0] for l in self.g("rev-list", "--objects", ref).split("\n") if l.strip()}
        return self._hist

    def pushed(self, folder):
        """True / False / None (no origin/main): the last commit that touched recipe.json or code/ of the folder is on origin/main"""
        rel = os.path.relpath(folder, self.top).replace("\\", "/")
        c = self.g("log", "-1", "--format=%H", "--", rel + "/recipe.json", rel + "/code").strip()
        return self.on_origin(c) if c else None

    def on_origin(self, commit):
        """True / False, or None when this repository has no origin/main to ask"""
        if not self.exists("origin/main"):
            return None
        if commit not in self._anc:
            self._anc[commit] = gitfast.run(self.top, "merge-base", "--is-ancestor", commit, "origin/main").returncode == 0
        return self._anc[commit]

    def uncommitted(self, folder):
        """True when recipe.json or code/ of an experiment folder is not committed as it is on disk"""
        if folder not in self._dirty:
            rel = os.path.relpath(folder, self.top).replace("\\", "/")
            self._dirty[folder] = bool(self.g("status", "--porcelain", "--", rel + "/recipe.json", rel + "/code").strip())
        return self._dirty[folder]

    def in_git(self, folder):
        """True when the experiment folder holds a file that git tracks (a folder that only exists on disk is another chat's work in progress: no recipe is put into it)"""
        rel = os.path.relpath(folder, self.top).replace("\\", "/")
        return bool(self.g("ls-files", "--", rel).strip())

    def tracked(self):
        if self._tracked is None:
            self._tracked = [p for p in self.g("ls-files", "--", self.pf or ".").split("\n") if p]
        return self._tracked


def command_scripts(cmd):
    try:
        parts = shlex.split(cmd)
    except ValueError:
        parts = cmd.split()
    return [p for p in (q.strip(";&|") for q in parts) if re.search(r"\.(py|sh)$", p) and not p.startswith("-")]


EXEC_LINE = re.compile(r"subprocess|os\.system|Popen|check_call|check_output|\bbash\b|\bpython3?\b|runpy|execfile|\bsh\b")


def refs(path, text):
    """what a file can run, as (is_module, token): its imports (the module, its dotted parts and the imported names, also in a parenthesised list over several lines) and the
    .py / .sh files it names where it executes them (every non-comment line of a shell script; in Python only a line that starts a process). Prose and comments name files
    without running them: following them made closures of hundreds of files"""
    out = []
    if path.endswith(".py"):
        for m in IMPORT.finditer(text):
            if m.group(1):
                names = [re.sub(r"\s+as\s+\w+", "", re.sub(r"#[^\n]*", "", x)).strip(" ()\t\r\n") for x in re.split(r"[,\n]", m.group(2))]
                mods = m.group(1).split(".") + names
            else:
                mods = [x.strip().split(" as ")[0] for x in m.group(3).split(",")]
            for md in mods:
                for part in md.split("."):
                    if part and part != "*" and re.fullmatch(r"\w+", part):
                        out += [(True, part + ".py"), (True, part + "/")]
        named = [l for l in text.split("\n") if EXEC_LINE.search(l) and not l.lstrip().startswith("#")]
    else:
        named = [re.sub(r"\s#.*$", "", l) for l in text.split("\n") if not l.lstrip().startswith("#")]
    return out + [(False, m.group(0)) for l in named for m in NAMED.finditer(l)]


def resolve(byname, tok, importer, strict=True):
    """the code paths a name or a path written in `importer` stands for: a path with a folder part must match the end of a path (or lie relative to the importer), an absolute
    one is matched by its longest tail, a bare name means the importer's folder first, then the only file of that name, then the only one of the importer's problem folder, then the only one outside temp/; a bare name that is still
    ambiguous is dropped when strict (a generic name like show.py would pull in every namesake)"""
    if tok.endswith("/"):
        t, cands = tok.rstrip("/"), byname.get(tok, [])
    else:
        t = posixpath.normpath(re.sub(r"^(?:\$\{?\w+\}?|~)/", "", tok.replace("\\", "/")))
        cands = byname.get(posixpath.basename(t), [])
        if t.startswith("/") or re.match(r"[A-Za-z]:", t):
            parts = t.strip("/").split("/")
            for i in range(len(parts)):
                hits = [c for c in cands if c == "/".join(parts[i:]) or c.endswith("/" + "/".join(parts[i:]))]
                if hits:
                    return hits
            return []
        if "/" in t:
            near = posixpath.normpath(posixpath.join(posixpath.dirname(importer), t))
            return [c for c in cands if c.endswith("/" + t) or c == t or c == near]
    same = [c for c in cands if posixpath.dirname(c) == posixpath.dirname(importer)]
    if same:
        return same
    if len(cands) == 1:
        return cands
    local = [c for c in cands if c.split("/", 1)[0] == importer.split("/", 1)[0] and "/temp/" not in "/" + c]
    if len(local) == 1:                                                  # the same name in two problem folders (losses.py): the importer's own
        return local
    out = [c for c in cands if "/temp/" not in "/" + c]
    return out if len(out) == 1 else ([] if strict else cands)


def closure(repo, snap, cmd):
    """(the code files the command can execute in a tree {path: blob id}, truncated)"""
    paths, byname = snap
    seen, level, unresolved = {}, [], []
    for tok in command_scripts(cmd):
        hits = resolve(byname, tok.replace("\\", "/").split("ML-Experiment/")[-1], repo.rel("x"), strict=False)
        level += hits
        if not hits:
            unresolved.append(tok)
    truncated = False
    while level:
        level = [p for p in dict.fromkeys(level) if p not in seen and p in paths]
        if not level:
            break
        if len(seen) + len(level) > MAX_FILES:
            truncated = True
            level = level[:max(0, MAX_FILES - len(seen))]
        seen.update({p: paths[p] for p in level})
        rf = repo.refs_of([(p, paths[p]) for p in level])
        level = [c for p in level for _, r in rf[paths[p]] for c in resolve(byname, r, p)]
        if truncated:
            break
    return seen, truncated, unresolved


def needed(repo, row, base, snapshot):
    """({path: blob id} of the files the command executes that differ from the base commit, truncated, the scripts of the command that were found nowhere), or None when the
    base or the snapshot cannot be read here"""
    if not (repo.exists(base) and repo.exists(snapshot)):
        return None
    bt = repo.tree(base)[0]
    cl, truncated, unresolved = closure(repo, repo.snap_tree(base, snapshot), row.get("command", ""))
    return {p: o for p, o in cl.items() if bt.get(p) != o}, truncated, unresolved


def ledger_rows():
    path = P.path("ledger")
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def exp_dir(run_id):
    """the experiment folder of a run: <prefix before the first dot>_<slug>; None when there is none or several"""
    pre = run_id.split(".")[0]
    hits = []
    for d in P.experiment_dirs():
        if os.path.isdir(d):
            hits += [os.path.join(d, n) for n in sorted(os.listdir(d)) if n == pre or n.startswith(pre + "_")]
    if len(hits) <= 1:
        return hits[0] if hits else None
    named = [h for h in hits if os.path.isfile(os.path.join(h, "experiment.md")) and run_id in open(os.path.join(h, "experiment.md"), encoding="utf-8", errors="ignore").read()]
    if len(named) == 1:                                                  # several folders share the prefix: the one whose card names the run
        return named[0]
    rest = run_id.split(".", 1)[1].replace(".", "_") if "." in run_id else ""
    slug = [h for h in hits if rest and "_" in os.path.basename(h) and rest.startswith(os.path.basename(h).split("_", 1)[1])]
    return slug[0] if len(slug) == 1 else None                           # else the run has no folder it belongs to


def expand_recipe(raw):
    """the runs of a recipe.json with their overlays as dicts (the file keeps each distinct overlay once and names it by its index: the runs of an experiment mostly share one)"""
    ov = raw.get("overlays", [])
    for e in raw.get("runs", {}).values():
        if isinstance(e.get("overlay"), int):
            e["overlay"] = ov[e["overlay"]]
    return raw


def load_recipe(folder):
    p = os.path.join(folder, "recipe.json")
    if not os.path.isfile(p):
        return {"schema": 2, "runs": {}}
    raw = expand_recipe(json.load(open(p, encoding="utf-8")))
    raw.pop("overlays", None)
    return raw


def dump_recipe(rec):
    """the text of recipe.json: one line per distinct overlay and one per run (a readable diff, and a file far below the size limit)"""
    overlays, index, runs = [], {}, {}
    for rid, e in rec["runs"].items():
        key = json.dumps(e.get("overlay", {}), sort_keys=True)
        if key not in index:
            index[key] = len(overlays)
            overlays.append(key)
        runs[rid] = dict(e, overlay=index[key])
    return ('{"schema": 2,\n "overlays": [\n  ' + ",\n  ".join(overlays) + '\n ],\n "runs": {\n  '
            + ",\n  ".join(f"{json.dumps(rid)}: {json.dumps(runs[rid], sort_keys=True)}" for rid in sorted(runs)) + "\n }}\n")


SHELL_C = re.compile(r"^((?:env\s+)?(?:[A-Za-z_]\w*=\S+\s+)*(?:nice\s+-n\s+\d+\s+)?)(bash|sh)\s+-c\s+(.+)$", re.S)


def printable(cmd, dest, top_names, top=""):
    """the ledger's command as a command line again: an absolute path of a Windows drive mount (/mnt/c/...) that leads into a folder of the repository is moved into `dest` (the
    checkout it was made in may have had another name, or lie elsewhere), and `[env VAR=x] bash -c <words>` (run.py joined the arguments with spaces and lost the quotes) gets its
    one argument back"""
    dest = dest.rstrip("/")

    def move(m):
        parts = m.group(0).split("/")
        for i in range(2, len(parts)):
            if parts[i] in top_names:
                return dest + "/" + "/".join(parts[i:])
        return m.group(0)

    cmd = re.sub(r"/mnt/[a-zA-Z]/[^\s'\"]+", move, cmd)
    if top:
        cmd = cmd.replace(top.rstrip("/") + "/", dest + "/")
    m = SHELL_C.match(cmd)
    try:
        words = len(shlex.split(m.group(3))) if m else 0
    except ValueError:
        return cmd
    return f"{m.group(1)}{m.group(2)} -c " + shlex.quote(m.group(3)) if m and words > 1 else cmd


def outside(path):
    """True for an overlay path that would leave the worktree (a recipe.json is text from a repository: it is read, not trusted)"""
    return path.startswith("/") or ".." in path.split("/") or bool(re.match(r"[A-Za-z]:", path))


def missing_scripts(repo, row, ent):
    """the .py / .sh names of the command that are neither in the base tree nor stored with the entry"""
    if not repo.exists(ent["base"]):
        return []
    names = repo.tree(ent["base"])[1]
    stored = {posixpath.basename(p) for p in ent.get("overlay", {})}
    out = []
    for tok in command_scripts(row.get("command", "")):
        n = posixpath.basename(posixpath.normpath(tok.replace("\\", "/")))
        if n not in names and n not in stored:
            out.append(tok)
    return out


def entry_state(repo, folder, row, ent):
    """why a stored entry cannot be repeated from a clone, or ''"""
    if not repo.exists(ent.get("base", "")):
        return f"base {ent.get('base', '')[:10] or 'none'} is not in this repository (git fetch)"
    if repo.on_origin(ent["base"]) is False:
        return f"base {ent['base'][:10]} is not on origin/main (push it)"
    if ent.get("truncated"):
        return f"the closure was cut at {MAX_FILES} files: the stored files may be incomplete"
    named = {o["blob"]: path for path, o in ent.get("overlay", {}).items() if "file" not in o}     # named by blob id: the version must be in the history of origin/main, as a clone has it
    gone = [b for b in named if b not in repo.history_blobs()]
    if gone:
        return f"the version {gone[0][:10]} of {posixpath.basename(named[gone[0]])} is not in the history of origin/main (git fetch, or push)"
    for path, o in ent.get("overlay", {}).items():
        if "file" not in o:
            continue
        f = os.path.join(folder, o["file"])
        if not os.path.isfile(f):
            return f"stored file {o['file']} is missing"
        if blob_id(open(f, "rb").read()) != o["blob"]:
            return f"stored file {o['file']} does not match its blob id"
    if repo.uncommitted(folder):
        return "recipe.json or code/ is not committed"
    if repo.pushed(folder) is False:
        return "recipe.json and code/ are committed but not pushed"
    if ent.get("unresolved"):
        return f"script {ent['unresolved'][0]} was found in no tree when the recipe was made"
    if ent.get("how") not in EXACT:                            # a run without a snapshot has no overlay: its scripts must be at the base
        miss = missing_scripts(repo, row, ent)
        if miss:
            return f"script {miss[0]} is not at the base commit and not stored (a run without a tag has no overlay)"
    return ""


# ----------------------------------------------------------------------------------------------------------------------------------------------- store
def cmd_store(repo, ids, apply):
    rows = [r for r in ledger_rows() if r.get("command") and any(i == "--all" or r["id"] == i or r["id"].startswith(i + ".") for i in ids)]
    by_folder, skipped, approx, notes_big, kept = {}, [], [], [], []
    for row in rows:
        folder = exp_dir(row["id"])
        if not folder:
            skipped.append((row["id"], "no experiment folder"))
            continue
        if not repo.in_git(folder):
            skipped.append((row["id"], "its experiment folder is not in git (yet)"))
            continue
        base, how, snap = repo.base_of(row)
        if not base:
            skipped.append((row["id"], "no base commit"))
            continue
        e = {"base": base, "how": how, "snapshot": snap, "overlay": {}}
        if how in EXACT:
            res = needed(repo, row, base, snap)
            if res is None:
                skipped.append((row["id"], "base or snapshot commit is not readable here"))
                continue
            e["overlay"] = res[0]
            if res[1]:
                e["truncated"] = True
            if res[2]:
                e["unresolved"] = res[2]
        else:
            approx.append(row["id"])
        by_folder.setdefault(folder, {})[row["id"]] = e
    nfiles = nbytes = nruns = nhist = nsan = npruned = 0
    hist = repo.history_blobs()
    priv = load_private(repo.top)
    all_stored, kinds = set(), {"new": {}, "modified": {}}
    for folder, runs in by_folder.items():
        data = repo.blobs({o for e in runs.values() for o in e["overlay"].values() if o not in hist})
        remap = {}
        for o in list(data):                                             # personal data never goes into git through a copy: the private patterns are applied, the entry names the new blob
            new = apply_private(data[o], priv)
            if new != data[o]:
                remap[o] = blob_id(new)
                data[remap[o]] = new
        for run_id, e in runs.items():
            big = [p for p, o in e["overlay"].items() if o not in hist and (o not in data or len(data[o]) > MAX_BLOB)]
            if big:
                notes_big.append(f"{run_id}: {len(big)} file(s) over {MAX_BLOB} bytes or unreadable, not stored: {', '.join(big[:3])}")
                e["truncated"] = True
            bt = repo.tree(e["base"])[0] if e["overlay"] else {}
            # a version that main's history already holds is named by its blob id only; the others are copied once into the experiment folder
            ov = {}
            for p, o in e["overlay"].items():
                if p in big:
                    continue
                if o in hist:
                    ov[p] = {"blob": o}
                    continue
                ob = remap.get(o, o)
                ov[p] = {"blob": ob, "file": f"code/{ob[:12]}_{posixpath.basename(p)}", **({"sanitised": True} if o in remap else {})}
                kinds["modified" if p in bt else "new"][ob] = len(data[ob])
            e["overlay"] = ov
            nhist += sum(1 for v in ov.values() if "file" not in v)
            nsan += sum(1 for v in ov.values() if v.get("sanitised"))

        def merge(rec):
            for run_id, e in runs.items():
                old = rec["runs"].get(run_id)
                if old and old.get("how") in EXACT and e["how"] not in EXACT:
                    kept.append(run_id)                                  # never trade an exact entry for an approximate one (a clone has no tag)
                else:
                    rec["runs"][run_id] = e
            return rec

        path = os.path.join(folder, "recipe.json")
        if apply:
            os.makedirs(os.path.join(folder, "code"), exist_ok=True)
            for e in runs.values():
                for o in e["overlay"].values():
                    if "file" in o:
                        open(os.path.join(folder, o["file"]), "wb").write(data[o["blob"]])
            with P.locked(path):
                rec = merge(load_recipe(folder))
                P.atomic_write(path, dump_recipe(rec))
                used = {v["file"] for e in rec["runs"].values() for v in e.get("overlay", {}).values() if "file" in v}
                for name in os.listdir(os.path.join(folder, "code")):                    # a copy that no run names any more (an older, wider store) goes; other files of a code/ folder stay
                    if re.match(r"[0-9a-f]{12}_", name) and "code/" + name not in used:
                        os.remove(os.path.join(folder, "code", name))
                        npruned += 1
                if not os.listdir(os.path.join(folder, "code")):
                    os.rmdir(os.path.join(folder, "code"))
        else:
            rec = merge(load_recipe(folder))
        nruns += len(runs)
        nfiles += sum(len(e["overlay"]) for e in runs.values())
        stored = {v["blob"] for e in runs.values() for v in e["overlay"].values() if "file" in v}
        nbytes += sum(len(data[o]) for o in stored)
        all_stored |= stored
    print(f"{'wrote' if apply else 'would write'} {len(by_folder)} recipe.json, {nruns} runs, {nfiles} overlay entries: {nhist} named by blob id (the version is in main's history), "
          f"{nfiles - nhist} copied ({len(all_stored)} distinct versions, {nbytes} bytes in the experiment folders; {nsan} entries changed by the private patterns)")
    print(f"  copied versions by kind (distinct): new files {len(kinds['new'])} ({sum(kinds['new'].values())} bytes), changed tracked files {len(kinds['modified'])} ({sum(kinds['modified'].values())} bytes)")
    print(f"  base by time (no local tag, approximate, no overlay): {len(approx)} runs" + (": " + ", ".join(approx[:6]) + (" ..." if len(approx) > 6 else "") if approx else ""))
    if npruned:
        print(f"  removed {npruned} copied file(s) that no run names any more")
    if kept:
        print(f"  kept the exact entry of {len(kept)} run(s) (no local tag to rebuild it from): {', '.join(kept[:5])}")
    for n in notes_big[:5]:
        print("  " + n)
    for rid, why in skipped[:10]:
        print(f"  skipped {rid}: {why}")
    if len(skipped) > 10:
        print(f"  ... {len(skipped) - 10} more skipped")
    return 0


# ----------------------------------------------------------------------------------------------------------------------------------------------- check
def classify(repo, row):
    folder = exp_dir(row["id"])
    if not row.get("command"):
        return "no command", folder, None
    ent = load_recipe(folder)["runs"].get(row["id"]) if folder else None
    if ent:
        why = entry_state(repo, folder, row, ent)
        if why:
            return "gap: " + why, folder, ent
        return ("ok" if ent["how"] in EXACT else "approximate: base by time, the edits that were not committed that day are unknown"), folder, ent
    if folder is None:
        return "gap: no experiment folder for this run id (nowhere to keep a recipe)", folder, None
    if (repo.prefix + "/" + row["id"]) in repo.tags():
        return "gap: only a local tag (run store)", folder, None
    return "gap: no recipe, no tag", folder, None


def cmd_check(repo, ids, show_list):
    rows = [r for r in ledger_rows() if not ids or any(r["id"] == i or r["id"].startswith(i + ".") for i in ids)]
    from collections import Counter, defaultdict
    cnt, who = Counter(), defaultdict(list)
    for r in rows:
        k = classify(repo, r)[0]
        k = re.sub(r"base [0-9a-f]{10}|stored file \S+|script \S+", lambda m: m.group(0).split()[0] + " ...", k)
        cnt[k] += 1
        who[k].append(r["id"])
    if ids and not rows:
        print(f"no ledger row matches {', '.join(ids)}")
        return 2
    print(f"{len(rows)} runs of {P.path('ledger')}")
    for k, n in cnt.most_common():
        print(f"  {n:5d}  {k}" + (f"   e.g. {', '.join(who[k][:3])}" if not k.startswith("ok") else ""))
        if show_list and not k.startswith("ok"):
            print("         " + " ".join(who[k]))
    bad = sum(n for k, n in cnt.items() if k.startswith(("gap", "no command")))
    return 1 if bad and ids else 0


# ----------------------------------------------------------------------------------------------------------------------------------------------- show / repeat
def card_arm(folder, run_id):
    p = os.path.join(folder or "", "experiment.md")
    if not os.path.isfile(p):
        return {}, {}
    m = re.search(r"```json card\n(.*?)\n```", open(p, encoding="utf-8").read(), re.S)
    try:
        card = json.loads(m.group(1)) if m else {}
    except ValueError:
        return {}, {}
    for name, arm in (card.get("arms") or {}).items():
        if run_id in (arm.get("run_ids") or []):
            return card, dict(arm, name=name)
    return card, {}


def inputs_of(repo, row, rows):
    out = []
    try:
        parts = shlex.split(row.get("command", ""))
    except ValueError:
        parts = row.get("command", "").split()
    for t in parts:
        if t.startswith("-") or not (t.startswith(("~/", "/", "./", "../")) or ("/" in t and t.lower().endswith(DATA_EXT))):
            continue
        if re.search(r"\.(py|sh)$", t):
            continue
        p = os.path.expanduser(t) if t.startswith("~") else os.path.normpath(os.path.join(repo.root, t))
        kind = "exists" if os.path.exists(p) else "MISSING here"
        m = re.search(r"(?:^|/)runs/([\w.-]+)/", t)
        prod = ""
        if m:
            hit = [r["id"] for r in rows if re.search(r"--name[ =]" + re.escape(m.group(1)) + r"\b", r.get("command", ""))]
            prod = f"; made by run {hit[0]}" if hit else "; producer not found in the ledger"
        elif "nuscenes" in t.lower():
            prod = "; raw data: data/NuScenes/README.md"
        elif t.startswith("~/"):
            prod = "; derived cache: the recipe table in the problem's data/README.md and data/rebuild.json"
        out.append(f"{t}  [{kind}{prod}]")
    return out


def results_of(repo, run_id):
    keys = {run_id, run_id.lower().replace(".", "_"), run_id.replace(".", "_")}
    return [p for p in repo.tracked() if any(k in posixpath.basename(p) for k in keys) and p.endswith((".json", ".md", ".csv"))][:8]


def cmd_show(repo, run_id):
    rows = ledger_rows()
    row = next((r for r in rows if r["id"] == run_id), None)
    if not row:
        sys.exit(f"no ledger row {run_id} in {P.path('ledger')}")
    state, folder, ent = classify(repo, row)
    card, arm = card_arm(folder, run_id)
    gaps = [] if state.startswith("ok") else [state]
    print(f"RUN {run_id}   status {row.get('status')}, exit {row.get('exit')}, {row.get('started')} -> {row.get('ended')}, seed {row.get('seed') or '-'}")
    print("\nQUESTION")
    for k in ("hypothesis", "prediction", "rule"):
        print(f"  {k}: {(row.get(k) or '-')[:600]}")
    if arm:
        print(f"  card {card.get('id')} arm {arm['name']}: {(arm.get('recipe') or '')[:600]}")
    print("\nCODE")
    if ent:
        print(f"  base {ent['base']} ({ent['how']}); overlay {len(ent['overlay'])} file(s) stored in {os.path.relpath(folder, repo.top) if folder else '-'}/code")
        for p, o in ent["overlay"].items():
            print(f"    {p}   ({'stored' if 'file' in o else 'version ' + o['blob'][:10] + ' in main history'})")
    else:
        base, how, snap = repo.base_of(row)
        print(f"  no recipe stored; base {base[:10] or 'unknown'} ({how}); the snapshot {snap[:10] or '-'} is read from the tag only if it exists here")
    print("\nCOMMAND   (cwd: the problem folder " + (repo.pf or ".") + ")\n  " + (row.get("command") or "-"))
    ins = inputs_of(repo, row, rows)
    print("\nINPUTS named in the command (heuristic)" + ("" if ins else ": none found"))
    for i in ins:
        print("  " + i)
        gaps += [f"input {i.split()[0]} is missing here"] if "MISSING" in i else []
    envs = [p for p in repo.tracked() if re.search(r"/tools/env/environment[\w.-]*\.(yml|txt)$", p)]
    print("\nENVIRONMENT   " + ("; ".join(envs) if envs else "none recorded"))
    print("\nEXPECTED   result: " + (row.get("result") or "-")[:500] + "\n  decision: " + (row.get("decision") or "-")[:300])
    for p in results_of(repo, run_id):
        print("  file: " + p)
    print(f"\nSTEPS\n  1. python3 autoresearch/tools/recipe.py repeat {run_id}      (worktree of the base commit with the stored files laid over it; prints the command)")
    print("  2. run the printed command (it goes through run.py as <id>.repro with --no-card: a repetition feeds no verdict)")
    print("  3. compare its metrics with EXPECTED within the noise band of the card; report the difference, do not adopt anything")
    print("\nGAPS   " + ("none" if not gaps else "\n  " + "\n  ".join(gaps)))
    return 0


def link_data(repo, dest):
    """the big data stays where it is: linked into the worktree (a folder that is partly tracked gets a link for every child that is not tracked), as provenance.py repro --repeat does"""
    notes = []
    for rel in ("data", posixpath.join(repo.pf, "data") if repo.pf else "data", posixpath.join(repo.pf, "runs") if repo.pf else "runs"):     # runs/: a command reads the output of an earlier run
        src, dst = os.path.join(repo.top, rel), os.path.join(dest, rel)
        if not os.path.isdir(src):
            continue
        try:
            if not os.path.lexists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.symlink(src, dst)
            elif os.path.isdir(dst) and not os.path.islink(dst):
                for child in os.listdir(src):
                    if not os.path.lexists(os.path.join(dst, child)):
                        os.symlink(os.path.join(src, child), os.path.join(dst, child))
        except OSError as e:
            notes.append(f"could not link {rel}: {e}")
    return notes


def cmd_repeat(repo, run_id, dest):
    row = next((r for r in ledger_rows() if r["id"] == run_id), None)
    if not row:
        sys.exit(f"no ledger row {run_id}")
    state, folder, ent = classify(repo, row)
    if not ent:
        base, how, snap = repo.base_of(row)
        if not base:
            sys.exit(f"{run_id}: no base commit")
        ent, folder = {"base": base, "how": how, "overlay": {}}, None
        print(f"WARNING: no stored recipe ({state}); using base {base[:10]} ({how}), no overlay: the code may differ from the run's", file=sys.stderr)
    elif state.startswith("gap") and not (state.startswith("gap: base") and "not on origin" in state):
        sys.exit(f"{run_id}: {state}")
    elif state.startswith("gap"):
        print(f"WARNING: {state}", file=sys.stderr)
    dest = os.path.abspath(dest or os.path.join(repo.top, "temp", "repro", run_id))
    if os.path.isdir(dest):                                                # a folder that is left over is reused only when it is exactly the base
        r = subprocess.run(["git", "-C", dest, "rev-parse", "HEAD"], capture_output=True, text=True)
        if not os.path.exists(os.path.join(dest, ".git")) or r.stdout.strip() != ent["base"]:
            sys.exit(f"{dest} exists and is not a checkout of {ent['base'][:10]} (HEAD {r.stdout.strip()[:10] or 'none'}): remove it with `git worktree remove --force` or pass another --dir")
    else:
        r = subprocess.run(["git", "-C", repo.top, "worktree", "add", "--detach", dest, ent["base"]], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"git worktree add failed: {r.stderr.strip()[:300]}")
    named = repo.blobs([o["blob"] for o in ent["overlay"].values() if "file" not in o])
    for p, o in ent["overlay"].items():
        if outside(p):
            sys.exit(f"{run_id}: the overlay names a path outside the checkout: {p}")
        target = os.path.join(dest, p)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if "file" in o:
            content = open(os.path.join(folder, o["file"]), "rb").read()
        else:
            content = named[o["blob"]]
        with open(target, "wb") as d:
            d.write(content)
        if p.endswith(".sh"):
            os.chmod(target, 0o755)
    notes = link_data(repo, dest)
    if notes:
        print("WARNING: " + "; ".join(notes), file=sys.stderr)
    work = os.path.normpath(os.path.join(dest, repo.pf))
    cmd = printable(row.get("command") or "", dest, repo.top_names(), repo.top)
    print(f"worktree {dest} at {ent['base'][:10]} with {len(ent['overlay'])} stored file(s); remove it later with: git worktree remove --force {dest}")
    print(f"cd {work} && AR_PROJECT=. python3 ../autoresearch/tools/run.py --id {run_id}.repro --hyp \"repetition of {run_id}\" "
          f"--pred \"same code, same data, same seed: the numbers agree within seed noise\" --rule \"by hand: a repetition, no verdict\" --no-card -- {cmd}")
    return 0


# ----------------------------------------------------------------------------------------------------------------------------------------------- selftest
def selftest():
    def sh(cwd, *a, env=None):
        r = subprocess.run(["git", "-C", cwd, "-c", "user.name=t", "-c", "user.email=t@example.invalid", *a], capture_output=True, text=True, env=env)
        assert r.returncode == 0, (a, r.stderr)
        return r.stdout.strip()

    def put(base, rel, text):
        os.makedirs(os.path.dirname(os.path.join(base, rel)), exist_ok=True)
        open(os.path.join(base, rel), "w", newline="\n").write(text)

    def raises(fn, *a):
        try:
            fn(*a)
        except SystemExit:
            return True
        return False

    # the printed command: quotes come back, every absolute prefix of the author's checkout moves into the worktree
    tn = {"p", "data", "bev"}
    assert printable("bash -c bash temp/x.sh && bash tools/env/_py14.sh a.py", "/d", tn) == "bash -c 'bash temp/x.sh && bash tools/env/_py14.sh a.py'"
    assert printable("env OMP=2 nice -n 10 bash -c bash a.sh && bash b.sh", "/d", tn) == "env OMP=2 nice -n 10 bash -c 'bash a.sh && bash b.sh'", "env VAR=x before bash -c"
    assert printable("OMP=2 bash -c env X=1 bash y.sh", "/d", tn) == "OMP=2 bash -c 'env X=1 bash y.sh'"
    assert printable("bash /mnt/c/ML-Experiment/p/x.sh /mnt/d/other-name/data/y /home/ml/data/z /mnt/c/Auto-research/bev/runs/r/model.pt", "/d", tn, "/c/top") == \
        "bash /d/p/x.sh /d/data/y /home/ml/data/z /d/bev/runs/r/model.pt", "a drive path into a folder of the repository moves into the worktree, whatever the checkout was called; a home path stays"
    assert outside("../x.py") and outside("a/../../x.py") and outside("/etc/x") and outside("C:/x.py") and not outside("prob/temp/e1/a.py")
    assert printable("bash -c 'echo hi'", "/d", tn) == "bash -c 'echo hi'" and printable("bash tools/x.sh a b", "/d", tn) == "bash tools/x.sh a b"
    assert refs("a.py", "from pkg import (\n    one,   # c\n    modb as m,\n)\nimport x, y.z\n").count((True, "modb.py")) == 1 and (True, "one.py") in refs("a.py", "from pkg import (\n one,\n two)\n")
    # prose and comments name files without running them; a line that starts a process, or a line of a shell script, does
    assert refs("a.py", "# see temp/e1/prose.py\n\"\"\"docs: temp/e1/doc.py\"\"\"\nsubprocess.run(['bash', 'temp/e1/run.sh'])\n") == [(False, "temp/e1/run.sh")]
    assert refs("a.sh", "# bash temp/e1/comment.sh\nbash temp/e1/real.sh   # but not temp/e1/tail.sh\n") == [(False, "temp/e1/real.sh")]
    bn = {"show.py": ["a/temp/x/show.py", "a/temp/y/show.py"], "lib.py": ["a/training/lib.py"], "go.sh": ["a/temp/e1/go.sh", "a/temp/e2/go.sh"], "pkg/": ["a/pkg/__init__.py"]}
    assert resolve(bn, "show.py", "a/temp/x/main.py") == ["a/temp/x/show.py"], "a bare name: the importer's folder first"
    assert resolve(bn, "show.py", "a/training/m.py") == [] and len(resolve(bn, "show.py", "a/training/m.py", strict=False)) == 2, "a generic name that stays ambiguous is dropped unless the command names it"
    assert resolve(bn, "lib.py", "a/temp/z/m.py") == ["a/training/lib.py"] and resolve(bn, "pkg/", "a/m.py") == ["a/pkg/__init__.py"]
    two = {"losses.py": ["p1/net/loss/losses.py", "p2/training/losses.py"]}
    assert resolve(two, "losses.py", "p1/training/train.py") == ["p1/net/loss/losses.py"] and resolve(two, "losses.py", "p3/a.py") == [], "the importer's problem decides"
    assert resolve(bn, "temp/e2/go.sh", "a/x.sh") == ["a/temp/e2/go.sh"] and resolve(bn, "$BFLC/temp/e1/go.sh", "a/x.sh") == ["a/temp/e1/go.sh"]
    assert resolve(bn, "/mnt/c/Auto-research/a/temp/e1/go.sh", "a/x.sh") == ["a/temp/e1/go.sh"] and resolve(bn, "temp/e3/go.sh", "a/x.sh") == []
    assert blob_id(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"                           # git's own blob id of "hello\n"

    old = os.environ.get("AR_PROJECT")
    with tempfile.TemporaryDirectory() as T:
        repo = os.path.join(T, "repo")
        os.makedirs(repo)
        sh(repo, "init", "-q", "-b", "main")
        put(repo, "root.txt", "r\n")
        sh(repo, "add", "-A")
        sh(repo, "commit", "-q", "-m", "root")
        root = sh(repo, "rev-parse", "HEAD")
        put(repo, "prob/project.json", json.dumps({"tag_prefix": "t", "ledger": "LEDGER.csv"}))
        put(repo, "prob/training/lib.py", "from pkg import (\n    one,\n    modb,\n)\n\n\ndef f():\n    return 1\n")
        put(repo, "prob/training/data/modb.py", "VALUE = 1\n")
        put(repo, "prob/training/other.py", "X = 1\n")
        put(repo, "prob/tools/env/_py.sh", "#!/bin/bash\npython3 \"$@\"\n")
        put(repo, "prob/data/README.md", "tracked part of a data folder\n")
        put(repo, "prob/winner/lib.py", "# a frozen copy of lib.py\n")
        put(repo, "prob/experiments/E1_x/experiment.md", "# E1\n```json card\n" + json.dumps({"id": "E1", "arms": {"a": {"recipe": "the arm", "run_ids": ["E1.a"]}}}) + "\n```\n")
        sh(repo, "add", "-A")
        sh(repo, "commit", "-q", "-m", "A")
        base = sh(repo, "rev-parse", "HEAD")
        # the working tree at the start: dirty edits of two tracked modules (one in a folder called data/), a scratch script that imports them, an unrelated scratch file
        put(repo, "prob/training/lib.py", "from pkg import (\n    one,\n    modb,\n)\n\n\ndef f():\n    return 2   # dirty\n")
        put(repo, "prob/training/data/modb.py", "VALUE = 2   # dirty too\n")
        put(repo, "prob/temp/e1/arm.py", "import lib\nfrom lib import f\nprint(f())   # SECRETNAME decided this\n")
        put(repo, "prob/temp/e1/unrelated.py", "print('x')\n")
        idx = os.path.join(T, "idx")
        env = dict(os.environ, GIT_INDEX_FILE=idx)
        sh(repo, "read-tree", "HEAD", env=env)
        sh(repo, "add", "-f", "-A", env=env)
        tree = sh(repo, "write-tree", env=env)
        snap = sh(repo, "commit-tree", tree, "-p", base, "-m", "snapshot")
        sh(repo, "tag", "-a", "-m", "run", "t/E1.a", snap)
        sh(repo, "checkout", "-q", "--", ".")
        sh(repo, "clean", "-fdq")
        def snapshot(files, delete=()):
            """a snapshot commit on `base`: the base tree plus these files (added by force) minus these paths, made through a temporary index as run.py does"""
            ix = os.path.join(T, "idx_" + str(len(os.listdir(T))))
            e2 = dict(os.environ, GIT_INDEX_FILE=ix)
            sh(repo, "read-tree", base, env=e2)
            for d in delete:
                sh(repo, "rm", "-q", "--cached", d, env=e2)
            for rel, text in files.items():
                put(repo, rel, text)
                sh(repo, "add", "-f", rel, env=e2)
                os.remove(os.path.join(repo, rel))
            return sh(repo, "commit-tree", sh(repo, "write-tree", env=e2), "-p", base, "-m", "snapshot")

        snap_del = snapshot({"prob/temp/e1/armd.py": "import other\n"}, delete=["prob/training/other.py"])
        snap2 = snapshot({"prob/temp/e1/arm.py": "import lib\nprint('second attempt')\n"})
        for tg in ("t/E1.d", "t/E1.r", "t/E1.p"):
            sh(repo, "tag", "-a", "-m", "run", tg, snap)
        put(repo, "prob/LEDGER.csv", "id,started,status,exit,command,seed,hypothesis,prediction,rule,result,decision,commit\n"
                                     f"E1.a,2099-01-01T00:00:00,ok,0,bash tools/env/_py.sh arm.py,0,h,p,r,res,dec,{snap[:10]}\n"
                                     "E1.b,2099-01-01T00:00:00,ok,0,bash tools/env/_py.sh arm.py,0,h,p,r,res,dec,\n"
                                     "E1.c,2099-01-01T00:00:00,ok,0,bash tools/env/_py.sh other.py,0,h,p,r,res,dec,\n"
                                     f"E1.d,2099-01-01T00:00:00,ok,0,bash tools/env/_py.sh nowhere.py,0,h,p,r,res,dec,{snap[:10]}\n"
                                     f"E1.r,2099-01-01T00:00:00,ok,0,bash tools/env/_py.sh arm.py,0,h,p,r,res,dec,{snap2[:10]}\n"
                                     f"E1.p,2099-01-01T00:00:00,ok,0,bash tools/env/_py.sh arm.py,0,h,p,r,res,dec,{base[:10]}\n")
        sh(repo, "add", "-A")
        sh(repo, "commit", "-q", "-m", "ledger")
        put(repo, "prob/training/lib.py", "from pkg import (\n    one,\n    modb,\n)\n\n\ndef f():\n    return 2   # dirty\n")      # the dirty edit of lib.py is committed later: its version is in main's history
        sh(repo, "add", "-A")
        sh(repo, "commit", "-q", "-m", "lib edit committed")
        os.environ["AR_PROJECT"] = os.path.join(repo, "prob")
        try:
            R = Repo()
            assert R.base_of({"id": "E1.a", "started": "2099-01-01T00:00:00"})[:2] == (base, "snapshot parent")
            assert R.base_of(ledger_rows()[4]) == (base, "snapshot parent", snap2), "the ledger's commit (the last attempt) wins over the tag (the first)"
            assert R.base_of(ledger_rows()[5]) == (base, "pinned commit", base), "a commit that is in main's history is a pinned run: its own base"
            cl2, _, _ = closure(R, R.snap_tree(base, snap_del), "bash tools/env/_py.sh armd.py")
            assert "prob/temp/e1/armd.py" in cl2 and "prob/training/other.py" not in cl2, "a file the snapshot deleted is not followed (and does not crash the closure)"
            top = globals()["MAX_FILES"]
            globals()["MAX_FILES"] = 1
            try:
                assert closure(R, R.snap_tree(base, snap), "bash tools/env/_py.sh arm.py")[1] is True, "the cap marks the closure as cut"
            finally:
                globals()["MAX_FILES"] = top
            assert R.base_of({"id": "E1.b", "started": "2000-01-01T00:00:00"})[0] == ""            # before the first commit: no base
            cl, cut, unres = closure(R, R.snap_tree(base, snap), "bash tools/env/_py.sh arm.py")
            assert not cut and cl.keys() >= {"prob/temp/e1/arm.py", "prob/training/lib.py", "prob/tools/env/_py.sh", "prob/training/data/modb.py"}, cl.keys()    # a folder called data/ inside training is code, a name in a multi-line import list is followed
            assert R.snap_tree(base, snap)[0] == R.tree(snap)[0], "base + difference == the listed tree"
            ov, cut, unres = needed(R, {"command": "bash tools/env/_py.sh arm.py"}, base, snap)
            assert set(ov) == {"prob/temp/e1/arm.py", "prob/training/lib.py", "prob/training/data/modb.py"}, ov   # what differs from the base and is executed: not the unrelated scratch, not the unchanged shell wrapper
            assert needed(R, {"command": "x.py"}, "0" * 40, snap) is None, "an unreadable base is None, never an empty overlay"
            assert unres == [] and needed(R, {"command": "bash tools/env/_py.sh nowhere.py"}, base, snap)[2] == ["nowhere.py"], "a script of the command that is found nowhere is recorded"
            assert base in R.history_blobs() and snap not in R.history_blobs(), "the history of main, not the snapshots"
            assert cmd_check(R, [], False) == 0
            put(repo, "prob/experiments/E9_loose/experiment.md", "not committed\n")
            assert R.in_git(os.path.join(repo, "prob/experiments/E1_x")) and not R.in_git(os.path.join(repo, "prob/experiments/E9_loose")), "a folder that git does not track gets no recipe"
            shutil.rmtree(os.path.join(repo, "prob/experiments/E9_loose"))
            for n in ("E2_alpha", "E2_beta_two"):                                                    # two folders share the prefix E2: the card or the slug decides
                put(repo, f"prob/experiments/{n}/experiment.md", "# " + n + ("\nruns: E2.zzz\n" if n == "E2_alpha" else "\n"))
            assert exp_dir("E2.zzz") == os.path.join(repo, "prob/experiments/E2_alpha") and exp_dir("E2.beta_two_s0") == os.path.join(repo, "prob/experiments/E2_beta_two") and exp_dir("E2.q") is None
            shutil.rmtree(os.path.join(repo, "prob/experiments/E2_alpha"))
            shutil.rmtree(os.path.join(repo, "prob/experiments/E2_beta_two"))
            assert classify(R, ledger_rows()[0])[0].startswith("gap: only a local tag")
            cmd_store(R, ["E1.a"], False)
            assert not os.path.exists(os.path.join(repo, "prob/experiments/E1_x/recipe.json")), "a dry run writes nothing"
            put(repo, ".private_patterns", "# a name that must not reach git\nSECRETNAME => the owner\n")
            cmd_store(R, ["E1"], True)
            folder = os.path.join(repo, "prob/experiments/E1_x")
            rec = load_recipe(folder)
            raw = json.load(open(os.path.join(folder, "recipe.json")))
            assert raw["schema"] == 2 and isinstance(raw["runs"]["E1.a"]["overlay"], int) and len(raw["overlays"]) == 3 and raw["runs"]["E1.b"]["overlay"] == raw["runs"]["E1.c"]["overlay"], "equal overlays are kept once"
            assert dump_recipe(expand_recipe(json.loads(dump_recipe(rec)))) == dump_recipe(rec), "dump and load round-trip"
            assert rec["runs"]["E1.a"]["base"] == base and set(rec["runs"]["E1.a"]["overlay"]) == set(ov)
            assert rec["runs"]["E1.b"]["how"].startswith("by time") and rec["runs"]["E1.b"]["overlay"] == {}
            assert rec["runs"]["E1.d"]["unresolved"] == ["nowhere.py"]
            assert rec["runs"]["E1.r"]["snapshot"] == snap2 and "second attempt" in open(os.path.join(folder, rec["runs"]["E1.r"]["overlay"]["prob/temp/e1/arm.py"]["file"])).read()
            assert rec["runs"]["E1.p"]["how"] == "pinned commit" and rec["runs"]["E1.p"]["overlay"] == {}
            ovr = rec["runs"]["E1.a"]["overlay"]
            arm = open(os.path.join(folder, ovr["prob/temp/e1/arm.py"]["file"])).read()
            assert "SECRETNAME" not in arm and "the owner decided this" in arm and ovr["prob/temp/e1/arm.py"]["sanitised"], "the private patterns are applied to a copied file"
            assert blob_id(arm.encode()) == ovr["prob/temp/e1/arm.py"]["blob"], "the entry names the blob of the stored (changed) content"
            assert apply_private(b"a SECRETNAME b", [(re.compile("SECRETNAME"), "X")]) == b"a X b" and apply_private(b"\xff\xfeSECRETNAME", [(re.compile("SECRETNAME"), "X")]) == b"\xff\xfeSECRETNAME"
            assert "file" not in ovr["prob/training/lib.py"] and "file" in ovr["prob/temp/e1/arm.py"] and "file" in ovr["prob/training/data/modb.py"], "a version in main's history is named, the rest is copied"
            assert sorted(os.listdir(os.path.join(folder, "code"))) == sorted({v["file"].split("/")[1] for e in rec["runs"].values() for v in e["overlay"].values() if "file" in v})
            assert classify(R, ledger_rows()[0])[0] == "gap: recipe.json or code/ is not committed"
            sh(repo, "add", "-A")
            sh(repo, "commit", "-q", "-m", "recipe")
            R = Repo()
            assert classify(R, ledger_rows()[0])[0] == "ok"
            assert classify(R, ledger_rows()[1])[0].startswith("gap: script arm.py is not at the base commit"), "a run without a tag has no overlay: its scripts must be at the base"
            assert classify(R, ledger_rows()[2])[0].startswith("approximate: base by time")
            assert classify(R, ledger_rows()[3])[0].startswith("gap: script nowhere.py was found in no tree"), "a script of the command that no tree held is a gap"
            put(repo, "prob/experiments/E1_x/code/zz_new.txt", "x\n")
            assert classify(Repo(), ledger_rows()[0])[0] == "gap: recipe.json or code/ is not committed", "an untracked file in code/ counts"
            os.remove(os.path.join(repo, "prob/experiments/E1_x/code/zz_new.txt"))
            rp0 = os.path.join(folder, "recipe.json")                                                # a named version that origin's history does not hold is a gap
            keep0 = open(rp0).read()
            open(rp0, "w").write(keep0.replace(ovr["prob/training/lib.py"]["blob"], "e" * 40))
            assert "is not in the history of origin/main" in classify(Repo(), ledger_rows()[0])[0]
            open(rp0, "w").write(keep0)
            sh(repo, "update-ref", "refs/remotes/origin/main", root)                                 # origin/main behind the base: a clone would not have it
            assert "is not on origin/main" in classify(Repo(), ledger_rows()[0])[0]
            sh(repo, "update-ref", "refs/remotes/origin/main", sh(repo, "rev-parse", "HEAD~1"))        # the commit with recipe.json is not pushed yet
            assert classify(Repo(), ledger_rows()[0])[0] == "gap: recipe.json and code/ are committed but not pushed"
            sh(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
            sh(repo, "tag", "-d", "t/E1.a")                                                          # a clone without the tag, and without the snapshot's name in the ledger
            lp = os.path.join(repo, "prob/LEDGER.csv")
            text = open(lp).read()
            open(lp, "w").write(text.replace(f"arm.py,0,h,p,r,res,dec,{snap[:10]}\n", "arm.py,0,h,p,r,res,dec,\n", 1))
            R2 = Repo()
            assert classify(R2, ledger_rows()[0])[0] == "ok", "the recipe replaces the tag"
            put(repo, "prob/experiments/E1_x/code/ffffffffffff_stale.py", "old\n")                  # a copy that no run names, and a file that is not a copy
            put(repo, "prob/experiments/E1_x/code/keep_me.sh", "mine\n")
            cmd_store(R2, ["E1"], True)                                                              # store again with no tag: the exact entry must stay
            assert load_recipe(folder)["runs"]["E1.a"]["how"] == "snapshot parent"
            assert not os.path.exists(os.path.join(folder, "code/ffffffffffff_stale.py")) and os.path.exists(os.path.join(folder, "code/keep_me.sh")), "an unreferenced copy is pruned, other files are not"
            os.remove(os.path.join(folder, "code/keep_me.sh"))
            dest = os.path.join(T, "wt")
            put(repo, "prob/data/big.bin", "not in git\n")                                           # the real data folder holds an untracked child
            os.makedirs(os.path.join(T, "wrong"))
            assert raises(cmd_repeat, R2, "E1.a", os.path.join(T, "wrong")), "an existing folder that is not the base is refused"
            other = os.path.join(T, "wrong2")
            sh(repo, "worktree", "add", "-q", "--detach", other, root)
            assert raises(cmd_repeat, R2, "E1.a", other), "a worktree of another commit is refused, not mixed with the overlay"
            subprocess.run(["git", "-C", repo, "worktree", "remove", "--force", other], capture_output=True)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                assert cmd_repeat(R2, "E1.a", dest) == 0
            printed = buf.getvalue().rstrip()
            assert "--no-card" in printed and "--id E1.a.repro" in printed and f"cd {dest}/prob && AR_PROJECT=. python3 ../autoresearch/tools/run.py" in printed \
                and printed.endswith("-- bash tools/env/_py.sh arm.py"), printed
            assert open(os.path.join(dest, "prob/training/lib.py")).read().endswith("# dirty\n") and os.path.isfile(os.path.join(dest, "prob/temp/e1/arm.py"))
            assert not os.path.exists(os.path.join(dest, "prob/temp/e1/unrelated.py"))
            assert os.path.islink(os.path.join(dest, "prob/data/big.bin")) and os.path.isfile(os.path.join(dest, "prob/data/README.md")), "an untracked child of a partly tracked data folder is linked"
            assert cmd_repeat(R2, "E1.a", dest) == 0, "the worktree is reused when it is the base"
            subprocess.run(["git", "-C", repo, "worktree", "remove", "--force", dest], capture_output=True)
            assert cmd_show(R2, "E1.a") == 0
            rp = os.path.join(folder, "recipe.json")                                                 # a cut closure is a gap
            keep = open(rp).read()
            cut = json.loads(keep)
            cut["runs"]["E1.a"]["truncated"] = True
            open(rp, "w").write(json.dumps(cut))
            assert classify(R2, ledger_rows()[0])[0].startswith("gap: the closure was cut")
            open(rp, "w").write(keep)
            assert classify(R2, ledger_rows()[0])[0] == "ok"
            some = next(v["file"] for v in rec["runs"]["E1.a"]["overlay"].values() if "file" in v)    # a changed stored file is a gap, not a silent wrong code
            open(os.path.join(folder, some), "ab").write(b"#")
            assert classify(R2, ledger_rows()[0])[0].startswith("gap: stored file")
            assert cmd_check(R2, ["E1.a"], False) == 1 and cmd_check(R2, ["NOPE"], False) == 2, "a selected run with a gap, and an id that matches nothing, are not exit 0"
            os.remove(os.path.join(folder, some))
            assert classify(R2, ledger_rows()[0])[0].startswith("gap: stored file") and cmd_check(R2, ["E1.a"], False) == 1
            # the by-time base follows the first parent: the merge commit is after the window, its side parent is inside it and newer than the first parent: the first parent wins
            sh(repo, "add", "-A")
            sh(repo, "commit", "-q", "-m", "recipe 2")
            first = sh(repo, "rev-parse", "HEAD")
            sh(repo, "checkout", "-q", "-b", "side", root)
            put(repo, "side.txt", "s\n")
            sh(repo, "add", "-A")
            sh(repo, "commit", "-q", "-m", "side", env=dict(os.environ, GIT_COMMITTER_DATE="2099-02-01T00:00:00"))
            sh(repo, "checkout", "-q", "main")
            sh(repo, "merge", "-q", "--no-ff", "-m", "merge", "side", env=dict(os.environ, GIT_COMMITTER_DATE="2099-05-01T00:00:00"))
            sh(repo, "update-ref", "-d", "refs/remotes/origin/main")
            assert Repo().base_of({"id": "E1.z", "started": "2099-03-01T00:00:00"})[0] == first
        finally:
            if old is None:
                os.environ.pop("AR_PROJECT", None)
            else:
                os.environ["AR_PROJECT"] = old
    print("recipe selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("show")
    s.add_argument("id")
    s = sub.add_parser("repeat")
    s.add_argument("id")
    s.add_argument("--dir")
    s = sub.add_parser("check")
    s.add_argument("--id", action="append", default=[])
    s.add_argument("--list", action="store_true")
    s = sub.add_parser("store")
    s.add_argument("ids", nargs="*")
    s.add_argument("--all", action="store_true")
    s.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.cmd:
        ap.print_help()
        return 2
    repo = Repo()
    if a.cmd == "show":
        return cmd_show(repo, a.id)
    if a.cmd == "repeat":
        return cmd_repeat(repo, a.id, a.dir)
    if a.cmd == "check":
        return cmd_check(repo, a.id, a.list)
    if not a.ids and not a.all:
        sys.exit("store: name an experiment id (E346), a run id (E346.g0) or --all")
    return cmd_store(repo, ["--all"] if a.all else a.ids, a.apply)


if __name__ == "__main__":
    sys.exit(main())
