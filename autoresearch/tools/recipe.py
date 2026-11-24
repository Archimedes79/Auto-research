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
    (the script, what it imports or names, to a depth of six) that differ from the base; a version that main's history holds (most dirty edits were committed
    later) is named by its blob id, the rest is copied once under the experiment folder as code/<blob id>_<name>
  - everything else is derived at read time from what git holds anyway (ledger row, card, tracked result files, rebuild recipes), never copied
The record is <experiment folder>/recipe.json: {"schema": 1, "runs": {<run id>: {"base", "how", "snapshot", "overlay": {<repo path>: {"blob", "file"?}}, "truncated"?}}}.
A run without a local tag gets its base from the ledger's start time (the last first-parent commit of origin/main before it): marked "by time", exact in about 96 % of the
tagged runs (the others waited in the queue after the snapshot); its overlay is empty, so `check` also tests that the scripts of its command exist at that base.
`store` never replaces an exact entry (tag parent) by an approximate one; it needs the tag, so run it on the machine that ran the experiment, then commit recipe.json and code/.
Not covered: the data and checkpoints a command reads (`show` lists the paths it names and where they come from), the environment's package versions (the env files are named),
and files that are not code (configs, registry json) that differed from the base: `repeat` runs those at the base's version.
"""
import argparse
import csv
import hashlib
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

MAX_FILES, MAX_DEPTH, MAX_BLOB = 600, 6, 1_000_000
NAMED = re.compile(r"[\w./~${}-]+\.(?:py|sh)\b")
IMPORT = re.compile(r"^[ \t]*(?:from[ \t]+([\w.]+)[ \t]+import[ \t]+(\([^)]*\)|[^\n]+)|import[ \t]+([\w., \t]+))", re.M)
DATA_EXT = (".pt", ".pth", ".ckpt", ".pkl", ".npy", ".npz", ".json", ".csv", ".yml", ".yaml")
LEGACY_TOP = "/mnt/c/ML-Experiment"          # the folder the first runs were started from (their ledger commands name it)


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
        self.skip = re.compile(rf"^(?:{pre})?(?:literature|data|results|runs|models|reports/source|_to_delete|\.claude)/|^data/|(?:^|/)_to_delete/|(?:^|/)experiments/[^/]+/code/")
        self._trees, self._tags, self._tracked, self._refs, self._dirty, self._anc = {}, None, None, {}, {}, {}

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
        return bool(commit) and gitfast.run(self.top, "cat-file", "-e", commit + "^{commit}").returncode == 0

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

    def base_of(self, row):
        """(base commit, how, the snapshot commit): the parent of the run's tag, else the last first-parent commit of origin/main (HEAD) before the ledger's start time"""
        tag = self.prefix + "/" + row["id"]
        if tag in self.tags():
            c = self.g("rev-list", "--parents", "-n", "1", tag).split()
            if len(c) >= 2:
                return c[1], "tag parent", c[0]
        started = (row.get("started") or "").replace("T", " ")
        ref = "origin/main" if self.exists("origin/main") else "HEAD"
        c = self.g("rev-list", "-n", "1", "--first-parent", f"--before={started}", ref).strip() if started else ""
        return c, "by time (approximate)" if c else "unknown", (row.get("commit") or "").strip()

    def history_blobs(self):
        """the ids of every object in the history of origin/main (HEAD without a remote): a file version that is in it needs no copy"""
        ref = "origin/main" if self.exists("origin/main") else "HEAD"
        return {l.split()[0] for l in self.g("rev-list", "--objects", ref).split("\n") if l.strip()}

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


def refs(path, text):
    """names of the code files a file can run: its imports (module, dotted parts and imported names, also in a parenthesised list over several lines) and every .py / .sh it names"""
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
                        out += [part + ".py", part + "/"]
    out += [posixpath.basename(m.group(0)) for m in NAMED.finditer(text)]
    return out


def pick(byname, name, importer):
    cands = byname.get(name, [])
    pool = [c for c in cands if "/temp/" not in "/" + c] or cands
    return [c for c in pool if posixpath.dirname(c) == posixpath.dirname(importer)] or pool


def closure(repo, snap, cmd):
    """(the code files the command can execute in a tree {path: blob id}, truncated)"""
    paths, byname = snap
    seen, level = {}, []
    for tok in command_scripts(cmd):
        t = posixpath.normpath(tok.replace("\\", "/").split("ML-Experiment/")[-1])
        hits = pick(byname, posixpath.basename(t), repo.rel("x"))
        level += [h for h in hits if h.endswith("/" + t) or h == t] or hits
    truncated = False
    for depth in range(MAX_DEPTH + 1):
        level = [p for p in dict.fromkeys(level) if p not in seen]
        if not level:
            break
        if len(seen) + len(level) > MAX_FILES or depth == MAX_DEPTH:
            truncated = True
            level = level[:max(0, MAX_FILES - len(seen))]
        seen.update({p: paths[p] for p in level})
        rf = repo.refs_of([(p, paths[p]) for p in level])
        level = [c for p in level for r in rf[paths[p]] for c in pick(byname, r, p)]
        if truncated:
            break
    return seen, truncated


def needed(repo, row, base, snapshot):
    """({path: blob id} of the files the command executes that differ from the base commit, truncated), or None when the base or the snapshot cannot be read here"""
    if not (repo.exists(base) and repo.exists(snapshot)):
        return None
    bt = repo.tree(base)[0]
    cl, truncated = closure(repo, repo.snap_tree(base, snapshot), row.get("command", ""))
    return {p: o for p, o in cl.items() if bt.get(p) != o}, truncated


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


def printable(cmd, dest, tops):
    """the ledger's command as a command line again: every absolute prefix of the author's checkout points into `dest`, and `bash -c <words>` (run.py joined the arguments with
    spaces and lost the quotes) gets its one argument back"""
    for t in dict.fromkeys(list(tops) + [LEGACY_TOP]):
        cmd = cmd.replace(t.rstrip("/") + "/", dest.rstrip("/") + "/")
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return cmd
    if len(parts) > 3 and parts[0] in ("bash", "sh") and parts[1] == "-c":
        return f"{parts[0]} -c " + shlex.quote(cmd.split(" -c ", 1)[1])
    return cmd


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
    if ent.get("truncated"):
        return f"the closure was cut at {MAX_FILES} files: the stored files may be incomplete"
    for path, o in ent.get("overlay", {}).items():
        if "file" not in o:                                  # named by blob id: the version must be in this repository's history
            if gitfast.run(repo.top, "cat-file", "-e", o["blob"]).returncode:
                return f"the version {o['blob'][:10]} of {posixpath.basename(path)} is not in this repository (git fetch)"
            continue
        f = os.path.join(folder, o["file"])
        if not os.path.isfile(f):
            return f"stored file {o['file']} is missing"
        if blob_id(open(f, "rb").read()) != o["blob"]:
            return f"stored file {o['file']} does not match its blob id"
    if repo.on_origin(ent["base"]) is False:
        return f"base {ent['base'][:10]} is not on origin/main (push it)"
    if repo.uncommitted(folder):
        return "recipe.json or code/ is not committed"
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
        base, how, snap = repo.base_of(row)
        if not base:
            skipped.append((row["id"], "no base commit"))
            continue
        e = {"base": base, "how": how, "snapshot": snap, "overlay": {}}
        if how == "tag parent":
            res = needed(repo, row, base, snap)
            if res is None:
                skipped.append((row["id"], "base or snapshot commit is not readable here"))
                continue
            e["overlay"] = res[0]
            if res[1]:
                e["truncated"] = True
        else:
            approx.append(row["id"])
        by_folder.setdefault(folder, {})[row["id"]] = e
    nfiles = nbytes = nruns = nhist = nsan = 0
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
                if old and old.get("how") == "tag parent" and e["how"] != "tag parent":
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
        return ("ok" if ent["how"] == "tag parent" else "ok, base by time (approximate)"), folder, ent
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
    for rel in ("data", posixpath.join(repo.pf, "data") if repo.pf else "data"):
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
    cmd = printable(row.get("command") or "", dest, [repo.top])
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
    assert printable("bash -c bash temp/x.sh && bash tools/env/_py14.sh a.py", "/d", ["/c/top"]) == "bash -c 'bash temp/x.sh && bash tools/env/_py14.sh a.py'"
    assert printable("bash /mnt/c/ML-Experiment/p/x.sh /c/top/data/y", "/d", ["/c/top"]) == "bash /d/p/x.sh /d/data/y"
    assert printable("bash -c 'echo hi'", "/d", []) == "bash -c 'echo hi'"
    assert printable("bash tools/x.sh a b", "/d", []) == "bash tools/x.sh a b"
    assert refs("a.py", "from pkg import (\n    one,   # c\n    modb as m,\n)\nimport x, y.z\n").count("modb.py") == 1 and "one.py" in refs("a.py", "from pkg import (\n one,\n two)\n")
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
        put(repo, "prob/LEDGER.csv", "id,started,status,exit,command,seed,hypothesis,prediction,rule,result,decision,commit\n"
                                     f"E1.a,2099-01-01T00:00:00,ok,0,bash tools/env/_py.sh arm.py,0,h,p,r,res,dec,{snap[:10]}\n"
                                     "E1.b,2099-01-01T00:00:00,ok,0,bash tools/env/_py.sh arm.py,0,h,p,r,res,dec,\n"
                                     "E1.c,2099-01-01T00:00:00,ok,0,bash tools/env/_py.sh other.py,0,h,p,r,res,dec,\n")
        sh(repo, "add", "-A")
        sh(repo, "commit", "-q", "-m", "ledger")
        put(repo, "prob/training/lib.py", "from pkg import (\n    one,\n    modb,\n)\n\n\ndef f():\n    return 2   # dirty\n")      # the dirty edit of lib.py is committed later: its version is in main's history
        sh(repo, "add", "-A")
        sh(repo, "commit", "-q", "-m", "lib edit committed")
        os.environ["AR_PROJECT"] = os.path.join(repo, "prob")
        try:
            R = Repo()
            assert R.base_of({"id": "E1.a", "started": "2099-01-01T00:00:00"})[:2] == (base, "tag parent")
            assert R.base_of({"id": "E1.b", "started": "2000-01-01T00:00:00"})[0] == ""            # before the first commit: no base
            cl, cut = closure(R, R.snap_tree(base, snap), "bash tools/env/_py.sh arm.py")
            assert not cut and cl.keys() >= {"prob/temp/e1/arm.py", "prob/training/lib.py", "prob/tools/env/_py.sh", "prob/training/data/modb.py"}, cl.keys()    # a folder called data/ inside training is code, a name in a multi-line import list is followed
            assert R.snap_tree(base, snap)[0] == R.tree(snap)[0], "base + difference == the listed tree"
            ov, cut = needed(R, {"command": "bash tools/env/_py.sh arm.py"}, base, snap)
            assert set(ov) == {"prob/temp/e1/arm.py", "prob/training/lib.py", "prob/training/data/modb.py"}, ov   # what differs from the base and is executed: not the unrelated scratch, not the unchanged shell wrapper
            assert needed(R, {"command": "x.py"}, "0" * 40, snap) is None, "an unreadable base is None, never an empty overlay"
            assert cmd_check(R, [], False) == 0
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
            assert raw["schema"] == 2 and isinstance(raw["runs"]["E1.a"]["overlay"], int) and len(raw["overlays"]) == 2 and raw["runs"]["E1.b"]["overlay"] == raw["runs"]["E1.c"]["overlay"], "equal overlays are kept once"
            assert dump_recipe(expand_recipe(json.loads(dump_recipe(rec)))) == dump_recipe(rec), "dump and load round-trip"
            assert rec["runs"]["E1.a"]["base"] == base and set(rec["runs"]["E1.a"]["overlay"]) == set(ov)
            assert rec["runs"]["E1.b"]["how"].startswith("by time") and rec["runs"]["E1.b"]["overlay"] == {}
            ovr = rec["runs"]["E1.a"]["overlay"]
            arm = open(os.path.join(folder, ovr["prob/temp/e1/arm.py"]["file"])).read()
            assert "SECRETNAME" not in arm and "the owner decided this" in arm and ovr["prob/temp/e1/arm.py"]["sanitised"], "the private patterns are applied to a copied file"
            assert blob_id(arm.encode()) == ovr["prob/temp/e1/arm.py"]["blob"], "the entry names the blob of the stored (changed) content"
            assert apply_private(b"a SECRETNAME b", [(re.compile("SECRETNAME"), "X")]) == b"a X b" and apply_private(b"\xff\xfeSECRETNAME", [(re.compile("SECRETNAME"), "X")]) == b"\xff\xfeSECRETNAME"
            assert "file" not in ovr["prob/training/lib.py"] and "file" in ovr["prob/temp/e1/arm.py"] and "file" in ovr["prob/training/data/modb.py"], "a version in main's history is named, the rest is copied"
            assert sorted(os.listdir(os.path.join(folder, "code"))) == sorted(v["file"].split("/")[1] for v in ovr.values() if "file" in v)
            assert classify(R, ledger_rows()[0])[0] == "gap: recipe.json or code/ is not committed"
            sh(repo, "add", "-A")
            sh(repo, "commit", "-q", "-m", "recipe")
            R = Repo()
            assert classify(R, ledger_rows()[0])[0] == "ok"
            assert classify(R, ledger_rows()[1])[0].startswith("gap: script arm.py is not at the base commit"), "a run without a tag has no overlay: its scripts must be at the base"
            assert classify(R, ledger_rows()[2])[0] == "ok, base by time (approximate)"
            sh(repo, "update-ref", "refs/remotes/origin/main", root)                                 # origin/main behind the base: a clone would not have it
            assert "is not on origin/main" in classify(Repo(), ledger_rows()[0])[0]
            sh(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
            sh(repo, "tag", "-d", "t/E1.a")                                                          # a clone without the tag
            R2 = Repo()
            assert classify(R2, ledger_rows()[0])[0] == "ok", "the recipe replaces the tag"
            cmd_store(R2, ["E1"], True)                                                              # store again with no tag: the exact entry must stay
            assert load_recipe(folder)["runs"]["E1.a"]["how"] == "tag parent"
            dest = os.path.join(T, "wt")
            put(repo, "prob/data/big.bin", "not in git\n")                                           # the real data folder holds an untracked child
            os.makedirs(os.path.join(T, "wrong"))
            assert raises(cmd_repeat, R2, "E1.a", os.path.join(T, "wrong")), "an existing folder that is not the base is refused"
            other = os.path.join(T, "wrong2")
            sh(repo, "worktree", "add", "-q", "--detach", other, root)
            assert raises(cmd_repeat, R2, "E1.a", other), "a worktree of another commit is refused, not mixed with the overlay"
            subprocess.run(["git", "-C", repo, "worktree", "remove", "--force", other], capture_output=True)
            assert cmd_repeat(R2, "E1.a", dest) == 0
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
            assert cmd_check(R2, ["E1.a"], False) == 1
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
