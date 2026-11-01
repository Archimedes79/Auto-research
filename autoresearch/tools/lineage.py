"""Lineage of a network: network -> training run -> code -> data -> evaluations, every link VERIFIED, not only listed (design:
autoresearch/reproducibility.md; Mirko asked for a general traceability approach for code, data and network). Read-only except `record`. Standard library only.

    python3 lineage.py show <name | run id | slotN | checkpoint path | run folder name> [--json]    the chain of one network, one line per check
    python3 lineage.py verify [NAME ...] [--all] [--strict]     every network of keep3.json (its best three), with --all also the registry's champion /
                                                                control, plus the names given; exit 1 on any MISMATCH, else 0
    python3 lineage.py record <run id> [--ckpt PATH] [--parent PATH] [--parent-kind init|resume]    write <run records>/<id>/lineage.json
    python3 lineage.py --selftest

Every check prints OK, MISSING (not recorded, an old run, or not found here: reported, never guessed) or MISMATCH (recorded and different); INFO lines carry
facts that are no check. `show` and `verify` run the same checks (verify over many networks), so the exit code of `show` is the verdict of that chain.
What is checked, in the order of the chain:
  1 checkpoint: the file exists; its size and hash equal each recorded one (keep3.json: sha256 of the whole file; lineage.json: the same; a card's
    provenance.subject.ckpt_hash: provenance.file_hash, 12 hex, a file over 64 MB by its first and last 4 MB plus its size, marked p). The hash kind used
    is printed with the line. No recorded hash anywhere = MISSING (the current hash is printed).
  2 run record <run records>/<id>/run.json of the training run: found by id (a dot read as _) or by a command that names <network>.json; the other records
    that mention the network are listed. The tag of the record resolves in git to its commit (a resumed run whose git_note says "tag existed" and names the
    tag's commit is the recorded case: attempt 1 holds the tag); the commit exists and is an ancestor of HEAD or held by a tag (local git, no network;
    whether the tag is pushed is check_git.py's list); the snapshot folder of the run (code/SNAPSHOT.txt) names the same commit; code_drift is listed
    (INFO; --strict: MISMATCH). The evaluation records <id>.* get the tag check too.
  3 parent: the checkpoint this one was started from (lineage.json, run.json, or the run's config.json init_from): exists, hash equals the recorded one.
  4 data: the GT signature of the run (gt.json, or its provenance) is a row of autogt/versions.csv, its git tag autogt/vN exists, a version named by a
    provenance block is the version of that signature, the GT store (when present here) names the same version.
  5 evaluations: the cards that name the network (<cards>/<name>/*.json and <cards>/<name>/eval/*.json for the network's names): commit in git, a
    provenance block, its GT signature as in 4, its raw-data manifest hash equal to the current datasets/NuScenes/manifest.json; then provenance.compare of
    every two cards with a provenance block (INFO: comparable / differ / not shown comparable).

Recording for FUTURE networks (nothing in run.py or train.py is changed here): `record` writes <run records>/<id>/lineage.json, a file of its own
({"schema": 1, "run", "made", "checkpoint": {path, bytes, sha256}, "parent": {kind, path, bytes, sha256}}); run.py owns run.json and rewrites it at the end of a
run, so a second writer would be overwritten. Paths are relative to the problem folder. The function is `record(run_id_or_run_dir, ckpt=None, parent=None,
parent_kind="init")`. PROPOSED (the owners of run.py / train.py decide): at the end of a successful training, train.py calls
`lineage.record(os.environ["AR_RUN_ID"], ckpt=<final checkpoint>, parent=cfg.init_from or None)` (run.py exports AR_RUN_ID to the command; lineage.py is
standard library only, so the trainer's python can import it from autoresearch/tools), or run.py does it after exit 0 from a `ckpt` line the trainer prints;
a resume of the same run needs no parent (same run id). Until then `python3 autoresearch/tools/lineage.py record <id> --ckpt ... [--parent ...]` by hand.
"""
import argparse
import contextlib
import csv
import datetime as dt
import glob
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import project as P  # noqa: E402
import provenance as PV  # noqa: E402

OK, MISSING, MISMATCH, INFO = "OK", "MISSING", "MISMATCH", "INFO"
UNKNOWN = PV.UNKNOWN
RANK = {OK: 0, INFO: 0, MISSING: 1, MISMATCH: 2}
SHA = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{7,40}")      # run.py writes words too ("skipped", "git-error") when it could not snapshot


# ---------------------------------------------------------------- small helpers
def jload(path):
    """a json file as a dict, None when it is missing, torn or not a dict"""
    try:
        with open(path, encoding="utf-8-sig") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def jread(path):
    """(dict or None, problem or None): a file that is absent is fine (None, None); one that exists but is torn or not a json object is a problem"""
    if not os.path.isfile(path):
        return None, None
    d = jload(path)
    return (d, None) if d is not None else (None, f"{path} exists but is not readable json")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hash_as(path, recorded):
    """(hash of the file in the kind of the recorded one, kind text): 64 hex = sha256 of the whole file, else provenance.file_hash"""
    if SHA.fullmatch(recorded or ""):
        return sha256_of(path), "sha256, whole file"
    return PV.file_hash(path), "file_hash, 12 hex, partial over 64 MB"


def worst(statuses):
    return max(statuses, key=RANK.get, default=OK)


class Ctx:
    """the problem and the repository it lies in (read once)"""

    def __init__(self):
        self.root = P.root()
        self.repo = PV.repo_root(self.root)
        self.cfg = P.cfg()
        self.problems = []                         # records that exist but cannot be read: a MISMATCH of the whole run, never "nothing to verify"

        def read(path):
            d, err = jread(path)
            if err:
                self.problems.append(err)
            return d
        self.keep = read(os.path.join(os.path.dirname(P.path("ledger")), "keep3.json"))
        regf = (self.keep or {}).get("registry", "models/registry.json")
        reg = (read(os.path.join(self.root, regf)) if regf else None) or {}
        self.reg = {k: v for k, v in reg.items() if isinstance(v, dict) and not k.startswith("_")}
        self.rr = P.path("run_records")
        self.cards = P.path("cards") if "cards" in self.cfg else os.path.join(self.root, "runs")
        self.recs = {}
        if os.path.isdir(self.rr):
            for i in sorted(os.listdir(self.rr)):
                r = read(os.path.join(self.rr, i, "run.json"))
                if r:
                    self.recs[i] = r
        self.versions = {}
        try:
            with open(os.path.join(self.repo, "autogt", "versions.csv"), newline="", encoding="utf-8") as f:
                self.versions = {r["signature"]: r for r in csv.DictReader(f)}
        except OSError:
            pass
        self.manifest = os.path.join(self.repo, "datasets", "NuScenes", "manifest.json")

    def rel(self, f):
        return os.path.relpath(f, self.root).replace("\\", "/")

    def git(self, *a):
        out = PV.git(self.repo, *a)
        return None if out is None else out.strip()


# ---------------------------------------------------------------- finding the network
def find_ckpt(c, key):
    pats = ([c.keep["ckpt"]] if c.keep and c.keep.get("ckpt") else []) + ["runs/{run}/model.pt", "runs/{run}/ckpt_*.pt"]
    for p in pats:
        hits = sorted(glob.glob(os.path.join(c.root, p.format(run=key))))
        if hits:
            return c.rel(hits[-1])         # the highest step of a family of ckpt_<step>.pt
    return None


def make_net(c, key, ckpt=None, names=(), slot=None, rec_id=None):
    ckpt = ckpt or (find_ckpt(c, key) if key else None)
    nm = {n for n in names if n} | ({key} if key else set()) | {n for n, v in c.reg.items() if key and v.get("run") == key}
    return dict(key=key, ckpt=ckpt, names=sorted(nm), slot=slot, rec_id=rec_id)


def keep_nets(c):
    out = []
    for i, e in enumerate((c.keep or {}).get("entries", []), 1):
        for ck in e.get("checkpoints", []) or [{}]:
            key = os.path.basename(os.path.dirname(ck["path"])) if ck.get("path") else e.get("key")
            out.append(make_net(c, key, ck.get("path"), [e["name"]], slot=dict(e, n=i)))
    return out


def resolve(c, target):
    for f in (target, os.path.join(c.root, target)):
        if os.path.isfile(f):
            return [make_net(c, os.path.basename(os.path.dirname(os.path.abspath(f))), c.rel(os.path.abspath(f)))]
    nets = keep_nets(c)
    m = re.fullmatch(r"slot(\d)", target)
    if m:
        hit = [n for n in nets if n["slot"]["n"] == int(m.group(1))]
    else:
        hit = [n for n in nets if target in n["names"] or target == n["key"]]
    if hit:
        return hit
    if target in c.reg and c.reg[target].get("run"):
        return [make_net(c, c.reg[target]["run"], names=[target])]
    if target in c.recs:
        lin = jload(os.path.join(c.rr, target, "lineage.json")) or {}
        ck = (lin.get("checkpoint") or {}).get("path")
        key = os.path.basename(os.path.dirname(ck)) if ck else (target.replace(".", "_") if find_ckpt(c, target.replace(".", "_")) else None)
        return [make_net(c, key, ck, rec_id=target)]
    if find_ckpt(c, target):
        return [make_net(c, target)]
    sys.exit(f"{target}: not a checkpoint file, a keep3 name or slot, a registry name, a run id or a run folder of this problem ({c.root})")


# ---------------------------------------------------------------- the checks (each returns lines (status, link, text))
def check_ckpt(c, net, lin, cards):
    out = []
    f = os.path.join(c.root, net["ckpt"]) if net["ckpt"] else None
    if not f or not os.path.isfile(f):
        return [(MISSING, "checkpoint", f"{net['ckpt'] or 'no checkpoint file found for ' + str(net['key'])} does not exist here (archived, slimmed, or a clone: checkpoints are not in git)")]
    size = os.path.getsize(f)
    out.append((OK, "checkpoint", f"{net['ckpt']} exists, {size} bytes"))
    sources = []                                   # (label, bytes, hash)
    if c.keep:
        for e in c.keep.get("entries", []):
            sources += [("keep3.json", k.get("bytes"), k.get("sha256")) for k in e.get("checkpoints", []) if k.get("path") == net["ckpt"]]
    ck = (lin or {}).get("checkpoint") or {}
    if ck.get("path") == net["ckpt"]:
        sources.append(("lineage.json", ck.get("bytes"), ck.get("sha256")))
    for cp, card in cards:
        s = (card.get("provenance") or {}).get("subject") or {}
        if s.get("ckpt") == os.path.basename(f) and s.get("ckpt_hash") not in (None, UNKNOWN):
            sources.append((f"provenance of {os.path.basename(os.path.dirname(cp))}/{os.path.basename(cp)}", None, s["ckpt_hash"]))
    if not sources:
        h, kind = PV.file_hash(f), "provenance.file_hash"
        return out + [(MISSING, "checkpoint hash", f"no recorded size or hash (keep3.json, lineage.json, card provenance); current {kind} {h}")]
    for label, nbytes, rec in sources:
        if nbytes is not None and nbytes != size:
            out.append((MISMATCH, "checkpoint hash", f"size {size} != {nbytes} recorded in {label} (not hashed)"))
            continue
        h, kind = hash_as(f, rec)
        out.append((OK if h == rec else MISMATCH, "checkpoint hash", f"{'equals' if h == rec else 'DIFFERS from'} {label} ({kind}): {h[:16]} vs {str(rec)[:16]}"))
    return out


def find_train_record(c, net):
    """(record id or None, other ids that mention the network): by id (a dot read as _) or by a command naming <network>.json; a mention alone is no training"""
    key = net["key"]
    if net["rec_id"]:
        return net["rec_id"], []
    if not key:
        return None, []
    hit = [i for i in c.recs if i == key or i.replace(".", "_") == key]
    hit = hit or [i for i, r in c.recs.items() if re.search(r"(?<![\w.-])" + re.escape(key) + r"\.json\b", str(r.get("command", "")))]
    mention = [i for i, r in c.recs.items() if key in str(r.get("command", "")) and i not in hit]
    return (hit[0] if hit else None), mention + hit[1:]


def check_run_state(c, rid, rec, rdir, kept):
    """the record must say the run finished OK: status ok + exit 0. A 'running' record whose folder has the .done flag is stale (MISMATCH); a failed or killed
    run whose checkpoint is kept as a network is a MISMATCH too; otherwise a run that did not finish OK is MISSING with the reason"""
    st, ex = rec.get("status"), rec.get("exit")
    done = [x for x in (os.path.join(rdir or "", ".done"), os.path.join(c.rr, rid, "done")) if os.path.exists(x)]
    head = f"{rid}: status {st}, exit {ex}, started {rec.get('started', UNKNOWN)}, command {str(rec.get('command'))[:70]}"
    if st == "ok" and ex == 0:
        return (OK, head)
    if st == "ok":
        return (MISMATCH, head + " (status ok but the exit code is not 0)")
    if st in ("running", None) and ex is None:
        return (MISMATCH, head + " (stale record: the run folder has a done flag)") if done else (MISSING, head + " (not finished, or a stale record: no done flag)")
    return (MISMATCH if kept else MISSING, head + (" (a kept network comes from a run that did not finish OK)" if kept else " (the run did not finish OK)"))


def code_diff(c, a, b):
    """files of the project's code folders that differ between two commits (None when git cannot tell)"""
    paths = [r for r in (os.path.relpath(os.path.join(c.root, d), c.repo).replace("\\", "/") for d in c.cfg.get("code_dirs", [])) if not r.startswith("..")]
    out = c.git("diff", "--numstat", a, b, "--", *paths) if paths else None
    return None if out is None else [ln.split("\t")[-1] for ln in out.splitlines() if ln]


def check_code(c, rid, rec, strict, label="run record"):
    out = []
    tag, commit, note = rec.get("tag") or "", rec.get("commit") or "", str(rec.get("git_note") or "")
    if not tag or not COMMIT.fullmatch(commit):
        return [(MISSING, label, f"{rid}: tag {tag or 'not recorded'}, commit {commit or 'not recorded'} (no snapshot commit was made, or an old run)")]
    tc = c.git("rev-parse", "--verify", "-q", f"refs/tags/{tag}^{{commit}}")
    if not tc:
        out.append((MISSING, label, f"{rid}: tag {tag} is not in this repository (git fetch --tags?)"))
    elif tc.startswith(commit) or commit.startswith(tc):
        out.append((OK, label, f"{rid}: tag {tag} -> commit {tc[:10]} as recorded"))
    elif "tag existed" in note and re.search(r"attempt 1 \((\w+)\)", note) and tc.startswith(re.search(r"attempt 1 \((\w+)\)", note).group(1)):
        m2 = re.search(r"this attempt ran (\w+)", note)
        if not m2 or not (commit.startswith(m2.group(1)) or m2.group(1).startswith(commit)):
            out.append((MISMATCH, label, f"{rid}: resumed run; git_note says this attempt ran {m2.group(1) if m2 else 'nothing'}, the record says {commit}"))
        else:
            out.append((OK, label, f"{rid}: resumed run, tag {tag} holds attempt 1 ({tc[:10]}), this attempt ran {commit} (git_note)"))
            files = code_diff(c, tc, commit)
            if files is None:
                out.append((MISSING, "resume code", f"{rid}: the code of the two attempts cannot be compared (git diff {tc[:10]} {commit[:10]} failed)"))
            elif files:
                out.append((MISMATCH if strict else INFO, "resume code", f"{rid}: the two attempts ran different code: {len(files)} file(s) of the code folders differ ({', '.join(files[:3])}{' ...' if len(files) > 3 else ''})"))
            else:
                out.append((OK, "resume code", f"{rid}: both attempts ran the same code (code folders identical)"))
    else:
        out.append((MISMATCH, label, f"{rid}: tag {tag} -> {tc[:10]} but the record says commit {commit}"))
    if c.git("cat-file", "-e", f"{commit}^{{commit}}") is None:
        out.append((MISSING, "code", f"{rid}: commit {commit} is not in this repository"))
    elif c.git("merge-base", "--is-ancestor", commit, "HEAD") is not None:
        out.append((OK, "code", f"{rid}: commit {commit} is an ancestor of HEAD"))
    else:
        tags = (c.git("tag", "--points-at", commit) or "").split()
        out.append((OK, "code", f"{rid}: commit {commit} is held by tag {', '.join(tags)} (local; pushed or not: check_git.py)") if tags else
                   (MISSING, "code", f"{rid}: commit {commit} exists but no branch or tag holds it (git gc can remove it)"))
    d = rec.get("code_drift") or {}
    if d.get("n"):
        out.append((MISMATCH if strict else INFO, "code drift", f"{rid}: {d['n']} file(s) differed from the commit when it ran: {', '.join(d.get('paths', [])[:4])}{' ...' if d['n'] > 4 else ''}"))
    elif rec.get("code_drift_check"):
        out.append((MISSING, "code drift", f"{rid}: the drift check itself failed: {rec['code_drift_check']}"))
    else:
        out.append((OK, "code drift", f"{rid}: none recorded"))
    return out


def check_snapshot(c, rdir, rec):
    f = os.path.join(rdir, "code", "SNAPSHOT.txt") if rdir else ""
    if not f or not os.path.isfile(f):
        return []
    m = re.match(r"\s*commit (\w+)", open(f, encoding="utf-8", errors="replace").readline())
    rc = rec.get("commit") or ""
    if not m or not COMMIT.fullmatch(rc):
        return [(MISSING, "code snapshot", f"{c.rel(f)} names commit {m.group(1) if m else 'none'}, the record has {rc or 'none'}: nothing to compare")]
    same = m.group(1).startswith(rc) or rc.startswith(m.group(1))
    return [(OK if same else MISMATCH, "code snapshot", f"the snapshot folder of the run names commit {m.group(1)}, the record {rc}")]


def check_parent(c, net, rdir, lin, rec):
    p = (lin or {}).get("parent") or (rec or {}).get("parent")
    cfg = jload(os.path.join(rdir, "config.json")) if rdir else None
    if not p:
        if cfg is not None and cfg.get("init_from"):
            p = dict(path=cfg["init_from"], kind="init", note="config.json init_from only: no hash recorded")
        elif cfg is not None:
            return [(OK, "parent", "config.json init_from is empty: a fresh start (a resume of the same run keeps its run id)")]
        else:
            return [(MISSING, "parent", "no parent recorded (no lineage.json, no config.json): fresh start or old run, not told")]
    f = os.path.join(c.root, p["path"])
    if not os.path.isfile(f):
        return [(MISSING, "parent", f"{p.get('kind', 'init')} from {p['path']}: the file does not exist here")]
    if not p.get("sha256"):
        return [(MISSING, "parent", f"{p.get('kind', 'init')} from {p['path']} exists; {p.get('note') or 'no hash recorded'} (current file_hash {PV.file_hash(f)})")]
    h, kind = hash_as(f, p["sha256"])
    return [(OK if h == p["sha256"] else MISMATCH, "parent", f"{p.get('kind', 'init')} from {p['path']}: hash {'equals' if h == p['sha256'] else 'DIFFERS from'} the recorded one ({kind})")]


def check_gt(c, sig_key, version, label):
    """sig_key = a GT cache key (0.5_4_<signature>) or a bare signature; version = the version a provenance block names, or None"""
    sig = (sig_key or "").split("_")[-1]
    if not sig or sig == UNKNOWN:
        return [(MISSING, "data (GT)", f"{label}: no GT signature recorded")]
    row = c.versions.get(sig)
    if row is None:
        byv = next((r for r in c.versions.values() if r["version"] == version), None)
        if byv:
            return [(MISMATCH, "data (GT)", f"{label}: names version {version} with signature {sig}, autogt/versions.csv has {byv['signature']} for it")]
        return [(MISSING, "data (GT)", f"{label}: signature {sig} is no row of autogt/versions.csv (an unregistered generator state)")]
    out = []
    if version not in (None, UNKNOWN) and version != row["version"]:
        return [(MISMATCH, "data (GT)", f"{label}: names version {version} but signature {sig} is {row['version']} in autogt/versions.csv")]
    tc = c.git("rev-parse", "--verify", "-q", f"refs/tags/{row['tag']}^{{commit}}")
    out.append((OK if tc else MISSING, "data (GT)", f"{label}: signature {sig} = {row['version']} ({row['date']}); git tag {row['tag']} " + (f"-> {tc[:10]}" if tc else "not found")))
    idx = jload(os.path.join(c.repo, "data", "autogt", sig, "index.json"))
    if idx and idx.get("gt_version") not in (None, row["version"]):
        out.append((MISMATCH, "data (GT)", f"{label}: the GT store {sig} says {idx['gt_version']}, versions.csv {row['version']}"))
    return out


def check_manifest(c, prov, label):
    rec = ((prov or {}).get("data") or {}).get("manifest_hash")
    if rec in (None, UNKNOWN):
        return (MISSING, f"{label}: no raw-data manifest hash recorded")
    if not os.path.isfile(c.manifest):
        return (MISSING, f"{label}: datasets/NuScenes/manifest.json not present here")
    cur = PV.file_hash(c.manifest)
    if cur == rec:
        return (OK, f"{label}: manifest hash {rec} = current")
    # the file changed since the card (it is rewritten whenever anything under datasets/NuScenes is regenerated): only the entry the card names decides
    data = (prov or {}).get("data") or {}
    now = ((jload(c.manifest) or {}).get("datasets") or {}).get(data.get("dataset") or "") or {}
    named = {k: data[k] for k in ("scenes", "samples") if data.get(k) is not None}
    changed = {k: (v, now.get(k)) for k, v in named.items() if now.get(k) != v}
    if changed:
        return (MISMATCH, f"{label}: manifest hash {rec} != current {cur}, and the entry of {data.get('dataset')} changed: " + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in changed.items()))
    return (MISSING, f"{label}: manifest file changed since the card (hash {rec} -> {cur}); " +
            (f"the entry it names ({data.get('dataset')}: " + ", ".join(f"{k} {v}" for k, v in named.items()) + ") is unchanged" if named else "the card names no scene / sample count, so it cannot tell whether its data changed"))


def find_cards(c, net):
    seen, out = set(), []
    for n in net["names"]:
        for pat in (f"{n}/*.json", f"{n}/eval/*.json"):
            for f in sorted(glob.glob(os.path.join(c.cards, pat))):
                if f in seen or os.path.getsize(f) > (50 << 20):
                    continue
                d = jload(f)
                if d and "metrics" in d and "split" in d and any(k in d for k in ("name", "commit", "provenance")):    # not a metrics.json that only repeats a card
                    seen.add(f)
                    out.append((f, d))
    return out


def named_by(f, d):
    """(checkpoint file name, run folder name) a card or result file names: its provenance.subject / own fields, else the metrics.json beside it (BEV), else the registry run (lane)"""
    s = (d.get("provenance") or {}).get("subject") or {}
    sib = jload(os.path.join(os.path.dirname(f), "metrics.json")) if os.path.basename(f) != "metrics.json" else None
    ck = s.get("ckpt") or d.get("checkpoint") or (sib or {}).get("checkpoint")
    run = d.get("run") or (sib or {}).get("run") or (d.get("registry") or {}).get("run")
    return ck, (os.path.basename(str(run).rstrip("/")) if run else None)


def tie(net, f, d):
    """(status, text): does the card name the network's checkpoint file / run folder? a different name = MISMATCH, none = MISSING"""
    ck, run = named_by(f, d)
    mine = os.path.basename(net["ckpt"]) if net["ckpt"] else None
    if (ck and mine and ck != mine) or (run and net["key"] and run != net["key"]):
        return (MISMATCH, f"names checkpoint {ck or '-'} / run {run or '-'}, the network is {mine or '-'} / {net['key']}")
    return (OK, "tied to the checkpoint") if ck or run else (MISSING, "names no checkpoint or run")


def check_rank_card(c, net):
    cp = (net["slot"] or {}).get("card")
    if not cp:
        return []
    f = os.path.join(c.root, cp)
    d = jload(f)
    if d is None:
        return [(MISSING, "ranking card", f"{cp} (keep3.json) does not exist here or is unreadable")]
    st, txt = tie(net, f, d)
    others = {os.path.basename(os.path.dirname(k.get("path", ""))) for k in net["slot"].get("checkpoints", [])} - {net["key"]}
    if st == MISMATCH and named_by(f, d)[1] in others:      # an arm held with several seeds has one ranking card: it is the card of one seed
        return [(MISSING, "ranking card", f"{cp} is the card of {named_by(f, d)[1]}, another checkpoint of this keep3 entry: nothing ties it to this one")]
    return [(st, "ranking card", f"{cp}: {txt}")]


def check_cards(c, cards, net):
    """one line per card with a provenance block, one line per group of cards that read alike without one (the lane project has 100+ per network)"""
    out, with_prov, groups = [], [], {}
    for f, d in cards:
        dn = os.path.dirname(f)
        name = f"{os.path.basename(os.path.dirname(dn)) if os.path.basename(dn) == 'eval' else os.path.basename(dn)}/{os.path.basename(f)}"
        cm, prov = str(d.get("commit") or "").split("+")[0], d.get("provenance")
        commit = (OK if cm and c.git("cat-file", "-e", f"{cm}^{{commit}}") is not None else MISSING, str(d.get("commit") or "no commit"))
        if isinstance(prov, dict) and not prov.get("error"):
            with_prov.append((name, d))
            parts = [commit, tie(net, f, d), (OK, "provenance"), (lambda r: (r[0], r[1].split(": ", 1)[1]))(check_manifest(c, prov, "manifest"))]
            parts += [(g[0], g[2].split(": ", 1)[1][:70]) for g in check_gt(c, (prov.get("gt") or {}).get("signature"), (prov.get("gt") or {}).get("version"), "card GT")]
            out.append((worst(s for s, _ in parts), "evaluation", f"card {name}: " + "; ".join(f"{s} {t}" for s, t in parts)))
            continue
        parts = [tie(net, f, d), (MISSING, "no usable provenance block" + (f" ({prov['error']})" if isinstance(prov, dict) else ""))]
        sig = (d.get("protocol") or {}).get("gt_signature")
        if sig:
            parts += [(g[0], "protocol " + g[2].split(": ", 1)[1][:70]) for g in check_gt(c, sig, None, "card GT")]
        groups.setdefault(tuple(parts), []).append((name, commit))
    for parts, members in groups.items():
        names = [n for n, _ in members]
        ok, dirty = sum(cm[0] == OK for _, cm in members), sum("+dirty" in cm[1] for _, cm in members)
        text = f"{len(members)} card(s) ({', '.join(names[:3])}{' ...' if len(names) > 3 else ''}): commit in git {ok}/{len(members)} ({dirty} made on a dirty tree); " + "; ".join(f"{s} {t}" for s, t in parts)
        out.append((worst([s for s, _ in parts] + [OK if ok == len(members) else MISSING]), "evaluation", text))
    for i in range(len(with_prov)):
        for j in range(i + 1, min(len(with_prov), i + 4)):
            text, code = PV.verdict(*PV.compare(with_prov[i][1], with_prov[j][1]))
            out.append((INFO, "compare", f"{with_prov[i][0]} vs {with_prov[j][0]}: {text.splitlines()[0] if code == 0 else text.replace(chr(10), '; ')}"))
    return out


def check_network(c, net, strict=False):
    """all checks of one network -> list of (status, link, text)"""
    rid, mention = find_train_record(c, net)
    rec = c.recs.get(rid) if rid else None
    lin, bad = jread(os.path.join(c.rr, rid, "lineage.json")) if rid else (None, None)
    rdir = os.path.dirname(os.path.join(c.root, net["ckpt"])) if net["ckpt"] else (os.path.join(c.root, "runs", net["key"]) if net["key"] else None)
    cards = find_cards(c, net)
    out = ([(MISMATCH, "lineage.json", bad)] if bad else []) + check_ckpt(c, net, lin, cards) + check_rank_card(c, net)
    if rec:
        st, txt = check_run_state(c, rid, rec, rdir, bool(net["slot"]))
        out.append((st, "run record", txt))
        out += check_code(c, rid, rec, strict)
        out += check_snapshot(c, rdir, rec)
        for r2 in sorted(i for i in c.recs if i.startswith(rid + ".")):
            out += check_code(c, r2, c.recs[r2], strict, label="eval record")[:2]
    else:
        out.append((MISSING, "run record", f"no record names {net['key']} as its run id or config (trained before the launcher, or under another id)"))
    if mention:
        out.append((INFO, "run record", "other records that mention the network: " + ", ".join(mention[:6]) + (" ..." if len(mention) > 6 else "")))
    out += check_parent(c, net, rdir, lin, rec)
    g = jload(os.path.join(rdir, "gt.json")) if rdir else None
    rp = (rec or {}).get("provenance") or {}
    sig = (g or {}).get("gt_signature") or (rp.get("gt") or {}).get("signature")
    out += check_gt(c, sig, (rp.get("gt") or {}).get("version"), "training GT")
    if rp:
        st, txt = check_manifest(c, rp, "training data")
        out.append((st, "data (raw)", txt))
    out += check_cards(c, cards, net) or [(MISSING, "evaluation", f"no card under {c.rel(c.cards)} names {', '.join(net['names'][:4]) or 'the network'}")]
    return out


# ---------------------------------------------------------------- record (the additive piece for future networks)
def record(run, ckpt=None, parent=None, parent_kind="init"):
    """write/merge <run records>/<id>/lineage.json: the checkpoint of the run and the checkpoint it was started from, each with size and sha256 (paths relative to the problem folder)"""
    root = P.root()
    rdir = run if os.path.isfile(os.path.join(run, "run.json")) else os.path.join(P.path("run_records"), run)
    if not os.path.isfile(os.path.join(rdir, "run.json")):
        sys.exit(f"no run record {rdir}")
    f = os.path.join(rdir, "lineage.json")
    d = jload(f) or {}
    d.update(schema=1, run=os.path.basename(os.path.normpath(rdir)), made=dt.datetime.now().isoformat(timespec="seconds"))
    for key, path, extra in (("checkpoint", ckpt, {}), ("parent", parent, {"kind": parent_kind})):
        if path:
            ap = path if os.path.isabs(path) else os.path.join(root, path)
            if not os.path.isfile(ap):
                sys.exit(f"{key} {ap} does not exist")
            d[key] = dict(path=os.path.relpath(ap, root).replace("\\", "/"), bytes=os.path.getsize(ap), sha256=sha256_of(ap), **extra)
    P.atomic_write(f, json.dumps(d, indent=1) + "\n")
    return f


# ---------------------------------------------------------------- commands
KINDS = {"checkpoint": ("checkpoint hash",), "code": ("code",), "data": ("data (GT)", "data (raw)")}


def report(c, nets, strict=False, as_json=False, require=()):
    """print the checks of every network; -> exit code (1 on a MISMATCH, an unreadable record, or a required kind without an OK link)"""
    allc, rows = {}, []
    for p in c.problems:
        allc[MISMATCH] = allc.get(MISMATCH, 0) + 1
        print(f"  {MISMATCH:8} {'record':15} {p}", file=sys.stderr if as_json else sys.stdout)
    for n in nets:
        ch = check_network(c, n, strict)
        for k in require:
            if not any(s == OK and lk in KINDS[k] for s, lk, _ in ch):
                ch.append((MISMATCH, "required", f"no OK link of kind {k} ({'/'.join(KINDS[k])}) for this network"))
        rows.append((n, ch))
        for s, _, _ in ch:
            allc[s] = allc.get(s, 0) + 1
    if as_json:
        print(json.dumps([dict(network=n, checks=[dict(status=s, link=lk, text=t) for s, lk, t in ch]) for n, ch in rows], indent=1))
    else:
        for n, ch in rows:
            slot = n["slot"]
            print(f"== {', '.join(n['names'][:3]) or n['key']}  (network {n['key']}" + (f", keep3 slot {slot['n']}, {c.keep.get('metric', 'value')} {slot['value']:.4f}" if slot else "") + ")")
            for s, lk, t in ch:
                print(f"  {s:8} {lk:15} {t}")
        print(f"{len(nets)} network(s): " + ", ".join(f"{k} {allc.get(k, 0)}" for k in (OK, MISSING, MISMATCH, INFO)))
    return 1 if allc.get(MISMATCH) else 0


def req(a):
    ks = [k for x in (getattr(a, "require", None) or []) for k in x.split(",") if k]
    bad = [k for k in ks if k not in KINDS]
    if bad:
        sys.exit(f"--require {bad}: kinds are {', '.join(KINDS)}")
    return ks


def cmd_verify(a):
    c = Ctx()
    nets = keep_nets(c)
    if a.all:
        nets += [make_net(c, c.reg[n]["run"], names=[n]) for n in ("champion", "control") if n in c.reg and c.reg[n].get("run")]
    for t in a.names:
        nets += resolve(c, t)
    seen, uniq = set(), []
    for n in nets:                                 # one network once, with all its names
        k = n["ckpt"] or n["key"]
        if k in seen:
            continue
        seen.add(k)
        uniq.append(n)
    if not uniq:
        print(f"nothing to verify: no keep3.json for {c.root} (and no names given)" + ("" if not c.problems else "; but " + "; ".join(c.problems)))
        return 1 if c.problems else 0
    return report(c, uniq, a.strict, a.json, req(a))


def cmd_show(a):
    c = Ctx()
    return report(c, resolve(c, a.target), a.strict, a.json, req(a))


def selftest():
    T = tempfile.mkdtemp(prefix="ar_lineage_")
    old = os.environ.get("AR_PROJECT")
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")

    def git(*a):
        return subprocess.run(["git", "-C", T, *a], check=True, capture_output=True, text=True, env=env).stdout.strip()

    def write(rel, text, mode="w"):
        p = os.path.join(T, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, mode) as f:
            f.write(text)
        return p

    def wj(rel, obj):
        with open(write(rel, ""), "w") as f:
            json.dump(obj, f)

    def run(*targets, **kw):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cmd_verify(argparse.Namespace(names=list(targets), all=False, strict=kw.get("strict", False), json=False, require=kw.get("require")))
        return rc, buf.getvalue()

    def show(target, as_json=False):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cmd_show(argparse.Namespace(target=target, strict=False, json=as_json))
        return rc, buf.getvalue()

    def line(txt, link, needle=""):
        hits = [ln for ln in txt.splitlines() if ln.split()[1:2] and re.match(r"\s+(\w+)\s+" + re.escape(link) + r"(?!\w)", ln) and needle in ln]
        assert hits, (link, needle, txt)
        return hits[0].split()[0]
    try:
        git("init", "-q")
        write("datasets/NuScenes/manifest.json", '{"datasets": {"v1.0-trainval": {"scenes": 850}}}')
        write("autogt/versions.csv", "version,signature,tag,date,generator,what_changed\nv5,abc123,autogt/v5,2026-10-31,x,y\n")
        write("code/a.py", "x = 1\n")
        git("add", "-A")
        git("commit", "-q", "-m", "c1")
        c1 = git("rev-parse", "HEAD")
        git("tag", "autogt/v5")
        write("code/a.py", "x = 2\n")
        git("add", "-A")
        git("commit", "-q", "-m", "c2")
        c2 = git("rev-parse", "HEAD")
        git("tag", "t/netrun", c2)
        orphan = git("commit-tree", git("rev-parse", "HEAD^{tree}"), "-m", "snapshot")           # like run.py's snapshot: held by a tag only
        git("tag", "t/orph", orphan)
        lost = git("commit-tree", git("rev-parse", "HEAD^{tree}"), "-m", "lost")                # no ref at all
        # the problem folder
        wj("proj/project.json", {"run_records": "records/runs", "ledger": "records/LEDGER.csv", "cards": "runs", "tag_prefix": "t", "code_dirs": ["../code"]})
        ck = write("proj/runs/netrun/ckpt_00000010.pt", "weights-A" * 50)
        ckb = open(ck, "rb").read()
        write("proj/runs/netrun/gt.json", '{"gt_signature": "0.5_4_abc123"}')
        write("proj/runs/netrun/config.json", '{"init_from": ""}')
        write("proj/runs/netrun/code/SNAPSHOT.txt", f"commit {c2[:10]} (git archive)\n")
        write("proj/runs/parent/ckpt.pt", "parent-weights")
        wj("proj/records/runs/netrun/run.json", {"id": "netrun", "status": "ok", "exit": 0, "command": "bash run.sh training/configs/netrun.json", "tag": "t/netrun", "commit": c2[:10],
                   "code_drift": {"n": 1, "paths": ["code/a.py"]}})
        wj("proj/records/runs/netrun.eval/run.json", {"id": "netrun.eval", "tag": "t/orph", "commit": orphan[:10]})
        wj("proj/records/runs/old_run/run.json", {"id": "old_run", "status": "ok", "command": "bash old.sh"})
        write("proj/runs/oldnet/model.pt", "old-weights")
        write("proj/runs/lostnet/model.pt", "lost-weights")
        write("proj/runs/skipnet/model.pt", "skip-weights")
        wj("proj/records/runs/skipnet/run.json", {"id": "skipnet", "tag": "t/netrun", "commit": "skipped"})
        wj("proj/records/runs/lostnet/run.json", {"id": "lostnet", "tag": "t/lostnet", "commit": lost[:10]})
        man = PV.file_hash(os.path.join(T, "datasets/NuScenes/manifest.json"))

        def prov(**k):
            p = {"schema": 1, "gt": {"signature": "abc123", "version": "v5"}, "evaluator": {"dirty": [], "code_hash": "e1"}, "data": {"frame_hash": "f1", "manifest_hash": man, "dataset": "v1.0-trainval", "scenes": 850},
                 "subject": {"ckpt": "ckpt_00000010.pt", "ckpt_hash": PV.file_hash(ck)}}
            for kk, v in k.items():
                a_, b_ = kk.split("__")
                p[a_] = dict(p[a_], **{b_: v})
            return p

        def card(path, **k):
            wj(path, {"name": "netrun", "split": "val", "metrics": {}, "commit": c1[:10], "protocol": {"gt_signature": "0.5_4_abc123"}, **k})
        card("proj/runs/netrun/eval/card.json", provenance=prov())
        card("proj/runs/netrun/eval/card2.json", provenance=prov(evaluator__code_hash="e2"))
        write("proj/runs/armA/model.pt", "arm-a")
        write("proj/runs/armB/model.pt", "arm-b")
        wj("proj/runs/netrun/eval/metrics.json", {"run": "runs/netrun", "checkpoint": "ckpt_00000010.pt", "split": "val", "metrics": {}})      # the BEV result file beside a card
        keep = {"metric": "board7", "slots": 3, "registry": "models/registry.json", "ckpt": "runs/{run}/ckpt_00000010.pt", "entries": [
            {"name": "netrun", "key": "netrun", "value": 0.7, "card": "runs/netrun/eval/metrics.json",
             "checkpoints": [{"path": "runs/netrun/ckpt_00000010.pt", "bytes": len(ckb), "sha256": hashlib.sha256(ckb).hexdigest()}]},
            {"name": "oldnet", "key": "oldnet", "value": 0.6, "card": "runs/oldnet/eval/metrics.json",
             "checkpoints": [{"path": "runs/oldnet/model.pt", "bytes": 11, "sha256": hashlib.sha256(b"old-weights").hexdigest()}]},
            {"name": "arm", "key": "arm", "value": 0.5, "card": "runs/armA/eval/metrics.json", "checkpoints": [{"path": f"runs/arm{x}/model.pt", "bytes": 5, "sha256": hashlib.sha256(f"arm-{x.lower()}".encode()).hexdigest()} for x in "AB"]}]}
        wj("proj/runs/armA/eval/metrics.json", {"run": "runs/armA", "checkpoint": "model.pt", "split": "val", "metrics": {}})
        wj("proj/records/keep3.json", keep)
        os.environ["AR_PROJECT"] = os.path.join(T, "proj")

        # a consistent chain: no MISMATCH; the old network is MISSING its record, not a MISMATCH
        rc, txt = run()
        assert rc == 0, txt
        assert line(txt, "checkpoint hash", "keep3.json") == OK and line(txt, "checkpoint hash", "provenance of") == OK
        assert line(txt, "run record", "netrun: status") == OK and line(txt, "run record", "no record names oldnet") == MISSING, txt
        assert line(txt, "code", "ancestor of HEAD") == OK and line(txt, "code drift", "1 file(s)") == INFO
        assert line(txt, "code snapshot") == OK and line(txt, "parent", "fresh start") == OK and line(txt, "parent", "not told") == MISSING
        assert line(txt, "data (GT)", "training GT: signature abc123 = v5") == OK
        assert line(txt, "evaluation", "card.json") == OK and "not shown comparable" not in txt and "evaluator.code_hash: e1 != e2" in txt, txt
        assert "held by tag t/orph" in txt, "the evaluation record's snapshot commit is held by its tag only"
        assert run(strict=True)[0] == 1, "--strict: drift is a MISMATCH"
        # lostnet: the commit exists but no ref holds it, and no hash is recorded: MISSING both, not a MISMATCH
        rc, txt = show("lostnet")
        assert rc == 0 and line(txt, "code", "no branch or tag holds it") == MISSING and line(txt, "checkpoint hash") == MISSING, txt
        rc, txt = show("skipnet")                                                       # a launcher word instead of a commit is not recorded, not a mismatch
        assert rc == 0 and line(txt, "run record", "commit skipped") == MISSING and "code snapshot" not in txt, txt
        # mutation 1: one byte of the checkpoint (same size)
        open(ck, "wb").write(ckb[:-1] + b"X")
        rc, txt = run()
        assert rc == 1 and line(txt, "checkpoint hash", "keep3.json") == MISMATCH and line(txt, "checkpoint hash", "provenance of") == MISMATCH, txt
        open(ck, "wb").write(ckb[:-1])                                              # a shorter file: the size is seen without hashing
        assert "size" in run()[1] and run()[0] == 1
        open(ck, "wb").write(ckb)
        assert run()[0] == 0
        assert "board7 0.7000" in run()[1], "the header names the keep3 metric"
        # a checkpoint that is recorded but not here (archived, slimmed, a clone) is MISSING, not a MISMATCH
        os.rename(os.path.join(T, "proj/runs/oldnet/model.pt"), os.path.join(T, "proj/runs/oldnet/model.moved"))
        rc, txt = run()
        assert rc == 0 and line(txt, "checkpoint", "oldnet/model.pt does not exist") == MISSING, txt
        os.rename(os.path.join(T, "proj/runs/oldnet/model.moved"), os.path.join(T, "proj/runs/oldnet/model.pt"))
        # the state of the run: failed / killed / stale 'running' must not read OK
        rj0 = json.load(open(os.path.join(T, "proj/records/runs/netrun/run.json")))
        for status, ex, done, want, code in (("failed", 75, False, MISMATCH, 1), ("running", None, True, MISMATCH, 1), ("running", None, False, MISSING, 0), ("ok", 1, False, MISMATCH, 1)):
            wj("proj/records/runs/netrun/run.json", dict(rj0, status=status, exit=ex))
            if done:
                write("proj/runs/netrun/.done", "")
            rc, txt = run()
            assert rc == code and line(txt, "run record", f"netrun: status {status}") == want, (status, ex, done, txt)
            if done:
                os.remove(os.path.join(T, "proj/runs/netrun/.done"))
        wj("proj/records/runs/lostnet/run.json", {"id": "lostnet", "status": "failed", "exit": 1, "tag": "t/lostnet", "commit": lost[:10]})
        rc, txt = show("lostnet")
        assert rc == 0 and line(txt, "run record", "lostnet: status failed") == MISSING, "a failed run that is no kept network reads MISSING, never OK"
        wj("proj/records/runs/netrun/run.json", rj0)
        # a card must name the checkpoint it is tied to; a ranking card (keep3) too
        wj("proj/runs/netrun/eval/metrics.json", {"run": "runs/netrun", "checkpoint": "ckpt_00000099.pt", "split": "val", "metrics": {}})
        rc, txt = run()
        assert rc == 1 and line(txt, "ranking card", "names checkpoint ckpt_00000099.pt") == MISMATCH and line(txt, "ranking card", "oldnet") == MISSING, txt
        wj("proj/runs/netrun/eval/metrics.json", {"run": "runs/netrun", "checkpoint": "ckpt_00000010.pt", "split": "val", "metrics": {}})
        assert line(run()[1], "ranking card", "netrun") == OK
        card("proj/runs/netrun/eval/card3.json", provenance=prov(subject__ckpt="ckpt_00000007.pt"))
        rc, txt = run()
        assert rc == 1 and "MISMATCH names checkpoint ckpt_00000007.pt" in txt, txt
        os.remove(os.path.join(T, "proj/runs/netrun/eval/card3.json"))
        card("proj/runs/netrun/card4.json", provenance=prov(subject__ckpt=None))                          # a card with provenance that names no checkpoint (no sibling metrics.json)
        rc, txt = run()
        assert rc == 0 and line(txt, "evaluation", "card netrun/card4.json") == MISSING and "MISSING names no checkpoint or run" in txt, txt
        os.remove(os.path.join(T, "proj/runs/netrun/card4.json"))
        assert run()[0] == 0
        # records that exist but cannot be read are a failure, an absent keep3.json is not
        kf = os.path.join(T, "proj/records/keep3.json")
        good = open(kf).read()
        open(kf, "w").write(good[:40])
        rc, txt = run()
        assert rc == 1 and "keep3.json exists but is not readable json" in txt and "nothing to verify" in txt, txt
        os.remove(kf)
        rc, txt = run()
        assert rc == 0 and "nothing to verify" in txt and "not readable" not in txt, txt
        open(kf, "w").write(good)
        write("proj/records/runs/torn/run.json", '{"id": "torn"')
        rc, txt = run()
        assert rc == 1 and "torn" in txt and "not readable json" in txt, txt
        os.remove(os.path.join(T, "proj/records/runs/torn/run.json"))
        # --require: lenient by default; a network without an OK link of the kind fails
        assert run()[0] == 0
        rc, txt = run(require=["code"])
        assert rc == 1 and "no OK link of kind code" in txt and txt.count("required") == 3, txt        # oldnet, armA, armB: no record, so no OK code link
        rc, txt = run(require=["checkpoint,data"])
        assert rc == 1 and "no OK link of kind data" in txt and "no OK link of kind checkpoint" not in txt, txt
        # a kept arm shows every one of its seeds
        rc, txt = show("arm")
        assert txt.count("== ") == 2 and "armA" in txt and "armB" in txt, txt
        assert line(txt, "ranking card", "armA/eval/metrics.json: tied") == OK and line(txt, "ranking card", "card of armA, another checkpoint") == MISSING and rc == 0, txt
        # mutation 2: the tag points to another commit
        git("tag", "-f", "t/netrun", c1)
        rc, txt = run()
        assert rc == 1 and line(txt, "run record", "tag t/netrun ->") == MISMATCH, txt
        rj, r0 = "proj/records/runs/netrun/run.json", json.load(open(os.path.join(T, "proj/records/runs/netrun/run.json")))
        wj(rj, dict(r0, git_note=f"tag existed (resume): tag t/netrun is attempt 1 ({c1[:10]}), this attempt ran {c2[:10]}"))      # the recorded resume case
        rc, txt = run()
        assert rc == 0 and "resumed run" in txt and line(txt, "resume code", "different code: 1 file(s)") == INFO, txt     # code/a.py differs between the attempts
        rc, txt = run(strict=True)
        assert rc == 1 and line(txt, "resume code") == MISMATCH, txt
        wj(rj, dict(r0, git_note=f"tag existed (resume): tag t/netrun is attempt 1 ({c1[:10]}), this attempt ran {c1[:10]}"))      # the note names another commit than the record
        rc, txt = run()
        assert rc == 1 and line(txt, "run record", "git_note says this attempt ran") == MISMATCH, txt
        git("tag", "-f", "t/netrun", orphan)                                         # attempt 1 = a commit with the same tree as attempt 2
        wj(rj, dict(r0, git_note=f"tag existed (resume): tag t/netrun is attempt 1 ({orphan[:10]}), this attempt ran {c2[:10]}"))
        rc, txt = run()
        assert rc == 0 and line(txt, "resume code", "same code") == OK, txt
        wj(rj, r0)
        git("tag", "-f", "t/netrun", c2)
        write("proj/runs/netrun/code/SNAPSHOT.txt", f"commit {c1[:10]} (git archive)\n")                                       # the snapshot folder names another commit
        rc, txt = run()
        assert rc == 1 and line(txt, "code snapshot") == MISMATCH, txt
        write("proj/runs/netrun/code/SNAPSHOT.txt", f"commit {c2[:10]} (git archive)\n")
        # mutation 3: the signature differs from the version a provenance block names (and an unregistered signature is MISSING)
        card("proj/runs/netrun/eval/card2.json", provenance=prov(gt__version="v4"))
        rc, txt = run()
        assert rc == 1 and "MISMATCH names version v4 but signature abc123 is v5" in txt, txt
        card("proj/runs/netrun/eval/card2.json", provenance=prov(gt__signature="zzz999"))
        rc, txt = run()
        assert rc == 1 and "MISMATCH names version v5 with signature zzz999" in txt, txt
        card("proj/runs/netrun/eval/card2.json", provenance=prov(gt__signature="zzz999", gt__version="v9"))
        rc, txt = run()
        assert rc == 0 and "MISSING signature zzz999 is no row" in txt, txt
        card("proj/runs/netrun/eval/card2.json", provenance=prov(evaluator__code_hash="e2"))
        # mutation 4: the data manifest changes: the entry the card names (850 scenes) changed = MISMATCH; the file changed elsewhere = MISSING (exit 0), not a failure
        write("datasets/NuScenes/manifest.json", '{"datasets": {"v1.0-trainval": {"scenes": 851}}}')
        rc, txt = run()
        assert rc == 1 and "MISMATCH manifest hash" in txt and "scenes 850 -> 851" in txt, txt
        write("datasets/NuScenes/manifest.json", '{"about": "regenerated", "datasets": {"v1.0-trainval": {"scenes": 850}}}')
        rc, txt = run()
        assert rc == 0 and "MISSING manifest file changed since the card" in txt and "is unchanged" in txt, txt
        write("datasets/NuScenes/manifest.json", '{"datasets": {"v1.0-trainval": {"scenes": 850}}}')
        assert run()[0] == 0
        # an old card (no provenance) is MISSING, never a guess
        card("proj/runs/netrun/eval/card2.json")
        rc, txt = run()
        assert rc == 0 and "MISSING no usable provenance block" in txt, txt
        # record: checkpoint + parent with sha256; a changed parent is a MISMATCH; show finds the network by checkpoint path, run id, and name
        record("netrun", ckpt="runs/netrun/ckpt_00000010.pt", parent="runs/parent/ckpt.pt", parent_kind="init")
        lin = json.load(open(os.path.join(T, "proj/records/runs/netrun/lineage.json")))
        assert lin["parent"]["sha256"] == hashlib.sha256(b"parent-weights").hexdigest() and lin["checkpoint"]["bytes"] == len(ckb) and lin["parent"]["kind"] == "init", lin
        rc, txt = run()
        assert rc == 0 and line(txt, "parent", "init from runs/parent/ckpt.pt") == OK and line(txt, "checkpoint hash", "lineage.json") == OK, txt
        write("proj/runs/parent/ckpt.pt", "parent-weightX")
        rc, txt = run()
        assert rc == 1 and line(txt, "parent", "DIFFERS") == MISMATCH, txt
        write("proj/runs/parent/ckpt.pt", "parent-weights")
        record("netrun", parent="runs/parent/ckpt.pt", parent_kind="resume")           # merges: the checkpoint entry stays
        assert json.load(open(os.path.join(T, "proj/records/runs/netrun/lineage.json")))["checkpoint"]["bytes"] == len(ckb)
        for t in ("runs/netrun/ckpt_00000010.pt", "netrun", "slot1"):             # a checkpoint path, a name (= run id here), a slot: the same network
            assert "keep3.json" in show(t)[1], t
        assert "old_run: status ok" in show("old_run")[1], "a run id without a network still shows its record"
        assert json.loads(show("netrun", True)[1])[0]["network"]["key"] == "netrun"
        try:
            record("no_such_run")
            raise AssertionError("record of an unknown run must exit")
        except SystemExit:
            pass
        print("lineage selftest passed")
    finally:
        if old is None:
            os.environ.pop("AR_PROJECT", None)
        else:
            os.environ["AR_PROJECT"] = old
        shutil.rmtree(T, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("show")
    s.add_argument("target")
    v = sub.add_parser("verify")
    v.add_argument("names", nargs="*")
    v.add_argument("--all", action="store_true", help="also the registry's champion and control")
    for p in (s, v):
        p.add_argument("--strict", action="store_true", help="a recorded code drift is a MISMATCH")
        p.add_argument("--json", action="store_true")
        p.add_argument("--require", action="append", help="comma list of kinds (checkpoint, code, data): exit 1 when a network has no OK link of that kind (default: lenient)")
    r = sub.add_parser("record")
    r.add_argument("run")
    r.add_argument("--ckpt")
    r.add_argument("--parent")
    r.add_argument("--parent-kind", default="init", choices=["init", "resume"])
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "show":
        return cmd_show(a)
    if a.cmd == "verify":
        return cmd_verify(a)
    if a.cmd == "record":
        print(record(a.run, a.ckpt, a.parent, a.parent_kind))
        return 0
    ap.error("show | verify | record | --selftest")


if __name__ == "__main__":
    sys.exit(main())
