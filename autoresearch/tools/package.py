"""The result package of a run and the clean-up of derived data (D40: derived data may be deleted, of the GT only the latest version stays). Stdlib only.

    python autoresearch/tools/package.py package <run id> [--run-dir d]   write <problem>/results/<id>/package.json (a text record, in git)
    python autoresearch/tools/package.py show <run id>                     print the package
    python autoresearch/tools/package.py clean [--apply] [--min-age-days 2] [--runs a,b]
                                                                            derived data of runs outside the best three: dry run lists and sizes it, --apply deletes
    python autoresearch/tools/package.py gt [--apply]                      GT stores of versions older than the latest: dry run, --apply deletes
    python autoresearch/tools/package.py --selftest

The result folder <problem>/results/<run id>/ holds the package: package.json, provenance.json and the copy of the model bundle (bundle/, not in git:
the code is in the tag). The package is the one place that says how a run was made and what came out: the code tag and commit, the GT version and
signature, the model bundle hash (provenance.py manifest), the numbers (top-level numbers of eval/metrics.json and latency.json, with the hash of the
file), the tier (best three or normal) and the list of DERIVED files (name, size) that a deeper look would use: pictures, videos, pickles, per-scene
folders. The package stays in git; clean removes the derived files and the package then lists them under `deleted` (date, size, files, how to rebuild).
Derived = rebuilt by evaluating the checkpoint again with the code of the tag: files with the suffix .pkl .npz .npy .png .jpg .jpeg .gif .mp4 .avi
.webp and everything under eval/suite, but only in the run folder's top level, eval/, report/ and prototype/. Not derived (never touched): checkpoints,
run.json, config.json, gt.json, logs, metrics / latency / card json, text files, code/, bundle/, autogt/ (the pinned sources of a resume), folders of
other evaluators (eval_2026-10-31_first, eval_legacy_*) and files with legacy, _ctl_ or partial in their path (a superseded evaluator made them:
evaluating again does not reproduce them), symbolic links.
clean refuses a run (and says why) when: the best three are unknown (no keep3.json, or no entry), it is in the best three (by run id, run folder
or the folder named in an entry's checkpoints and card), its record or the record of any run that belongs to it (<id>.eval, a second id for the same
folder) is not final or has a fresh heartbeat, its code tag does not exist, it has no checkpoint (the files could not be rebuilt), or a derived file is
younger than --min-age-days (at least one hour). Before anything is removed the package is written with the list and CLEAN_LOG.md gets its line; the
store files that mark an evaluation as finished (frames.jsonl, manifest.json) go first, so an interrupted delete leaves an unfinished store, never one
that claims to be complete. A later clean adds to the `deleted` list, it never replaces it. Tags must be on origin to be a safe base for a rebuild:
`check_git.py` lists the ones that are not.
gt keeps the latest version of autogt/versions.csv (the highest number) and every store that a run which is not final names in its gt.json (also the
`earlier` ones; a resume needs them); an older store is deleted only when its tag autogt/vN exists (it can be regenerated from the tag and the options in
its index.json) and it is a real folder inside data/autogt. A best-three network trained on an older GT cannot be evaluated again against its own GT
after that store is gone: it is regenerated first. Deletions are appended to <records folder>/CLEAN_LOG.md.
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import project as P  # noqa: E402
import provenance as V  # noqa: E402

DERIVED_SUFFIX = (".pkl", ".npz", ".npy", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".avi", ".webp")
DERIVED_DIRS = ("eval/suite",)
ALLOWED_TOP = ("eval", "report", "prototype", "")          # "" = a file in the run folder itself
NEVER_DIRS = ("code", "bundle", "autogt")
SUPERSEDED = ("legacy", "_ctl_", "partial")               # parts of a path that mark results of a former evaluator or an unfinished one
FIRST = ("frames.jsonl", "manifest.json")                 # the files that mark a store as complete: deleted first
FINAL_STATES = ("ok", "done")
METRIC_FILES = ("eval/metrics.json", "eval/latency.json")
MIN_AGE_DAYS = 1.0 / 24


def records_dir():
    return os.path.dirname(P.path("ledger"))


def run_record(run_id):
    p = os.path.join(P.path("run_records"), run_id, "run.json")
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None


def package_path(run_id):
    return os.path.join(V.results_dir(run_id), "package.json")


def is_final(r):
    return bool(r) and r.get("status") in FINAL_STATES and r.get("exit") == 0


def top3_runs():
    """folder names and ids of the best three (keep3.json beside the ledger): the entry's name, key, runs, and the run folder named in its checkpoint and
    card paths (runs/<folder>/...); None when the record is missing, unreadable or holds no entry (then nothing is deleted)"""
    p = os.path.join(records_dir(), "keep3.json")
    if not os.path.isfile(p):
        return None
    try:
        entries = json.load(open(p, encoding="utf-8")).get("entries") or []
    except (OSError, ValueError, AttributeError):
        return None
    out = set()
    for e in entries:
        out.update(x for x in (e.get("runs") or []) if isinstance(x, str))
        out.update(x for x in (e.get("key"), e.get("name")) if isinstance(x, str))
        paths = [c.get("path", "") for c in (e.get("checkpoints") or []) if isinstance(c, dict)] + [e.get("card") or ""]
        for pa in paths:
            parts = str(pa).replace("\\", "/").split("/")
            if len(parts) >= 2 and parts[0] == "runs":
                out.add(parts[1])
    out.discard("")
    return out or None


def is_derived(rel):
    rel = rel.replace("\\", "/")
    parts = rel.split("/")
    top = parts[0] if len(parts) > 1 else ""
    if top in NEVER_DIRS or top not in ALLOWED_TOP or any(s in rel.lower() for s in SUPERSEDED):
        return False
    return rel.lower().endswith(DERIVED_SUFFIX) or any(rel == d or rel.startswith(d + "/") for d in DERIVED_DIRS)


def derived_files(run_dir):
    """[(relative path, bytes, mtime)] of the derived files of a run folder, sorted; symbolic links are neither followed nor listed"""
    out = []
    for dp, dn, fn in os.walk(run_dir, followlinks=False):
        dn[:] = [d for d in dn if d != "__pycache__" and not os.path.islink(os.path.join(dp, d))]
        for f in fn:
            full = os.path.join(dp, f)
            if os.path.islink(full):
                continue
            rel = os.path.relpath(full, run_dir).replace("\\", "/")
            if is_derived(rel):
                try:
                    st = os.lstat(full)
                except OSError:
                    continue
                out.append((rel, st.st_size, st.st_mtime))
    return sorted(out)


def inside(path, root):
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(root)]) == os.path.realpath(root)
    except ValueError:
        return False


def numbers(path, cap=80):
    """top-level numbers of a json file (bool excluded), at most cap keys; {} when it is not readable"""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(d, dict):
        return {}
    flat = {k: v for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return dict(sorted(flat.items())[:cap])


def build(run_id, run_dir=None):
    """the package dict of a run (nothing is written); the provenance manifest is made first when it is missing"""
    r = run_record(run_id)
    if r is None:
        sys.exit(f"no run record for {run_id}")
    run_dir = run_dir or V.find_run_dir(run_id)
    prov_file = os.path.join(V.results_dir(run_id), "provenance.json")
    prov = V.manifest(run_id, run_dir) if not os.path.isfile(prov_file) else json.load(open(prov_file, encoding="utf-8"))
    have_dir = bool(run_dir and os.path.isdir(run_dir))
    top = top3_runs()
    name = os.path.basename(run_dir.rstrip("/\\")) if have_dir else run_id.replace(".", "_")
    pk = {"schema": 1, "made": dt.datetime.now().isoformat(timespec="seconds"), "run": run_id, "run_dir": os.path.relpath(run_dir, P.root()).replace("\\", "/") if have_dir else V.UNKNOWN,
          "status": {"status": r.get("status"), "exit": r.get("exit"), "final": is_final(r)},
          "tier": "top3" if top is not None and (name in top or run_id in top) else ("normal" if top is not None else V.UNKNOWN),
          "code": {"tag": r.get("tag") or V.UNKNOWN, "commit": r.get("commit") or V.UNKNOWN, "code_hash": r.get("code_hash") or V.UNKNOWN},
          "gt": {"version": V._get(prov, ("subject", "train_gt", "version")) or V._get(prov, ("gt", "version")) or V.UNKNOWN,
                 "signature": V._get(prov, ("subject", "train_gt", "signature")) or V._get(prov, ("gt", "signature")) or V.UNKNOWN},
          "bundle": {"code_hash": V._get(prov, ("bundle", "code_hash")) or V.UNKNOWN, "files": len(V._get(prov, ("bundle", "files")) or [])},
          "model": {"ckpt": V._get(prov, ("subject", "ckpt")) or V.UNKNOWN, "ckpt_hash": V._get(prov, ("subject", "ckpt_hash")) or V.UNKNOWN, "step": V._get(prov, ("subject", "step")),
                    "config_hash": V._get(prov, ("subject", "config_hash")) or V.UNKNOWN, "seed": prov.get("seed", V.UNKNOWN)},
          "provenance": {"file": "provenance.json", "hash": V.file_hash(prov_file) if os.path.isfile(prov_file) else V.UNKNOWN},
          "metrics": {}, "metric_files": {}, "deep": {"files": [], "bytes": 0}, "deleted": []}
    if have_dir:
        for rel in METRIC_FILES:
            f = os.path.join(run_dir, rel)
            if os.path.isfile(f):
                pk["metrics"][rel] = numbers(f)
                pk["metric_files"][rel] = V.file_hash(f)
        pk["evaluation"] = evaluation_block(os.path.join(run_dir, "eval", "card.json"))
        files = derived_files(run_dir)
        pk["deep"] = {"files": [{"path": p, "bytes": b} for p, b, _ in files], "bytes": sum(b for _, b, _ in files)}
    pk["experiment"] = experiment_of(run_id, name)
    return pk


def evaluation_block(card_path):
    """which evaluator and which frames made the numbers: from the provenance block of the evaluation card (eval/card.json); unknown when there is none"""
    out = {"evaluator": {"commit": V.UNKNOWN, "code_hash": V.UNKNOWN, "dirty": V.UNKNOWN}, "split": V.UNKNOWN, "frames": None, "frame_hash": V.UNKNOWN, "manifest_hash": V.UNKNOWN,
           "gt": {"version": V.UNKNOWN, "signature": V.UNKNOWN}, "card": V.UNKNOWN}
    try:
        prov = json.load(open(card_path, encoding="utf-8")).get("provenance") or {}
    except (OSError, ValueError, AttributeError):
        return out
    ev, data, gt = (prov.get(k) or {} for k in ("evaluator", "data", "gt"))
    dirty = ev.get("dirty")
    out.update(card=os.path.basename(os.path.dirname(os.path.dirname(card_path))) + "/eval/card.json", split=data.get("split", V.UNKNOWN), frames=data.get("n"),
               frame_hash=data.get("frame_hash", V.UNKNOWN), manifest_hash=data.get("manifest_hash", V.UNKNOWN))
    out["evaluator"] = {"commit": ev.get("commit", V.UNKNOWN), "code_hash": ev.get("code_hash", V.UNKNOWN),
                        "dirty": V.UNKNOWN if dirty is None else (dirty if dirty in ([], V.UNKNOWN) else f"{len(dirty)} file(s) differed from the commit")}
    out["gt"] = {"version": gt.get("version", V.UNKNOWN), "signature": gt.get("signature", V.UNKNOWN)}
    return out


def experiment_of(run_id, folder):
    """the experiment card that names this run: the first experiments/*/experiment.md that holds the run id or the run folder name; unknown when none does"""
    base = os.path.join(P.root(), "experiments")
    if not os.path.isdir(base):
        return V.UNKNOWN
    for d in sorted(os.listdir(base)):
        md = os.path.join(base, d, "experiment.md")
        try:
            if os.path.isfile(md) and os.path.getsize(md) < 400_000:
                text = open(md, encoding="utf-8", errors="replace").read()
                if any(re.search(r"(?<![\w.])" + re.escape(x) + r"(?![\w])", text) for x in (run_id, folder) if x):
                    return f"experiments/{d}/experiment.md"
        except OSError:
            continue
    return V.UNKNOWN


def write(run_id, run_dir=None):
    """write the package; the list of earlier deletions is carried over, never replaced"""
    old = json.load(open(package_path(run_id), encoding="utf-8")) if os.path.isfile(package_path(run_id)) else {}
    pk = build(run_id, run_dir)
    pk["deleted"] = old.get("deleted", [])
    os.makedirs(os.path.dirname(package_path(run_id)), exist_ok=True)
    P.atomic_write(package_path(run_id), json.dumps(pk, indent=1) + "\n")
    return pk


def tag_exists(tag):
    if not tag or tag == V.UNKNOWN:
        return False
    return V.git(V.repo_root(), "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}") is not None


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024.0


def log_line(text):
    p = os.path.join(records_dir(), "CLEAN_LOG.md")
    new = not os.path.isfile(p)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        if new:
            f.write("# Clean-up log\n\nWritten by `autoresearch/tools/package.py clean|gt --apply` (D40): what was deleted, when, and how to get it back.\n\n")
        f.write(f"- {dt.datetime.now().isoformat(timespec='seconds')}: {text}\n")


def run_ids():
    d = P.path("run_records")
    return sorted(x for x in os.listdir(d) if os.path.isfile(os.path.join(d, x, "run.json"))) if os.path.isdir(d) else []


def related_unfinished(rid, now, fresh_s=600):
    """why a run must not be cleaned yet, from its own record and every record that belongs to it: <id>.<something> (the evaluations of a finished
    training), another id that maps to the same run folder, a heartbeat younger than fresh_s seconds; None when all is final and quiet"""
    folder = rid.replace(".", "_")
    for other in run_ids():
        if other == rid or other.startswith(rid + ".") or other.replace(".", "_") == folder:
            r = run_record(other)
            if not is_final(r):
                return f"the record {other} is not final (status {(r or {}).get('status')}, exit {(r or {}).get('exit')})"
            hb = os.path.join(P.path("run_records"), other, "heartbeat")
            if os.path.isfile(hb) and now - os.path.getmtime(hb) < fresh_s:
                return f"the record {other} has a fresh heartbeat"
    return None


def clean(apply=False, min_age_days=2.0, only=None, now=None):
    """-> list of (run id, verdict text, bytes to delete or deleted); verdict "delete" / "deleted" or "skip: <reason>" """
    now = now or time.time()
    min_age_days = max(min_age_days, MIN_AGE_DAYS)
    top = top3_runs()
    runs_root = os.path.join(P.root(), P.cfg_of(P.root()).get("cards", "runs"))
    out = []
    for rid in (only or run_ids()):
        r = run_record(rid)
        if r is None:
            out.append((rid, "skip: no run record", 0))
            continue
        run_dir = V.find_run_dir(rid)
        if not run_dir:
            continue
        files = derived_files(run_dir)
        if not files:
            continue
        name = os.path.basename(run_dir.rstrip("/\\"))
        total = sum(b for _, b, _ in files)
        why = None
        if top is None:
            why = "the best three are unknown (keep3.json missing or empty)"
        elif name in top or rid in top:
            why = "in the best three"
        elif os.path.islink(run_dir) or not inside(run_dir, runs_root):
            why = "the run folder is a link or lies outside the runs folder"
        elif not is_final(r):
            why = f"the run is not final (status {r.get('status')}, exit {r.get('exit')})"
        else:
            why = related_unfinished(rid, now)
        if why is None:
            if not tag_exists(r.get("tag")):
                why = f"its code tag {r.get('tag')} does not exist here: the files could not be rebuilt"
            elif V.final_ckpt(run_dir)[0] is None:
                why = "no checkpoint in the run folder: the files could not be rebuilt"
            elif any(now - m < min_age_days * 86400 for _, _, m in files):
                why = f"a derived file is younger than {min_age_days:g} days"
        if why:
            out.append((rid, "skip: " + why, total))
            continue
        if not apply:
            out.append((rid, "delete", total))
            continue
        pk = write(rid, run_dir)                                   # the package and the log first: they list what goes, also when the delete stops halfway
        pk["deleted"].append({"date": dt.date.today().isoformat(), "bytes": total, "files": [{"path": p, "bytes": b} for p, b, _ in files],
                              "how_to_rebuild": f"git tag {r.get('tag')} + GT {pk['gt']['version']} + the checkpoint {pk['model']['ckpt']}: run the evaluation again"})
        P.atomic_write(package_path(rid), json.dumps(pk, indent=1) + "\n")
        log_line(f"run {rid}: {len(files)} derived files, {human(total)} (list in results/{rid}/package.json); rebuild from tag {r.get('tag')}")
        order = sorted(files, key=lambda f: (os.path.basename(f[0]) not in FIRST, f[0]))
        for rel, _, _ in order:
            try:
                os.remove(os.path.join(run_dir, rel))
            except FileNotFoundError:
                pass
        for rel, _, _ in order:                                    # only the folders that held a deleted file, and only when they are empty
            d = os.path.dirname(os.path.join(run_dir, rel))
            while d != run_dir and os.path.isdir(d) and not os.listdir(d) and os.path.relpath(d, run_dir).split(os.sep)[0] not in NEVER_DIRS:
                os.rmdir(d)
                d = os.path.dirname(d)
        pk = write(rid, run_dir)                                   # rebuilt: the files that are left, and the list of deletions carried over
        out.append((rid, "deleted", total))
    return out


def gt_stores(root=None):
    root = root or os.path.join(V.repo_root(), "data", "autogt")
    return sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "index.json"))) if os.path.isdir(root) else []


def versions():
    p = os.path.join(V.repo_root(), "autogt", "versions.csv")
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def vnum(row):
    m = re.fullmatch(r"v(\d+)", row.get("version", ""))
    return int(m.group(1)) if m else -1


def sigs_of(gt_json):
    """the bare signatures a gt.json names: gt_signature as a string or a list, and the `earlier` ones"""
    vals = []
    for k in ("gt_signature", "earlier"):
        v = gt_json.get(k)
        vals += v if isinstance(v, list) else [v]
    return {str(v).split("_")[-1] for v in vals if isinstance(v, str) and v and not v.startswith("unknown")}


def gt_plan(root=None):
    """-> (latest row, [(signature, verdict text)]): the latest version of versions.csv stays, so does every store that a run which is not final names"""
    root = root or os.path.join(V.repo_root(), "data", "autogt")
    rows = versions()
    latest = max(rows, key=vnum)
    needed = {}
    for rid in run_ids():
        if is_final(run_record(rid)):
            continue
        rd = V.find_run_dir(rid)
        g = os.path.join(rd, "gt.json") if rd else ""
        if g and os.path.isfile(g):
            try:
                found = sigs_of(json.load(open(g, encoding="utf-8")))
            except (OSError, ValueError, AttributeError):
                found = set()
            for sig in found:
                needed.setdefault(sig, []).append(rid)
    plan = []
    by_sig = {x["signature"]: x for x in rows}
    for sig in gt_stores(root):
        row = by_sig.get(sig)
        path = os.path.join(root, sig)
        if sig == latest["signature"]:
            plan.append((sig, f"keep: the latest version ({latest['version']})"))
        elif sig in needed:
            plan.append((sig, f"keep: needed by the unfinished run(s) {', '.join(needed[sig][:4])}"))
        elif row is None:
            plan.append((sig, "keep: not in autogt/versions.csv, it cannot be regenerated from a tag"))
        elif os.path.islink(path) or not inside(path, root):
            plan.append((sig, "keep: a link, not a store folder of this data/autogt"))
        elif not tag_exists(row["tag"]):
            plan.append((sig, f"keep: the tag {row['tag']} does not exist here"))
        else:
            plan.append((sig, f"delete: {row['version']}, regenerated from the tag {row['tag']} and the options in its index.json"))
    return latest, plan


def gt_clean(apply=False, root=None):
    root = root or os.path.join(V.repo_root(), "data", "autogt")
    latest, plan = gt_plan(root)
    done = []
    for sig, text in plan:
        if text.startswith("delete") and apply:
            log_line(f"GT store {sig} ({text.split(':')[1].split(',')[0].strip()}) is being deleted; regenerate from the tag and the options of its index.json (git show autogt/<version>)")
            shutil.rmtree(os.path.join(root, sig))
            done.append(sig)
    return latest, plan, done


def selftest():
    import subprocess
    tmp = tempfile.mkdtemp()
    old_env = os.environ.get("AR_PROJECT")
    global tag_exists, versions
    real_tag, real_versions = tag_exists, versions
    try:
        proj = os.path.join(tmp, "proj")
        os.makedirs(os.path.join(proj, "records"))
        json.dump({"ledger": "records/LEDGER.csv", "run_records": "records/runs", "cards": "runs"}, open(os.path.join(proj, "project.json"), "w"))
        os.environ["AR_PROJECT"] = proj
        tags = {"tg/ok"}
        tag_exists = lambda t: t in tags       # noqa: E731

        def mk(rid, status="ok", exit_=0, tag="tg/ok", ck=True, old=10, extra=()):
            os.makedirs(os.path.join(proj, "records", "runs", rid), exist_ok=True)
            json.dump({"id": rid, "status": status, "exit": exit_, "tag": tag, "commit": "abc", "code_hash": "h"}, open(os.path.join(proj, "records", "runs", rid, "run.json"), "w"))
            d = os.path.join(proj, "runs", rid.replace(".", "_"))
            os.makedirs(os.path.join(d, "eval", "suite", "sub"), exist_ok=True)
            os.makedirs(os.path.join(d, "code"), exist_ok=True)
            json.dump({"gt_signature": "0.5_4_aaa"}, open(os.path.join(d, "gt.json"), "w"))
            json.dump({"seed": 0}, open(os.path.join(d, "config.json"), "w"))
            json.dump({"mAP": 0.5, "n": 3, "nested": {"x": 1}, "flag": True}, open(os.path.join(d, "eval", "metrics.json"), "w"))
            for rel in ("eval/scene_records.pkl", "eval/suite/sub/a.json", "eval/suite/frames.jsonl", "eval/pic.png", "code/keep.png", "autogt/ex/x.png", "eval_legacy_2026/old.png",
                        "eval/scene_records_legacy_x.pkl", "eval/scene_records_ctl_a.pkl", "eval/empty_marker/.keep", *extra):
                os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True)
                open(os.path.join(d, rel), "wb").write(b"x" * 100)
            if ck:
                open(os.path.join(d, "ckpt_00000010.pt"), "wb").write(b"w" * 10)
            for root, _, fs in os.walk(d):
                for f in fs:
                    os.utime(os.path.join(root, f), (time.time() - old * 86400,) * 2)
            return d
        a = mk("x.a_s0")
        pk = write("x.a_s0")
        assert pk["tier"] == V.UNKNOWN and pk["metrics"]["eval/metrics.json"] == {"mAP": 0.5, "n": 3}, "numbers only, no nested dict, no bool: " + str(pk["metrics"])
        want = ["eval/pic.png", "eval/scene_records.pkl", "eval/suite/frames.jsonl", "eval/suite/sub/a.json"]
        assert [x["path"] for x in pk["deep"]["files"]] == want and pk["deep"]["bytes"] == 400, pk["deep"]
        for rel, yes in (("x/y.MP4", False), ("eval/y.MP4", True), ("code/a.png", False), ("bundle/a.png", False), ("autogt/e/a.png", False), ("log.jsonl", False), ("ckpt_1.pt", False),
                         ("eval_2026/a.png", False), ("eval/a_legacy_b.pkl", False), ("eval/c_ctl_d.npz", False), ("eval/suite_x/a.png", True), ("eval/suite/a.json", True), ("a.png", True)):
            assert is_derived(rel) is yes, rel
        json.dump({"provenance": {"evaluator": {"commit": "c1", "code_hash": "h1", "dirty": ["a.py"]}, "data": {"split": "val", "n": 7, "frame_hash": "fh", "manifest_hash": "mh"},
                                  "gt": {"version": "v5", "signature": "sg"}}}, open(os.path.join(a, "eval", "card.json"), "w"))
        os.makedirs(os.path.join(proj, "experiments", "E1_demo"))
        open(os.path.join(proj, "experiments", "E1_demo", "experiment.md"), "w").write("# E1\n\narm a: run x.a_s0 and the folder x_a_s0b\n")
        os.makedirs(os.path.join(proj, "experiments", "E2_other"))
        open(os.path.join(proj, "experiments", "E2_other", "experiment.md"), "w").write("# E2\n\nx.a_s01 is another run\n")
        pe = write("x.a_s0")
        assert pe["evaluation"] == {"evaluator": {"commit": "c1", "code_hash": "h1", "dirty": "1 file(s) differed from the commit"}, "split": "val", "frames": 7, "frame_hash": "fh",
                                    "manifest_hash": "mh", "gt": {"version": "v5", "signature": "sg"}, "card": "x_a_s0/eval/card.json"}, pe["evaluation"]
        assert pe["experiment"] == "experiments/E1_demo/experiment.md", pe["experiment"]
        os.rename(os.path.join(proj, "experiments", "E1_demo"), os.path.join(proj, "experiments", "E9_demo"))
        assert experiment_of("x.a_s0", "x_a_s0") == "experiments/E9_demo/experiment.md" and experiment_of("x.a_s01", "x_a_s01") == "experiments/E2_other/experiment.md", "an id is not found inside a longer id"
        os.rename(os.path.join(proj, "experiments", "E9_demo"), os.path.join(proj, "experiments", "E1_demo"))
        assert experiment_of("x.zzz", "x_zzz") == V.UNKNOWN and evaluation_block(os.path.join(tmp, "none.json"))["split"] == V.UNKNOWN
        os.remove(os.path.join(a, "eval", "card.json"))
        # no keep3.json, an empty one, one without entries, a broken one: nothing is deleted
        for content in (None, '{"entries": []}', "{}", "{broken"):
            kp = os.path.join(proj, "records", "keep3.json")
            if content is None:
                assert not os.path.exists(kp)
            else:
                open(kp, "w").write(content)
            res = clean(only=["x.a_s0"])
            assert res and res[0][1].startswith("skip: the best three are unknown"), (content, res)
        # the best three are found by folder name, by run id, by a checkpoint path, by a card path (another spelling in `runs` must not expose a network)
        json.dump({"entries": [{"name": "n1", "key": "k1", "runs": ["x_keep_s0"], "checkpoints": [{"path": "runs/x_ckpt_s0/ckpt_00000010.pt"}], "card": "runs/x_card_s0/eval/metrics.json"},
                               {"name": "n2", "key": "x.id_s0", "runs": []}]}, open(os.path.join(proj, "records", "keep3.json"), "w"))
        assert top3_runs() == {"n1", "k1", "x_keep_s0", "x_ckpt_s0", "x_card_s0", "n2", "x.id_s0"}, top3_runs()
        mk("x.b_s0", status="stopped", exit_=75)
        mk("x.c_s0", tag="tg/missing")
        mk("x.d_s0", ck=False)
        mk("x.e_s0", old=0.5)
        mk("x.keep_s0")
        mk("x.ckpt_s0")
        mk("x.card_s0")
        mk("x.id_s0")
        mk("x.f_s0")
        mk("x.g_s0")
        mk("x.g_s0.eval", status="running", exit_=None)         # an evaluation of g that is still running: g must wait
        mk("x.h_s0")
        mk("x.h_s1", old=10)
        hb = os.path.join(proj, "records", "runs", "x.h_s1", "heartbeat")
        open(hb, "w").write("alive")
        mk("x.i_s0", status="ok", exit_=1)
        every = ["x.a_s0", "x.b_s0", "x.c_s0", "x.d_s0", "x.e_s0", "x.keep_s0", "x.ckpt_s0", "x.card_s0", "x.id_s0", "x.f_s0", "x.g_s0", "x.h_s0", "x.h_s1", "x.i_s0"]
        res = {rid: v for rid, v, _ in clean(only=every)}
        assert res["x.a_s0"] == "delete" and res["x.f_s0"] == "delete" and res["x.h_s0"] == "delete", res
        assert "not final" in res["x.b_s0"] and "does not exist" in res["x.c_s0"] and "no checkpoint" in res["x.d_s0"] and "younger" in res["x.e_s0"], res
        assert all("best three" in res[k] for k in ("x.keep_s0", "x.ckpt_s0", "x.card_s0", "x.id_s0")), res
        assert "x.g_s0.eval is not final" in res["x.g_s0"] and "fresh heartbeat" in res["x.h_s1"] and "not final" in res["x.i_s0"], res
        mk("x.m_s0", old=0.005)                                    # about seven minutes old
        assert "younger than" in clean(only=["x.m_s0"], min_age_days=0)[0][1], "a minimum age of one hour is enforced"
        assert clean(only=["x.e_s0"], min_age_days=0)[0][1] == "delete", "12 hours are old enough for the minimum"
        assert os.path.isfile(os.path.join(a, "eval", "pic.png")) and not os.path.exists(os.path.join(proj, "records", "CLEAN_LOG.md")), "a dry run deletes and logs nothing"
        out = clean(apply=True, only=["x.a_s0"])
        assert out == [("x.a_s0", "deleted", 400)], out
        assert not os.path.exists(os.path.join(a, "eval", "pic.png")) and not os.path.exists(os.path.join(a, "eval", "suite")), "derived files and the folders they emptied are gone"
        for keep in ("ckpt_00000010.pt", "eval/metrics.json", "code/keep.png", "autogt/ex/x.png", "gt.json", "config.json", "eval_legacy_2026/old.png", "eval/scene_records_legacy_x.pkl",
                     "eval/scene_records_ctl_a.pkl", "eval/empty_marker/.keep"):
            assert os.path.isfile(os.path.join(a, keep)), keep
        assert os.path.isdir(os.path.join(a, "code")) and os.path.isdir(os.path.join(a, "eval", "empty_marker")), "folders that held nothing derived stay"
        pk2 = json.load(open(package_path("x.a_s0"), encoding="utf-8"))
        assert len(pk2["deleted"]) == 1 and pk2["deleted"][0]["bytes"] == 400 and len(pk2["deleted"][0]["files"]) == 4 and "tg/ok" in pk2["deleted"][0]["how_to_rebuild"] and pk2["deep"]["files"] == [], pk2
        # a second cycle: an evaluation writes new derived files, clean again; the first list stays
        for rel in ("eval/new.png",):
            open(os.path.join(a, rel), "wb").write(b"y" * 50)
            os.utime(os.path.join(a, rel), (time.time() - 5 * 86400,) * 2)
        assert clean(apply=True, only=["x.a_s0"]) == [("x.a_s0", "deleted", 50)]
        pk3 = json.load(open(package_path("x.a_s0"), encoding="utf-8"))
        assert [d["bytes"] for d in pk3["deleted"]] == [400, 50] and write("x.a_s0")["deleted"] == pk3["deleted"], pk3["deleted"]
        assert clean(only=["x.a_s0"]) == [], "nothing left to delete"
        log = open(os.path.join(proj, "records", "CLEAN_LOG.md"), encoding="utf-8").read()
        assert log.count("run x.a_s0") == 2, log
        # the order: the files that mark a store as complete go first; an interrupted delete is logged and listed before it starts
        b = mk("x.j_s0")
        real_remove, seen = os.remove, []
        def failing_remove(p):
            seen.append(os.path.basename(p))
            if len(seen) == 2:
                raise PermissionError("locked")
            real_remove(p)
        os.remove = failing_remove
        try:
            clean(apply=True, only=["x.j_s0"])
            raise AssertionError("the failed delete must be visible")
        except PermissionError:
            pass
        finally:
            os.remove = real_remove
        assert seen[0] == "frames.jsonl", seen
        pj = json.load(open(package_path("x.j_s0"), encoding="utf-8"))
        assert len(pj["deleted"]) == 1 and "x.j_s0" in open(os.path.join(proj, "records", "CLEAN_LOG.md"), encoding="utf-8").read(), "listed and logged before the first delete"
        assert not os.path.exists(os.path.join(b, "eval", "suite", "frames.jsonl")), "frames.jsonl was the first to go"
        # a link as run folder is not walked
        mk("x.k_s0")
        shutil.move(os.path.join(proj, "runs", "x_k_s0"), os.path.join(tmp, "elsewhere"))
        try:
            os.symlink(os.path.join(tmp, "elsewhere"), os.path.join(proj, "runs", "x_k_s0"))
            assert "link" in clean(only=["x.k_s0"])[0][1], clean(only=["x.k_s0"])
        except OSError:
            pass                                                   # no symlinks on this file system
        # the tag guard against a real repository: an existing tag, a missing one
        tag_exists = real_tag
        repo = os.path.join(tmp, "repo")
        os.makedirs(repo)
        genv = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
        for args in (["init", "-q"], ["commit", "-q", "--allow-empty", "-m", "x"], ["tag", "t1"]):
            subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, env=genv)
        real_root = V.repo_root
        V.repo_root = lambda start=None: repo
        try:
            assert tag_exists("t1") is True and tag_exists("t2") is False and tag_exists("") is False and tag_exists(V.UNKNOWN) is False
        finally:
            V.repo_root = real_root
        tag_exists = lambda t: t in tags       # noqa: E731
        # GT stores: the highest version stays (an unordered versions.csv), a store that an unfinished run names (a list, `earlier`) stays, an old regenerable one goes
        gtroot = os.path.join(tmp, "gt")
        for sig in ("old1", "old2", "old3", "new", "orph", "lnk"):
            os.makedirs(os.path.join(gtroot, sig if sig != "lnk" else "lnk_real"))
            json.dump({"signature": sig}, open(os.path.join(gtroot, sig if sig != "lnk" else "lnk_real", "index.json"), "w"))
        rows = [{"version": "v3", "signature": "new", "tag": "tg/ok"}, {"version": "v1", "signature": "old1", "tag": "tg/ok"}, {"version": "v2", "signature": "old2", "tag": "tg/ok"},
                {"version": "v0", "signature": "old3", "tag": "tg/ok"}, {"version": "v10", "signature": "lnk", "tag": "tg/ok"}, {"version": "v9", "signature": "nine", "tag": "tg/ok"}]
        versions = lambda: rows       # noqa: E731
        os.makedirs(os.path.join(gtroot, "lnk_real"), exist_ok=True)
        json.dump({"gt_signature": ["0.5_4_old2", "0.5_4_zzz"], "earlier": ["0.5_4_old3", "unknown: the run has checkpoints from before gt.json existed"]}, open(os.path.join(proj, "runs", "x_b_s0", "gt.json"), "w"))
        latest, plan = gt_plan(gtroot)
        got = dict(plan)
        assert latest["version"] == "v10", latest              # the highest number, not the last row
        assert got["new"].startswith("delete") and got["old1"].startswith("delete"), got
        assert got["old2"].startswith("keep: needed by") and got["old3"].startswith("keep: needed by") and got["orph"].startswith("keep: not in"), got
        tags.discard("tg/ok")
        assert dict(gt_plan(gtroot)[1])["old1"].startswith("keep: the tag"), "no tag, no deletion"
        tags.add("tg/ok")
        n_before = len(open(os.path.join(proj, "records", "CLEAN_LOG.md"), encoding="utf-8").read().splitlines())
        gt_clean(False, gtroot)
        assert len(open(os.path.join(proj, "records", "CLEAN_LOG.md"), encoding="utf-8").read().splitlines()) == n_before and os.path.isdir(os.path.join(gtroot, "old1")), "a dry run changes nothing"
        _, _, done = gt_clean(True, gtroot)
        assert sorted(done) == ["new", "old1"] and not os.path.exists(os.path.join(gtroot, "old1")) and os.path.isdir(os.path.join(gtroot, "old2")), done
        assert "GT store old1" in open(os.path.join(proj, "records", "CLEAN_LOG.md"), encoding="utf-8").read(), "logged"
        print("package selftest passed")
    finally:
        tag_exists, versions = real_tag, real_versions
        if old_env is None:
            os.environ.pop("AR_PROJECT", None)
        else:
            os.environ["AR_PROJECT"] = old_env
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["package", "show", "clean", "gt"])
    ap.add_argument("run", nargs="?")
    ap.add_argument("--run-dir")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-age-days", type=float, default=2.0)
    ap.add_argument("--runs", help="comma list of run ids (clean)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "package" and a.run:
        pk = write(a.run, a.run_dir)
        print(f"written: {package_path(a.run)} (tier {pk['tier']}, GT {pk['gt']['version']}, bundle {pk['bundle']['code_hash']}, {len(pk['deep']['files'])} derived files, {human(pk['deep']['bytes'])})")
    elif a.cmd == "show" and a.run:
        print(open(package_path(a.run), encoding="utf-8").read())
    elif a.cmd == "clean":
        res = clean(a.apply, a.min_age_days, a.runs.split(",") if a.runs else None)
        for rid, v, b in res:
            print(f"{rid:34} {v:70} {human(b)}")
        print(f"{'deleted' if a.apply else 'would delete'}: {human(sum(b for _, v, b in res if v in ('delete', 'deleted')))} in {sum(1 for _, v, _ in res if v in ('delete', 'deleted'))} run(s); {sum(1 for _, v, _ in res if v.startswith('skip'))} skipped")
        print("a rebuild needs the code tag: check_git.py lists the tags that are not on origin yet")
    elif a.cmd == "gt":
        latest, plan, done = gt_clean(a.apply)
        print(f"latest GT version: {latest['version']} ({latest['signature']})")
        for sig, text in plan:
            print(f"  {sig:14} {text}")
        print(("deleted: " + ", ".join(done)) if a.apply else "dry run: nothing deleted (--apply)")
    else:
        ap.error("package <run id> | show <run id> | clean [--apply] | gt [--apply] | --selftest")


if __name__ == "__main__":
    sys.exit(main())
