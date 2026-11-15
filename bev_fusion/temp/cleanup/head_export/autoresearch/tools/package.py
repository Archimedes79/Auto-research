"""The result package of a run and the clean-up of derived data (D40: derived data may be deleted, of the GT only the latest version stays). Stdlib only.

    python autoresearch/tools/package.py package <run id> [--run-dir d]   write <problem>/results/<id>/package.json (a text record, in git)
    python autoresearch/tools/package.py show <run id>                     print the package
    python autoresearch/tools/package.py clean [--apply] [--min-age-days 2] [--runs a,b]
                                                                            derived data of runs outside the best three: dry run lists and sizes it, --apply deletes
    python autoresearch/tools/package.py gt [--apply]                      GT stores of versions older than the latest: dry run, --apply deletes
    python autoresearch/tools/package.py --selftest

A project whose project.json has "package": {"split_record": "<command>"} (the placeholders {run_dir} and {out_dir} are paths relative to the problem folder, which is also the
working folder of the command, bare or inside quotes; bev_fusion: `python3 training/splits.py split-json {run_dir} {out_dir}`) gets results/<id>/split.json (the scene lists of the
run) in every package. The command writes into a temp folder, the file moves to results/<id>/ only after it was checked and before package.json is written: `package` and `clean` (also
its dry run) refuse, and leave results/ as it was, when the command fails, leaves no json object or the run folder is missing. A project without the key is unaffected.

The result folder <problem>/results/<run id>/ holds the package: package.json, provenance.json and the copy of the model bundle (bundle/, not in git:
the code is in the tag). The package is the one place that says how a run was made and what came out: the code tag and commit, the GT version and
signature, the model bundle hash (provenance.py manifest), the numbers (the numbers of eval/metrics.json and latency.json: top level and the metrics / latency dicts, with the hash of the
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
gt also looks at the lane GT stores (data/autogt/lane/<lane signature>, versions in autogt/lane_versions.csv, l1, l2, ...): the latest lane version stays, per rig (lane_center = Spurmitte,
bev_fusion = BEV Fusion; when its code makes the GT of an older version, `same_gt_as`, the store of that one), so does a store that a run which
is not final names in its gt.json (`lane_signature`: NO trainer writes it today, so the lane stores are not protected by run records and the dry run prints a WARNING), an older one
goes when its tag autogt/lane-lN exists (it is made again by `python -m autogt generate-lane`).
gt keeps the latest version of autogt/versions.csv (the highest number; when its code makes the GT of an older version, `same_gt_as`, the store of that
one: they share a folder) and every store that is still needed. A run records the CODE signature (0.5_4_6a7aa7864a, v9), the folder is the STORE's (c643a7539d, v8): every signature
goes through provenance.canonical_sig first. data/autogt is shared by all problem folders, so the run records of EVERY <repo>/*/project.json are read. Needed = named in the gt.json
(also the `earlier` ones; a resume needs them) of a run that is not final; or of a network of the best three (keep3.json); or of a FINAL run that has no result package yet
(the clean-up writes the package first: `package.py package <id>`). An older store is deleted only when its tag autogt/vN exists (it can be regenerated from the tag and the options in
its index.json) and it is a real folder inside data/autogt. A network trained on an older GT cannot be evaluated again against its own GT
after that store is gone: it is regenerated first. Deletions are appended to <records folder>/CLEAN_LOG.md.
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
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


def records_dir(root=None):
    return os.path.dirname(P.path("ledger") if root is None else os.path.join(root, P.cfg_of(root)["ledger"]))


def run_records_path(root=None):
    return P.path("run_records") if root is None else os.path.join(root, P.cfg_of(root)["run_records"])


def run_record(run_id, root=None):
    p = os.path.join(run_records_path(root), run_id, "run.json")
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None


def package_path(run_id):
    return os.path.join(V.results_dir(run_id), "package.json")


def is_final(r):
    return bool(r) and r.get("status") in FINAL_STATES and r.get("exit") == 0


def keep3_entries(root=None):
    """the entries of keep3.json beside the ledger of a problem folder (default: the current one); None when the record is missing, unreadable or holds no entry"""
    p = os.path.join(records_dir(root), "keep3.json")
    if not os.path.isfile(p):
        return None
    try:
        entries = json.load(open(p, encoding="utf-8")).get("entries") or []
    except (OSError, ValueError, AttributeError):
        return None
    return [e for e in entries if isinstance(e, dict)] or None


def entry_names(e):
    """folder names and ids of one best-three entry: its name, key, runs, and the run folder named in its checkpoint and card paths (runs/<folder>/...)"""
    out = set()
    out.update(x for x in (e.get("runs") or []) if isinstance(x, str))
    out.update(x for x in (e.get("key"), e.get("name")) if isinstance(x, str))
    paths = [c.get("path", "") for c in (e.get("checkpoints") or []) if isinstance(c, dict)] + [e.get("card") or ""]
    for pa in paths:
        parts = str(pa).replace("\\", "/").split("/")
        if len(parts) >= 2 and parts[0] == "runs":
            out.add(parts[1])
    out.discard("")
    return out


def top3_runs():
    """folder names and ids of the best three (keep3.json beside the ledger); None when the record is missing, unreadable or holds no entry (then nothing is deleted)"""
    entries = keep3_entries()
    out = set().union(*(entry_names(e) for e in entries)) if entries else set()
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
    """the numbers of a json file (bool excluded), at most cap keys, sorted; {} when it is not readable. Top-level numbers, and the numbers of the dicts `metrics` and
    `latency` as 'metrics.<key>' / 'latency.<key>' (that is where bev_suite writes them: eval/metrics.json holds {"metrics": {...}, "better": {...}} and eval/latency.json
    {"latency": {...}})"""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(d, dict):
        return {}
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)  # noqa: E731
    flat = {k: v for k, v in d.items() if num(v)}
    for key in ("metrics", "latency"):
        if isinstance(d.get(key), dict):
            flat.update({f"{key}.{k}": v for k, v in d[key].items() if num(v)})
    return dict(sorted(flat.items())[:cap])


def build(run_id, run_dir=None):
    """the package dict of a run (nothing is written but the provenance manifest, which is made first when it is missing or of an earlier state of the run)"""
    r = run_record(run_id)
    if r is None:
        sys.exit(f"no run record for {run_id}")
    run_dir = run_dir or V.find_run_dir(run_id)
    prov_file = os.path.join(V.results_dir(run_id), "provenance.json")
    prov = V.manifest(run_id, run_dir)                       # the stored file unless it is of an earlier state of the run (resumed: a newer checkpoint, another status); then rebuilt
    have_dir = bool(run_dir and os.path.isdir(run_dir))
    top = top3_runs()
    name = os.path.basename(run_dir.rstrip("/\\")) if have_dir else run_id.replace(".", "_")
    tg = V._get(prov, ("subject", "train_gt")) or V._get(prov, ("gt",)) or {}
    if tg.get("key") not in (None, V.UNKNOWN) and tg.get("version") in (None, V.UNKNOWN):      # a manifest of before the store was named canonically (v9's key, the store of v8) or of a deleted store
        tg = V.gt_block(tg["key"], V.repo_root())
    pk = {"schema": 1, "made": dt.datetime.now().isoformat(timespec="seconds"), "run": run_id, "run_dir": os.path.relpath(run_dir, P.root()).replace("\\", "/") if have_dir else V.UNKNOWN,
          "status": {"status": r.get("status"), "exit": r.get("exit"), "final": is_final(r)},
          "tier": "top3" if top is not None and (name in top or run_id in top) else ("normal" if top is not None else V.UNKNOWN),
          "code": {"tag": r.get("tag") or V.UNKNOWN, "commit": r.get("commit") or V.UNKNOWN, "code_hash": r.get("code_hash") or V.UNKNOWN},
          "gt": {"version": tg.get("version") or V.UNKNOWN, "signature": tg.get("signature") or V.UNKNOWN},
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
    """the experiment card that names this run: the first experiment.md (folders of project.json "experiment_dirs" in order, open cards first) that holds the run id
    or the run folder name; path from the problem folder; unknown when none does"""
    for base in P.experiment_dirs():
        for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            md = os.path.join(base, d, "experiment.md")
            try:
                if os.path.isfile(md) and os.path.getsize(md) < 400_000:
                    text = open(md, encoding="utf-8", errors="replace").read()
                    if any(re.search(r"(?<![\w.])" + re.escape(x) + r"(?![\w])", text) for x in (run_id, folder) if x):
                        return os.path.relpath(md, P.root()).replace("\\", "/")
            except OSError:
                continue
    return V.UNKNOWN


def split_hook(run_id, run_dir):
    """run the project's split-record command (project.json "package" -> "split_record") into a fresh temp folder; -> (folder, file) with a split.json that holds a json
    object, or None when the project has no hook. SystemExit (nothing written below results/, the temp folder removed) when there is no run folder, the command fails or
    it leaves no json object: a run without its split record gets no package. A file from an earlier package is never looked at"""
    cmd = (P.cfg().get("package") or {}).get("split_record")
    if not cmd:
        return None
    if not run_dir or not os.path.isdir(run_dir):
        sys.exit(f"{run_id}: no run folder, so no split record (project.json package.split_record): the package is not written")
    tmp = tempfile.mkdtemp(prefix="split_")
    try:
        def rel(p):
            try:
                return os.path.relpath(p, P.root()).replace("\\", "/")
            except ValueError:                                     # another drive
                return p
        argv = [t.replace("{run_dir}", rel(run_dir)).replace("{out_dir}", rel(tmp)) for t in shlex.split(cmd)]      # substituted after the split: a path with a space stays one argument, quotes in the template work
        try:
            r = subprocess.run(argv, cwd=P.root(), capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as e:
            sys.exit(f"{run_id}: the split record command {cmd!r} cannot run ({e}): the package is not written")
        path = os.path.join(tmp, "split.json")
        try:
            ok = r.returncode == 0 and isinstance(json.load(open(path, encoding="utf-8")), dict)
        except (OSError, ValueError):
            ok = False
        if not ok:
            sys.exit(f"{run_id}: the split record command {cmd!r} failed (exit {r.returncode}) or wrote no json object: the package is not written; "
                     + (r.stderr or r.stdout).strip()[-300:])
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return tmp, path


def write(run_id, run_dir=None):
    """write the package; the list of earlier deletions is carried over, never replaced. The split record (if the project has one) is made first, in a temp folder, and moved
    to results/<id>/split.json when the package is built: no record, no package, and a refused run leaves nothing behind in results/"""
    old = json.load(open(package_path(run_id), encoding="utf-8")) if os.path.isfile(package_path(run_id)) else {}
    hook = split_hook(run_id, run_dir or V.find_run_dir(run_id))
    try:
        pk = build(run_id, run_dir)
        if hook:
            dest = os.path.join(V.results_dir(run_id), "split.json")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            P.atomic_write(dest, open(hook[1], encoding="utf-8").read())
            pk["split_record"] = {"file": "split.json", "hash": V.file_hash(dest)}
    finally:
        if hook:
            shutil.rmtree(hook[0], ignore_errors=True)
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


def run_ids(root=None):
    d = run_records_path(root)
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
            try:
                hook = split_hook(rid, run_dir)                    # the dry run asks what --apply asks (the command writes into a temp folder only)
                if hook:
                    shutil.rmtree(hook[0], ignore_errors=True)
            except SystemExit as e:
                out.append((rid, f"skip: {e}", total))
                continue
            out.append((rid, "delete", total))
            continue
        try:
            pk = write(rid, run_dir)                               # the package and the log first: they list what goes, also when the delete stops halfway
        except SystemExit as e:                                    # e.g. the split record cannot be written: nothing of this run is deleted
            out.append((rid, f"skip: {e}", total))
            continue
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
        try:
            write(rid, run_dir)                                    # rebuilt: the files that are left, and the list of deletions carried over
        except SystemExit as e:                                    # the files are gone and listed in the package written above: the other runs go on
            print(f"warning: {rid}: deleted and listed, but the package could not be refreshed: {e}", file=sys.stderr)
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


def sigs_of(gt_json, keys=("gt_signature", "earlier")):
    """the bare signatures a gt.json names: gt_signature as a string or a list, and the `earlier` ones (or the other `keys`)"""
    vals = []
    for k in keys:
        v = gt_json.get(k)
        vals += v if isinstance(v, list) else [v]
    return {str(v).split("_")[-1] for v in vals if isinstance(v, str) and v and not v.startswith("unknown")}


def problem_roots(top=None):
    """the problem folders of the repository (<repo>/*/project.json): data/autogt is shared by all of them, so a clean-up of it reads the runs of every one"""
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(glob.escape(top or V.repo_root()), "*", "project.json")))


def gt_names(root, name, keys, rows):
    """the canonical signatures (V.canonical_sig over `rows`) that the gt.json of the run folder `name` of a problem folder names under `keys`; set() when it has none"""
    rd = V.find_run_dir(name, root)
    g = os.path.join(rd, "gt.json") if rd else ""
    try:
        return {V.canonical_sig(s, rows) for s in sigs_of(json.load(open(g, encoding="utf-8")), keys)} if os.path.isfile(g) else set()
    except (OSError, ValueError, AttributeError):
        return set()


def gt_users(rows, keys=("gt_signature", "earlier"), problems=None):
    """canonical signature -> {"unfinished": [...], "unpackaged": [...]} from the run records of every problem folder: the runs that are not final and name the store in their
    gt.json under one of `keys` (a resume needs it; an unreadable record counts as not final), and the final runs that name it and have no results/<id>/package.json yet
    (the clean-up writes the result package first). Entries read "<problem folder>:<run id>"."""
    out = {}
    for root in problem_roots() if problems is None else problems:
        who, results = os.path.basename(root.rstrip("/\\")), P.cfg_of(root).get("results", "results")
        for rid in run_ids(root):
            try:
                final = is_final(run_record(rid, root))
            except (OSError, ValueError):
                final = False
            packaged = os.path.isfile(os.path.join(root, results, rid, "package.json"))
            for sig in gt_names(root, rid, keys, rows):
                slot = out.setdefault(sig, {"unfinished": [], "unpackaged": []})
                if not final:
                    slot["unfinished"].append(f"{who}:{rid}")
                elif not packaged:
                    slot["unpackaged"].append(f"{who}:{rid}")
    return out


def top3_users(rows, problems=None):
    """canonical signature -> ["<problem folder>:<entry>"] of the networks of the best three (keep3.json of every problem folder) that were trained on that store: the run folders
    an entry names (entry_names) -> their gt.json, the `earlier` GT too"""
    out = {}
    for root in problem_roots() if problems is None else problems:
        who = os.path.basename(root.rstrip("/\\"))
        for e in keep3_entries(root) or []:
            sigs = set().union(*(gt_names(root, n, ("gt_signature", "earlier"), rows) for n in entry_names(e)))
            for rec in [e] + [c for c in (e.get("checkpoints") or []) if isinstance(c, dict)]:          # a keep3 record that carries the signature itself counts too
                sigs |= {V.canonical_sig(s, rows) for s in sigs_of(rec, ("gt_signature",))}
            for sig in sigs:
                out.setdefault(sig, []).append(f"{who}:{e.get('name') or e.get('key')}")
    return out


def lane_versions():
    p = os.path.join(V.repo_root(), "autogt", "lane_versions.csv")
    if not os.path.isfile(p):
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def lane_stores(root):
    return sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "index.json"))) if os.path.isdir(root) else []


LANE_GAP = ("WARNING: lane stores cannot be protected by run records: no trainer writes the lane GT it read into its run folder (a gt.json has no lane_signature), so a lane "
            "store is safe only as the latest version of its rig; a job that reads an older one (a running lane training, a best-three network) is not seen. Check the jobs first")


def lane_notes(plan):
    """the warnings that belong to a lane plan: LANE_GAP whenever it would delete a store"""
    return [LANE_GAP] if any(text.startswith("delete") for _, text in plan) else []


def lane_plan(root=None, problems=None):
    """-> (latest lane row or None, [(signature, verdict text)]) for data/autogt/lane/: the latest lane version stays, for each rig (lane_center = Spurmitte, bev_fusion = BEV Fusion;
    a row without `rig` is lane_center), and when its code makes the GT of an older version (`same_gt_as`) that version's store is the one that stays; so does a store that a
    run which is not final names (`lane_signature` in its gt.json, in any problem folder: written by no trainer today, see LANE_GAP); an older one goes when its tag exists"""
    root = root or os.path.join(V.repo_root(), "data", "autogt", "lane")
    rows = lane_versions()
    if not rows:
        return None, []
    num = lambda r: int(re.fullmatch(r"l(\d+)", r.get("version", "")).group(1)) if re.fullmatch(r"l(\d+)", r.get("version", "")) else -1      # noqa: E731
    rig = lambda r: r.get("rig") or "lane_center"                                                                                         # noqa: E731
    latest = max(rows, key=num)
    needed = {s: u["unfinished"] for s, u in gt_users(rows, ("lane_signature",), problems).items() if u["unfinished"]}
    by_sig = {x["signature"]: x for x in rows}
    keep_sig = {}                                    # store signature -> the latest row (per rig) whose store it is
    for r_ in sorted({rig(x) for x in rows}):
        last = max((x for x in rows if rig(x) == r_), key=num)
        shared, seen = last, set()
        while (shared.get("same_gt_as") or "") and shared["version"] not in seen:
            seen.add(shared["version"])
            nxt = next((x for x in rows if x["version"] == shared["same_gt_as"] and rig(x) == r_), None)
            if nxt is None:
                break
            shared = nxt
        keep_sig[shared["signature"]] = (last, shared)
    plan = []
    for sig in lane_stores(root):
        row, path = by_sig.get(sig), os.path.join(root, sig)
        if sig in keep_sig:
            last, shared = keep_sig[sig]
            plan.append((sig, f"keep: the latest lane version ({last['version']}, {rig(last)})" + (f", its code makes the GT of {shared['version']}, the version of this store" if shared is not last else "")))
        elif sig in needed:
            plan.append((sig, f"keep: needed by the unfinished run(s) {', '.join(needed[sig][:4])}"))
        elif row is None:
            plan.append((sig, "keep: not in autogt/lane_versions.csv, it cannot be regenerated from a tag"))
        elif os.path.islink(path) or not inside(path, root):
            plan.append((sig, "keep: a link, not a store folder of this data/autogt/lane"))
        elif not tag_exists(row["tag"]):
            plan.append((sig, f"keep: the tag {row['tag']} does not exist here"))
        else:
            plan.append((sig, f"delete: lane {row['version']}, regenerated from the tag {row['tag']} (python -m autogt generate-lane in a worktree of it)"))
    return latest, plan


def lane_clean(apply=False, root=None, problems=None):
    root = root or os.path.join(V.repo_root(), "data", "autogt", "lane")
    latest, plan = lane_plan(root, problems)
    done = []
    for sig, text in plan:
        if text.startswith("delete") and apply:
            log_line(f"lane GT store {sig} ({text.split(':')[1].split(',')[0].strip()}) is being deleted; regenerate from its tag autogt/lane-lN with `python -m autogt generate-lane`")
            shutil.rmtree(os.path.join(root, sig))
            done.append(sig)
    return latest, plan, done


def gt_plan(root=None, problems=None):
    """-> (latest row, [(signature, verdict text)]): the latest version of versions.csv stays, so does every store that is still needed, by the run records of EVERY problem folder
    (`problems`, default problem_roots()): a run that is not final, a network of the best three, a final run that has no result package yet. Signatures are compared as stores"""
    root = root or os.path.join(V.repo_root(), "data", "autogt")
    rows = versions()
    latest = max(rows, key=vnum)
    by_version = {x["version"]: x for x in rows}
    shared, seen = latest, set()                  # a version whose code makes the GT of an older one (`same_gt_as`) shares that one's store folder
    while shared.get("same_gt_as") in by_version and shared["version"] not in seen:
        seen.add(shared["version"])
        shared = by_version[shared["same_gt_as"]]
    problems = problem_roots() if problems is None else problems
    users, best = gt_users(rows, problems=problems), top3_users(rows, problems)
    plan = []
    by_sig = {x["signature"]: x for x in rows}
    for sig in gt_stores(root):
        row = by_sig.get(sig)
        path = os.path.join(root, sig)
        use = users.get(sig, {})
        if sig == shared["signature"]:
            plan.append((sig, f"keep: the latest version ({latest['version']})" + (f", its code makes the GT of {shared['version']}, the version of this store" if shared is not latest else "")))
        elif use.get("unfinished"):
            plan.append((sig, f"keep: needed by the unfinished run(s) {', '.join(use['unfinished'][:4])}"))
        elif sig in best:
            plan.append((sig, f"keep: the best three network(s) {', '.join(best[sig][:4])} trained on it"))
        elif use.get("unpackaged"):
            plan.append((sig, f"keep: used by the final run(s) {', '.join(use['unpackaged'][:4])}"
                              + (f" and {len(use['unpackaged']) - 4} more" if len(use["unpackaged"]) > 4 else "") + " that have no result package yet (package.py package <id> first)"))
        elif row is None:
            plan.append((sig, "keep: not in autogt/versions.csv, it cannot be regenerated from a tag"))
        elif os.path.islink(path) or not inside(path, root):
            plan.append((sig, "keep: a link, not a store folder of this data/autogt"))
        elif not tag_exists(row["tag"]):
            plan.append((sig, f"keep: the tag {row['tag']} does not exist here"))
        else:
            plan.append((sig, f"delete: {row['version']}, regenerated from the tag {row['tag']} and the options in its index.json"))
    return latest, plan


def gt_clean(apply=False, root=None, problems=None):
    root = root or os.path.join(V.repo_root(), "data", "autogt")
    latest, plan = gt_plan(root, problems)
    done = []
    for sig, text in plan:
        if text.startswith("delete") and apply:
            log_line(f"GT store {sig} ({text.split(':')[1].split(',')[0].strip()}) is being deleted; regenerate from the tag and the options of its index.json (git show autogt/<version>)")
            shutil.rmtree(os.path.join(root, sig))
            done.append(sig)
    return latest, plan, done


def selftest():
    tmp = tempfile.mkdtemp()
    old_env = os.environ.get("AR_PROJECT")
    global tag_exists, versions, lane_versions, problem_roots
    real_tag, real_versions, real_lane, real_problems = tag_exists, versions, lane_versions, problem_roots
    try:
        proj = os.path.join(tmp, "proj")
        os.makedirs(os.path.join(proj, "records"))
        json.dump({"ledger": "records/LEDGER.csv", "run_records": "records/runs", "cards": "runs"}, open(os.path.join(proj, "project.json"), "w"))
        os.environ["AR_PROJECT"] = proj
        problem_roots = lambda top=None: [proj]       # noqa: E731  (the real one globs the repository's problem folders: the tests below pass their own list where it matters)
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
            json.dump({"mAP": 0.5, "n": 3, "nested": {"x": 1}, "flag": True, "metrics": {"seg/iou": 0.4, "txt": "a"}, "better": {"seg/iou": "higher"}}, open(os.path.join(d, "eval", "metrics.json"), "w"))
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
        assert pk["tier"] == V.UNKNOWN and pk["metrics"]["eval/metrics.json"] == {"mAP": 0.5, "n": 3, "metrics.seg/iou": 0.4}, "numbers only (top level and the metrics / latency dicts), no other nested dict, no bool: " + str(pk["metrics"])
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
        versions = lambda: rows + [{"version": "v11", "signature": "eleven", "tag": "tg/ok", "same_gt_as": "v3"}]       # noqa: E731
        latest2, plan2 = gt_plan(gtroot)               # v11 makes the GT of v3: the store of v3 stays, it is the one the latest version reads
        assert latest2["version"] == "v11" and dict(plan2)["new"].startswith("keep: the latest version (v11)") and dict(plan2)["old1"].startswith("delete"), plan2
        versions = lambda: rows       # noqa: E731
        n_before = len(open(os.path.join(proj, "records", "CLEAN_LOG.md"), encoding="utf-8").read().splitlines())
        gt_clean(False, gtroot)
        assert len(open(os.path.join(proj, "records", "CLEAN_LOG.md"), encoding="utf-8").read().splitlines()) == n_before and os.path.isdir(os.path.join(gtroot, "old1")), "a dry run changes nothing"
        _, _, done = gt_clean(True, gtroot)
        assert sorted(done) == ["new", "old1"] and not os.path.exists(os.path.join(gtroot, "old1")) and os.path.isdir(os.path.join(gtroot, "old2")), done
        assert "GT store old1" in open(os.path.join(proj, "records", "CLEAN_LOG.md"), encoding="utf-8").read(), "logged"
        # the second card folder (project.json experiment_dirs): a card there is found, the open one wins when both name the run
        pj0 = json.load(open(os.path.join(proj, "project.json")))
        os.makedirs(os.path.join(proj, "results", "experiments", "E5_done"))
        open(os.path.join(proj, "results", "experiments", "E5_done", "experiment.md"), "w").write("# E5\n\nruns x.fin_s0 and x.a_s0\n")
        assert experiment_of("x.fin_s0", "x_fin_s0") == V.UNKNOWN, "without the key only experiments/ is searched"
        json.dump(dict(pj0, experiment_dirs=["experiments", "results/experiments"]), open(os.path.join(proj, "project.json"), "w"))
        assert experiment_of("x.fin_s0", "x_fin_s0") == "results/experiments/E5_done/experiment.md" and experiment_of("x.a_s0", "x_a_s0") == "experiments/E1_demo/experiment.md", "both folders, open first"
        json.dump(pj0, open(os.path.join(proj, "project.json"), "w"))
        # the split record: a project whose project.json names a hook gets results/<id>/split.json in the package and no package when the hook fails (markers in the run folder steer the hook)
        hook = os.path.join(tmp, "split_hook.py")
        open(hook, "w").write(
            "import json, os, sys\n"
            "if len(sys.argv) != 3:\n    sys.exit(4)\n"
            "run, out = sys.argv[1:3]\n"
            "has = lambda m: os.path.exists(os.path.join(run, m))\n"
            "cnt = os.path.join(run, 'calls')\n"
            "n = int(open(cnt).read()) + 1 if os.path.exists(cnt) else 1\n"
            "open(cnt, 'w').write(str(n))\n"
            "if has('no_split') or (has('fail_from') and n >= int(open(os.path.join(run, 'fail_from')).read())):\n    sys.exit(3)\n"
            "if has('list'):\n    json.dump([1, 2], open(os.path.join(out, 'split.json'), 'w'))\n"
            "elif not has('silent'):\n"
            "    json.dump({'run': os.path.basename(run), 'seed': json.load(open(os.path.join(run, 'config.json')))['seed']}, open(os.path.join(out, 'split.json'), 'w'))\n")
        bare = f"{shlex.quote(sys.executable)} {shlex.quote(hook)} {{run_dir}} {{out_dir}}"
        quoted = f"{shlex.quote(sys.executable)} {shlex.quote(hook)} \"{{run_dir}}\" '{{out_dir}}'"
        with_hook = lambda cmd: json.dump(dict(pj0, package={"split_record": cmd}), open(os.path.join(proj, "project.json"), "w"))     # noqa: E731
        stmp = os.path.join(tmp, "tt")
        os.makedirs(stmp)
        tempfile.tempdir = stmp                                    # the hook's temp folders go here: none may be left behind
        s0 = mk("x.s_s0")
        assert "split_record" not in write("x.s_s0") and not os.path.exists(os.path.join(proj, "results", "x.s_s0", "split.json")), "a project without the key is unaffected"
        with_hook(bare)
        pks = write("x.s_s0")
        sj = os.path.join(proj, "results", "x.s_s0", "split.json")
        assert json.load(open(sj)) == {"run": "x_s_s0", "seed": 0} and pks["split_record"] == {"file": "split.json", "hash": V.file_hash(sj)}, pks.get("split_record")
        assert os.listdir(stmp) == [], "the temp folder of the hook is removed"
        before, before_sj = open(package_path("x.s_s0"), "rb").read(), open(sj, "rb").read()
        for why in ("no_split", "silent", "list", "no run folder"):
            for m in ("no_split", "silent", "list"):
                if os.path.exists(os.path.join(s0, m)):
                    os.remove(os.path.join(s0, m))
            if why == "no run folder":
                shutil.move(s0, s0 + "_away")
            else:
                open(os.path.join(s0, why), "w").close()           # 'silent': the hook exits 0 and writes nothing; the split.json of the package above is still there and must not count
            try:
                write("x.s_s0")
                raise AssertionError(f"the package must be refused: {why}")
            except SystemExit as e:
                assert "package is not written" in str(e), (why, str(e))
        shutil.move(s0 + "_away", s0)
        assert open(package_path("x.s_s0"), "rb").read() == before and open(sj, "rb").read() == before_sj, "a refused package leaves package.json and split.json as they were"
        assert os.listdir(stmp) == [], "a refused run leaves no temp folder"
        p0 = mk("x.p_s0")                                          # refused before anything is made: no provenance.json, no bundle/ in results/
        open(os.path.join(p0, "no_split"), "w").close()
        try:
            write("x.p_s0")
            raise AssertionError("must be refused")
        except SystemExit:
            assert not os.path.exists(os.path.join(proj, "results", "x.p_s0")), os.listdir(os.path.join(proj, "results", "x.p_s0"))
        # a path with a space is one argument, with the placeholders bare or inside quotes of the template
        mk("x.sp s0")
        for cmd in (bare, quoted):
            with_hook(cmd)
            write("x.sp s0")
            spj = os.path.join(proj, "results", "x.sp s0", "split.json")
            assert json.load(open(spj)) == {"run": "x_sp s0", "seed": 0}, cmd
            os.remove(spj)
        with_hook(bare)
        # clean: the dry run and --apply both skip a run without its split record and delete nothing of it; a failure of the second write does not stop the other runs
        u = mk("x.u_s0")
        open(os.path.join(u, "no_split"), "w").close()
        for apply in (False, True):
            res = clean(apply=apply, only=["x.u_s0"])
            assert res[0][1].startswith("skip: ") and "split record" in res[0][1] and os.path.isfile(os.path.join(u, "eval", "pic.png")), (apply, res)
        w = mk("x.w_s0")
        open(os.path.join(w, "fail_from"), "w").write("2")        # the first write (before the delete) works, the second (after it) fails
        mk("x.v_s0")
        got = {rid: v for rid, v, _ in clean(apply=True, only=["x.u_s0", "x.w_s0", "x.v_s0"])}
        assert got["x.u_s0"].startswith("skip: ") and got["x.w_s0"] == "deleted" and got["x.v_s0"] == "deleted", got
        assert not os.path.exists(os.path.join(w, "eval", "pic.png")) and len(json.load(open(package_path("x.w_s0")))["deleted"][0]["files"]) == 4, "w: deleted and listed in the package written before"
        tempfile.tempdir = None
        json.dump(pj0, open(os.path.join(proj, "project.json"), "w"))
        # the lane GT stores: the latest lane version stays (numeric order: l10 > l2), one that an unfinished run names stays, an older one with a tag goes
        lroot = os.path.join(tmp, "lane")
        for sig in ("lold", "lnew", "lorph", "lneed"):
            os.makedirs(os.path.join(lroot, sig))
            json.dump({"signature": sig}, open(os.path.join(lroot, sig, "index.json"), "w"))
        lane_versions = lambda: [{"version": "l1", "signature": "lold", "tag": "tg/ok"}, {"version": "l2", "signature": "lneed", "tag": "tg/ok"},    # noqa: E731
                                 {"version": "l10", "signature": "lnew", "tag": "tg/ok"}]
        json.dump({"gt_signature": "0.5_4_zzz", "lane_signature": ["lneed"]}, open(os.path.join(proj, "runs", "x_b_s0", "gt.json"), "w"))
        llatest, lplan = lane_plan(lroot)
        lgot = dict(lplan)
        assert llatest["version"] == "l10" and lgot["lnew"].startswith("keep: the latest") and lgot["lneed"].startswith("keep: needed by") \
            and lgot["lorph"].startswith("keep: not in") and lgot["lold"].startswith("delete"), lgot
        tags.discard("tg/ok")
        assert dict(lane_plan(lroot)[1])["lold"].startswith("keep: the tag"), "no tag, no deletion"
        tags.add("tg/ok")
        # two rigs and a version that makes an older version's GT (autogt/lane_versions.csv, l5): per rig the latest version stays, and where `same_gt_as` is set the earlier store
        for sig in ("lbf", "l4s"):
            os.makedirs(os.path.join(lroot, sig))
            json.dump({"signature": sig}, open(os.path.join(lroot, sig, "index.json"), "w"))
        saved_rows = lane_versions
        lane_versions = lambda: [{"version": "l1", "signature": "lold", "tag": "tg/ok"}, {"version": "l2", "signature": "lneed", "tag": "tg/ok"},    # noqa: E731
                                 {"version": "l4", "signature": "l4s", "tag": "tg/ok"}, {"version": "l5", "signature": "l5code", "tag": "tg/ok", "same_gt_as": "l4"},
                                 {"version": "l5", "signature": "lbf", "tag": "tg/ok", "rig": "bev_fusion"}]
        rlatest, rplan = lane_plan(lroot)
        rgot = dict(rplan)
        assert rlatest["version"] == "l5" and rgot["l4s"].startswith("keep: the latest lane version (l5, lane_center), its code makes the GT of l4") \
            and rgot["lbf"].startswith("keep: the latest lane version (l5, bev_fusion)") and rgot["lold"].startswith("delete") and rgot["lneed"].startswith("keep: needed by"), rgot
        lane_versions = saved_rows
        shutil.rmtree(os.path.join(lroot, "lbf"))
        shutil.rmtree(os.path.join(lroot, "l4s"))
        lane_clean(False, lroot)
        assert os.path.isdir(os.path.join(lroot, "lold")), "a dry run changes nothing"
        _, _, ldone = lane_clean(True, lroot)
        assert ldone == ["lold"] and not os.path.exists(os.path.join(lroot, "lold")) and os.path.isdir(os.path.join(lroot, "lneed")), ldone
        assert "lane GT store lold" in open(os.path.join(proj, "records", "CLEAN_LOG.md"), encoding="utf-8").read(), "logged"
        # review 2026-11-10 (1): the package of a run that was resumed after the first package names its final checkpoint, not the one of the stored manifest
        mk("x.r_s0", status="stopped", exit_=75)
        pr0 = write("x.r_s0")
        assert pr0["status"]["final"] is False and pr0["model"]["ckpt"] == "ckpt_00000010.pt", pr0["model"]
        rjr = os.path.join(proj, "records", "runs", "x.r_s0", "run.json")
        json.dump(dict(json.load(open(rjr)), status="ok", exit=0), open(rjr, "w"))
        open(os.path.join(proj, "runs", "x_r_s0", "ckpt_00000030.pt"), "wb").write(b"w" * 30)
        pr1 = write("x.r_s0")
        assert pr1["status"]["final"] is True and pr1["model"]["ckpt"] == "ckpt_00000030.pt" and pr1["model"]["step"] == 30, "a resumed run: " + str(pr1["model"])
        # (2) data/autogt is shared by every problem folder; signatures are compared as stores; the best three, a torn record and a final run without its package keep a store
        proj2 = os.path.join(tmp, "proj2")
        os.makedirs(os.path.join(proj2, "records"))
        json.dump({"ledger": "records/LEDGER.csv", "run_records": "records/runs", "cards": "runs"}, open(os.path.join(proj2, "project.json"), "w"))
        assert real_problems(tmp) == [proj, proj2], real_problems(tmp)

        def run2(rid, folder, status, exit_, gt, package=False):
            os.makedirs(os.path.join(proj2, "records", "runs", rid))
            json.dump({"id": rid, "status": status, "exit": exit_}, open(os.path.join(proj2, "records", "runs", rid, "run.json"), "w"))
            os.makedirs(os.path.join(proj2, "runs", folder))
            json.dump({"gt_signature": gt}, open(os.path.join(proj2, "runs", folder, "gt.json"), "w"))
            if package:
                os.makedirs(os.path.join(proj2, "results", rid))
                open(os.path.join(proj2, "results", rid, "package.json"), "w").write("{}")
        rows2 = [{"version": f"v{n}", "signature": f"s{n}", "tag": "tg/ok"} for n in (3, 4, 5, 6, 7, 8)] + [{"version": "v9", "signature": "s9", "tag": "tg/ok", "same_gt_as": "v8"},
                                                                                                            {"version": "v10", "signature": "s10", "tag": "tg/ok"}]
        gt2 = os.path.join(tmp, "gt2")
        for sig in ("s3", "s4", "s5", "s6", "s7", "s8", "s10"):
            os.makedirs(os.path.join(gt2, sig))
            json.dump({"signature": sig}, open(os.path.join(gt2, sig, "index.json"), "w"))
        os.makedirs(os.path.join(proj2, "runs", "t.C_s1", "code"))               # run_ar.sh names the folder of the code snapshot after the config: no gt.json in it
        run2("t.C_s1", "t_C_s1", "running", None, "0.5_4_s9")                      # not final, in the OTHER problem, recorded under the CODE signature of v9 (its store is v8's)
        run2("b7", "b7", "ok", 0, "0.5_4_s7", package=True)                        # a network of the best three of the other problem
        json.dump({"entries": [{"name": "n7", "key": "b7", "runs": ["b7"], "checkpoints": [{"path": "runs/b7/ckpt_00000001.pt"}]},
                               {"name": "n3", "key": "nofolder", "checkpoints": [{"path": "runs/nofolder/ckpt_1.pt", "gt_signature": "0.5_4_s3"}]}]}, open(os.path.join(proj2, "records", "keep3.json"), "w"))
        run2("f6", "f6", "ok", 0, "0.5_4_s6")                                      # final, no result package
        run2("torn", "torn", "ok", 0, "0.5_4_s4")
        open(os.path.join(proj2, "records", "runs", "torn", "run.json"), "w").write('{"id": "torn"')       # a record that cannot be read may be a running job
        versions = lambda: rows2       # noqa: E731
        problem_roots = lambda top=None: [proj, proj2]       # noqa: E731
        _, p2 = gt_plan(gt2)
        g2 = dict(p2)
        assert g2["s10"].startswith("keep: the latest") and g2["s5"].startswith("delete"), g2
        assert g2["s8"].startswith("keep: needed by the unfinished run(s) proj2:t.C_s1"), "the run of another problem folder, in the folder that holds its gt.json, under the code signature of v9: " + g2["s8"]
        assert g2["s7"].startswith("keep: the best three network(s) proj2:n7"), g2["s7"]
        assert g2["s3"].startswith("keep: the best three network(s) proj2:n3"), "an entry whose record carries the signature itself, with no run folder: " + g2["s3"]
        assert "proj2:f6" in g2["s6"] and "no result package" in g2["s6"] and g2["s6"].startswith("keep"), g2["s6"]
        assert g2["s4"].startswith("keep: needed by") and "proj2:torn" in g2["s4"], g2["s4"]
        assert all(dict(gt_plan(gt2, [proj])[1])[s].startswith("delete") for s in ("s8", "s7", "s6", "s4", "s3")), "read only the current problem, the stores would go"
        os.makedirs(os.path.join(proj2, "results", "f6"))
        open(os.path.join(proj2, "results", "f6", "package.json"), "w").write("{}")
        assert dict(gt_plan(gt2)[1])["s6"].startswith("delete"), "the package of the final run is written: the store may go"
        _, _, done2 = gt_clean(True, gt2)
        assert sorted(done2) == ["s5", "s6"] and all(os.path.isdir(os.path.join(gt2, s)) for s in ("s3", "s4", "s7", "s8", "s10")), done2
        assert lane_notes([("l1", "keep: x")]) == [] and lane_notes([("l1", "delete: lane l1")]) == [LANE_GAP] and "cannot be protected by run records" in LANE_GAP
        # a stored manifest that says "version unknown" (made before the store was named canonically, or the store is deleted) is read again for the package: v9's key is the GT of v8
        g3 = os.path.join(tmp, "gtrepo")
        os.makedirs(os.path.join(g3, "autogt"))
        open(os.path.join(g3, "autogt", "versions.csv"), "w").write("version,signature,tag,date,generator,what_changed,same_gt_as\nv8,c8c8,t,d,g,w,\nv9,c9c9,t,d,g,w,v8\n")
        mk("x.o_s0")
        json.dump({"gt_signature": "0.5_4_c9c9"}, open(os.path.join(proj, "runs", "x_o_s0", "gt.json"), "w"))
        V.repo_root = lambda start=None: g3
        try:
            po = write("x.o_s0")
            assert po["gt"] == {"version": "v8", "signature": "c8c8"}, po["gt"]
            provf = os.path.join(proj, "results", "x.o_s0", "provenance.json")
            old_prov = json.load(open(provf))
            old_prov["subject"]["train_gt"] = {"key": "0.5_4_c9c9", "signature": "c9c9", "version": "unknown"}          # what the 2026-11-02 tool stored
            json.dump(old_prov, open(provf, "w"))
            assert write("x.o_s0")["gt"] == {"version": "v8", "signature": "c8c8"}, "the package reads the GT again from the key"
        finally:
            V.repo_root = real_root
        print("package selftest passed")
    finally:
        tempfile.tempdir = None
        tag_exists, versions, lane_versions, problem_roots = real_tag, real_versions, real_lane, real_problems
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
        print(f"written: {package_path(a.run)} (tier {pk['tier']}, GT {pk['gt']['version']}, bundle {pk['bundle']['code_hash']}, {len(pk['deep']['files'])} derived files, {human(pk['deep']['bytes'])}"
              + (f", {pk['split_record']['file']}" if pk.get("split_record") else "") + ")")
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
        lane_latest, lane_p, lane_done = lane_clean(a.apply)
        if lane_latest:
            print(f"latest lane GT version: {lane_latest['version']} ({lane_latest['signature']})")
            for sig, text in lane_p:
                print(f"  {sig:14} {text}")
            for note in lane_notes(lane_p):
                print(note)
        print(("deleted: " + ", ".join(done + lane_done)) if a.apply else "dry run: nothing deleted (--apply)")
    else:
        ap.error("package <run id> | show <run id> | clean [--apply] | gt [--apply] | --selftest")


if __name__ == "__main__":
    sys.exit(main())
