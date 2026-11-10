#!/usr/bin/env python3
"""make_manifest.py - the manifest of the raw nuScenes data: which dataset version, which archives, which files, on D:.

    python3 data/NuScenes/tools/make_manifest.py                  # writes data/NuScenes/manifest.json from the files on disk (no network, read-only on D:)
    python3 data/NuScenes/tools/make_manifest.py --check          # compares the disk with the committed manifest; exit 1 and the differences when they differ
    python3 data/NuScenes/tools/make_manifest.py --selftest       # a fake data root in a temporary folder: determinism, known answers, the check finds changes

Why: the data itself is never in git, so a result must be able to NAME the data it was made from. This manifest is small enough for git (about 25 KB) and pins
the data by what cannot change unnoticed: the S3 ETag and size of every archive that was downloaded (from the download logs; an archive is deleted after it was
unpacked and compared with the data, the manifest then keeps its entry with `present: false`, the date, and the result of the comparison under `on_disk_check`:
`--archive-check FILE` merges it), the SHA-256 of the metadata tables
(`v1.0-trainval/*.json`, the dataset version itself), of the map expansion, of the licence and of the stored file lists, and, per sensor folder, the number of
files, their bytes and the SHA-256 of the sorted list "path<TAB>size" (so one missing or extra file shows). A result card names the manifest by its SHA-256
(`sha256sum data/NuScenes/manifest.json`); the GT built from this data is named by `GtGenerator/versions.csv` and its `data/autogt/<signature>/index.json`
(pointers under "links", not copies). Paths in the manifest are relative to the data root (`D:\\Auto-research\\NuScenes`) or to the repository.

The folder fingerprints read 1.5 million files; from WSL that takes 30 minutes, so `scan_tree.ps1` (PowerShell, native, about a minute) does it when `powershell.exe`
is reachable and the root is below /mnt/<drive>; otherwise the same value is computed in Python. `--no-deep` skips the fingerprints (seconds instead of minutes).
`--check --quick` compares sizes only (no SHA-256, a few seconds). The selftest needs only Python.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

SCHEMA = 2                                               # 2: archives carry present / removed / sha256 / on_disk_check; data_identity
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_ROOT = "/mnt/d/Auto-research/NuScenes"
DEFAULT_OUT = os.path.join(REPO, "data", "NuScenes", "manifest.json")
IGNORE = {"generated", "generator"}                      # never part of the comparison
# where archives lie while they are unpacked: ../NuScenes_zips (sibling of the data root; since 2026-11-02) or downloads/ (inside it; the former place)
ZIPS_BASES = ("downloads", os.path.join("..", "NuScenes_zips"))
ARCHIVE_DIRS = (("nuscenes_camera", "camera"), ("nuscenes_radar", "radar"), ("nuscenes_radar_456", "radar"))
ABOUT = ("Identity of the raw nuScenes data on D: and of the files that are derived from it by pointer. Made by data/NuScenes/tools/make_manifest.py from the files "
         "on disk (no network); `make_manifest.py --check` compares the disk with it. Paths are relative to the data root; \"$REPO\" and \"$ROOT\" in view targets "
         "stand for the repository and the data root. A result names the data by `data_identity` (the SHA-256 of the data part of this file: tables, files, file lists, "
         "folder fingerprints; archives, views and dates are not in it, so deleting an archive or making the file again does not change it).")
POSITIVE = ("etag", "etag_verified", "unpacked_marker", "unpacked_members")      # what logs and markers can only add: a scan that knows nothing must not erase it


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_entry(path, hash_files=True):
    e = {"bytes": os.path.getsize(path)}
    if hash_files:
        e["sha256"] = sha256_file(path)
    return e


def rel(path, root):
    return os.path.relpath(path, root).replace(os.sep, "/")


# ---- archives, from the download logs of get_blob_part.sh -------------------------------------------------------------------------------------------------
def parse_log(path):
    """{archive name: {etag, verified, members}} from a part log: '<time> download <name> (etag <etag>)', 'OK <name>', '<time> unpack to <dest> (only: <members>)'
    followed by '<time> unpacked <name>' (printed by extract_blob.sh after tar succeeded: an 'unpack to' without it failed and counts for nothing).
    Several successful unpack steps add up (an archive unpacked in two steps): members = 'all' when one of them had no 'only', else the sorted union."""
    info, cur, seen, pending = {}, None, set(), None
    if not os.path.isfile(path):
        return info
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        m = re.match(r"\d\d:\d\d:\d\d download (\S+) \(etag (\S+)\)", line)
        if m:
            cur, seen, pending = info.setdefault(m.group(1), {}), set(), None
            cur["etag"] = m.group(2)
            continue
        m = re.match(r"OK (\S+)$", line)
        if m:
            info.setdefault(m.group(1), {})["verified"] = True
            continue
        m = re.match(r"\d\d:\d\d:\d\d unpack to \S+(?: \(only: (.*)\))?\s*$", line)
        if m and cur is not None:
            pending = set(m.group(1).split()) if m.group(1) else {"all"}
            continue
        m = re.match(r"\d\d:\d\d:\d\d unpacked (\S+)\s*$", line)
        if m and pending is not None and cur is not None and m.group(1) in info:
            seen |= pending
            pending = None
            cur["members"] = "all" if "all" in seen else " ".join(sorted(seen))
    return info


def carry(new, old, today, keep=lambda key: True):
    """entries of `old` that `new` lacks: the file is gone from the disk (an archive deleted after unpacking). They stay in the manifest with `present: false` and
    the date the manifest first missed them (`removed`); an entry that is already marked keeps its date. Which keys count as 'may be gone' is `keep`."""
    for k, v in (old or {}).items():
        if k not in new and isinstance(v, dict) and keep(k):
            e = dict(v)
            if e.get("present", True):
                e["present"], e["removed"] = False, today
            new[k] = e
    return new


LOG_FIELDS = ("etag", "etag_verified", "unpacked_members")      # what the part log says; the marker file ("unpacked_marker") is a separate witness


def merge_known(prev, scanned, has_log=False):
    """the entry of an archive that is on disk: the scan wins, except that a scan with no log or marker (None / False) does not erase what was known - unless a log
    covers the archive (has_log): then what it says replaces the log fields together (a log that shows a failed unpack resets unpacked_members; the marker only adds).
    Another file under the same name (other bytes) takes nothing over; a copy that was downloaded again (the old entry was `present: false`) loses the old
    `on_disk_check` and `sha256`."""
    if prev.get("bytes") != scanned["bytes"]:
        return dict(scanned)
    e = {k: v for k, v in prev.items() if k != "removed"}
    for k, v in scanned.items():
        if k in POSITIVE and v in (None, False) and e.get(k) not in (None, False) and not (has_log and k in LOG_FIELDS):
            continue
        e[k] = v
    if prev.get("present") is False:
        e.pop("on_disk_check", None)
        e.pop("sha256", None)
    return e


def zips_bases(root):
    return [os.path.join(root, b) for b in ZIPS_BASES]


def scan_archives(root, previous=None, today=None, warn=None):
    """{archive name: entry} of the archives on disk (`present: true`) plus those of the `previous` manifest that are gone (`present: false`, see carry).
    warn(text) is called for an archive found in two places and for a .tgz below a zips base that the layout (ARCHIVE_DIRS) does not know."""
    today = today or time.strftime("%Y-%m-%d")
    warn = warn or (lambda s: None)
    out, seen_paths = {}, set()
    for zbase in zips_bases(root):
        for sub, kind in ARCHIVE_DIRS:
            base = os.path.normpath(os.path.join(zbase, sub))
            if not os.path.isdir(base):
                continue
            logs = {}
            for fn in sorted(os.listdir(base)):
                if re.match(r"part_\d+\.log$", fn):
                    logs.update(parse_log(os.path.join(base, fn)))
            for dp, dn, fns in os.walk(base):
                dn.sort()
                for f in sorted(fns):
                    if f.endswith(".tgz"):
                        p = os.path.join(dp, f)
                        seen_paths.add(os.path.normpath(p))
                        lg = logs.get(f, {})
                        e = {"kind": kind, "path": rel(p, root), "bytes": os.path.getsize(p), "etag": lg.get("etag"), "etag_verified": bool(lg.get("verified")),
                             "unpacked_marker": os.path.exists(p + ".unpacked"), "unpacked_members": lg.get("members"), "present": True}
                        if f in out:
                            warn("archive %s lies in two places (%s and %s): the larger file is taken" % (f, out[f]["path"], e["path"]))
                            if out[f]["bytes"] >= e["bytes"]:
                                continue
                        if f in (previous or {}):
                            e = merge_known(previous[f], e, has_log="etag" in lg)
                        out[f] = e
    for zbase in zips_bases(root):
        zb = os.path.normpath(zbase)
        if not os.path.isdir(zb):
            continue
        for dp, dn, fns in os.walk(zb):
            dn[:] = sorted(d for d in dn if not (dp == zb and d == "occ3d"))          # the Occ3D files are handled by occ3d()
            for f in fns:
                p = os.path.normpath(os.path.join(dp, f))
                if f.endswith(".tgz") and p not in seen_paths:
                    warn("%s is an archive the layout does not know (expected below %s): it is not in the manifest" % (rel(p, root), ", ".join(d for d, _ in ARCHIVE_DIRS)))
    return carry(out, previous, today)


def data_markers(root):
    """the files every nuScenes archive leaves in the data ('Thank you for downloading X.tgz'): which archives were unpacked into nuscenes_full. Not a proof that the
    data of an archive is still there: see 'folders'."""
    out = {}
    for ds in ("nuscenes_full", "nuscenes_mini"):
        d = os.path.join(root, ds)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.startswith(".v1.0-") and fn.endswith(".txt"):
                m = re.search(r"downloading (\S+?\.tgz)", open(os.path.join(d, fn), encoding="utf-8", errors="replace").read())
                out[ds + "/" + fn] = m.group(1) if m else None
    return out


# ---- tables, maps, licence, file lists ------------------------------------------------------------------------------------------------------------------
def json_len(path):
    with open(path, encoding="utf-8") as f:
        return len(json.load(f))


def datasets(root, hash_files=True):
    out, files = {}, {}
    for ds in ("nuscenes_full", "nuscenes_mini"):
        d = os.path.join(root, ds)
        if not os.path.isdir(d):
            continue
        for ver in sorted(x for x in os.listdir(d) if re.match(r"v1\.0-(trainval|mini|test)$", x) and os.path.isdir(os.path.join(d, x))):
            tdir = os.path.join(d, ver)
            tables = {fn: file_entry(os.path.join(tdir, fn), hash_files) for fn in sorted(os.listdir(tdir)) if fn.endswith(".json")}
            e = {"dir": ds + "/" + ver, "tables": tables}
            for name in ("scene", "sample"):
                p = os.path.join(tdir, name + ".json")
                if os.path.isfile(p):
                    e[name + "s"] = json_len(p)
            out[ver] = e
        lic = os.path.join(d, "LICENSE")
        if os.path.isfile(lic):
            files[ds + "/LICENSE"] = file_entry(lic, hash_files)
        exp = os.path.join(d, "maps", "expansion")
        if os.path.isdir(exp):
            for fn in sorted(os.listdir(exp)):
                if fn.endswith(".json"):
                    files[ds + "/maps/expansion/" + fn] = file_entry(os.path.join(exp, fn), hash_files)
    return out, files


def file_lists(root, hash_files=True):
    out, d = {}, os.path.join(root, "manifests")
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".tsv"):
                p = os.path.join(d, fn)
                e = file_entry(p, hash_files)
                with open(p, "rb") as f:
                    e["lines"] = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 24), b""))
                out["manifests/" + fn] = e
    return out


# ---- folder fingerprints --------------------------------------------------------------------------------------------------------------------------------
def fingerprint(entries):
    """entries [(relative path with /, size)] -> (files, bytes, sha256 of the sorted lines 'path<TAB>size<LF>'): the definition scan_tree.ps1 follows"""
    entries = sorted(entries)
    text = "".join("%s\t%d\n" % (p, s) for p, s in entries)
    return {"files": len(entries), "bytes": sum(s for _, s in entries), "list_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def scan_py(path):
    if not os.path.isdir(path):
        return {"files": None, "bytes": None, "list_sha256": None}
    entries = []
    for dp, _, fns in os.walk(path):
        for f in fns:
            p = os.path.join(dp, f)
            entries.append((rel(p, path), os.path.getsize(p)))
    return fingerprint(entries)


def can_use_powershell(root):
    return bool(shutil.which("powershell.exe")) and bool(shutil.which("wslpath")) and re.match(r"^/mnt/[a-z]/", root) is not None


def scan_ps(root, folders):
    ps1 = subprocess.check_output(["wslpath", "-w", os.path.join(HERE, "scan_tree.ps1")], text=True).strip()
    rootw = subprocess.check_output(["wslpath", "-w", root], text=True).strip()
    out = subprocess.check_output(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1, "-Root", rootw, "-Folders", ",".join(folders)], text=True)
    data = json.loads(out)
    return {r["folder"]: {"files": r["files"], "bytes": r["bytes"], "list_sha256": r["list_sha256"]} for r in (data if isinstance(data, list) else [data])}


def folder_names(root):
    names = []
    for ds, ver in (("nuscenes_full", "v1.0-trainval"), ("nuscenes_mini", "v1.0-mini")):
        d = os.path.join(root, ds)
        if not os.path.isdir(d):
            continue
        for top in ("samples", "sweeps"):
            td = os.path.join(d, top)
            if os.path.isdir(td):
                names += [ds + "/" + top + "/" + s for s in sorted(os.listdir(td)) if os.path.isdir(os.path.join(td, s))]
        names += [ds + "/" + x for x in ("maps", ver) if os.path.isdir(os.path.join(d, x))]
    if os.path.isdir(os.path.join(root, "occ3d", "gts")):
        names.append("occ3d/gts")
    return names


def scan_folders(root, scanner="auto"):
    names = folder_names(root)
    use_ps = scanner == "ps" or (scanner == "auto" and can_use_powershell(root))
    return (scan_ps(root, names) if use_ps else {n: scan_py(os.path.join(root, n)) for n in names}), ("powershell" if use_ps else "python")


# ---- Occ3D, views, links --------------------------------------------------------------------------------------------------------------------------------
def drive_ids(repo):
    """the Google Drive ids of the Occ3D fetch, read from tools/get_occ3d_labels.sh (working tree, else the tag that holds the former history/ copy)"""
    text, path = None, os.path.join(repo, "data", "NuScenes", "tools", "get_occ3d_labels.sh")
    if os.path.isfile(path):
        text = open(path, encoding="utf-8", errors="replace").read()
    else:
        try:
            text = subprocess.check_output(["git", "-C", repo, "show", "nuscenes/tools-history-2026-11-01:datasets/NuScenes/tools/history/get_occ3d_labels.sh"],
                                           text=True, stderr=subprocess.DEVNULL)
        except Exception:
            return {}
    return {m.group(2): m.group(1) for m in re.finditer(r"get_gdrive_file\.sh\"?\s+(\S+)\s+\"[^\"]*/occ3d/([^\"]+)\"", text)}


def occ3d(root, repo, hash_files=True, previous=None, today=None):
    """the Occ3D labels: Drive ids, hashes of the two downloaded files (`present: false` once they were deleted after unpacking) and of the unpacked annotations"""
    today = today or time.strftime("%Y-%m-%d")
    prev_files = (previous or {}).get("files") or {}
    out = {"drive_ids": drive_ids(repo), "files": {}}
    for zbase in zips_bases(root):
        for fn in ("annotations.json", "gts.tar.gz"):
            p = os.path.normpath(os.path.join(zbase, "occ3d", fn))
            if os.path.isfile(p):
                key = rel(p, root)
                e = dict(file_entry(p, hash_files), present=True)
                out["files"][key] = merge_known(prev_files[key], e) if isinstance(prev_files.get(key), dict) else e
    def moved(k):                                                       # the same file now lies at another path: moved, not removed
        old = prev_files[k]
        return any(os.path.basename(j) == os.path.basename(k) and v.get("present") and v.get("bytes") == old.get("bytes") for j, v in out["files"].items() if j != k)
    carry(out["files"], prev_files, today, keep=lambda k: "/occ3d/" in k and not k.startswith("occ3d/") and not moved(k))
    p = os.path.join(root, "occ3d", "annotations.json")
    if os.path.isfile(p):
        out["files"]["occ3d/annotations.json"] = file_entry(p, hash_files)
    g = os.path.join(root, "occ3d", "gts")
    if os.path.isdir(g):
        out["scenes"] = len([x for x in os.listdir(g) if os.path.isdir(os.path.join(g, x))])
    return out


def views(home, root, repo):
    out = {}
    for v in ("nuscenes_full", "nuscenes"):
        d = os.path.join(home, "data", v)
        if not os.path.isdir(d):
            continue
        links = {}
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n)
            if os.path.islink(p):
                t = os.readlink(p)
                if t.startswith(root + "/"):
                    t = "$ROOT/" + t[len(root) + 1:]
                elif t.startswith(repo + "/"):
                    t = "$REPO/" + t[len(repo) + 1:]
                links[n] = t
        out["~/data/" + v] = links
    return out


LINKS = {"gt": {"registry": "GtGenerator/versions.csv", "tags": "autogt/v<N>", "store": "data/autogt/<signature>/index.json", "check": "python -m GtGenerator data <dataroot>"},
         "lane_additions": "bev_fusion_lane_center/data/rebuild.json",
         "recipes": "data/NuScenes/README.md and data/NuScenes/tools/ (link_views.sh builds the views)"}


# ---- build, diff, check ---------------------------------------------------------------------------------------------------------------------------------
def apply_checks(archives, occ_files, checks):
    """put the result of comparing an archive with the data into its entry (`on_disk_check`); the check must belong to the file of the entry (bytes, SHA-256)"""
    for name, c in (checks or {}).items():
        target = archives.get(name)
        if target is None:                                              # the Occ3D files live in occ3d.files under their path
            hits = [k for k in occ_files if k == name or k.endswith("/" + name)]
            live = [k for k in hits if occ_files[k].get("present", True)]
            pick = live if len(live) == 1 else (hits if len(hits) == 1 else [])
            if len(hits) > 1 and not pick:
                raise SystemExit("--archive-check names %s, which matches several Occ3D entries of the manifest: %s" % (name, ", ".join(hits)))
            target = occ_files[pick[0]] if pick else None
        if target is None:
            raise SystemExit("--archive-check names an archive that is neither on disk nor in the existing manifest: %s" % name)
        missing = [k for k in ("archive_bytes", "archive_sha256", "differences") if c.get(k) is None]
        if missing:
            raise SystemExit("--archive-check for %s lacks %s: a result must name the file it was made on (bytes, SHA-256) and the number of differences" % (name, ", ".join(missing)))
        c = dict(c, complete=bool(c["differences"] == 0 and not c.get("truncated") and not c.get("nothing_compared") and (c.get("files") or 0) > 0 and not c.get("skipped_members")))
        if c["archive_bytes"] != target.get("bytes"):
            raise SystemExit("--archive-check for %s was made on a file of %s bytes, the manifest has %s: not the same file" % (name, c["archive_bytes"], target.get("bytes")))
        if c.get("archive_sha256") and target.get("sha256") and c["archive_sha256"] != target["sha256"]:
            raise SystemExit("--archive-check for %s names SHA-256 %s, the manifest has %s: not the same file" % (name, c["archive_sha256"][:12], target["sha256"][:12]))
        target["on_disk_check"] = c
        if c.get("archive_sha256"):
            target["sha256"] = c["archive_sha256"]                      # the identity of the exact file that was compared


def data_identity(m):
    """SHA-256 of the data part of the manifest: tables, licence / map files, file lists, folder fingerprints, the Occ3D scene count and annotations. Archives, markers,
    views, links, dates and the tool's own hash are not in it: deleting an archive or making the file again does not change what a result says about its data."""
    core = {k: m.get(k) for k in ("datasets", "files", "file_lists", "folders")}
    o = m.get("occ3d") or {}
    core["occ3d"] = {"scenes": o.get("scenes"), "annotations": (o.get("files") or {}).get("occ3d/annotations.json")}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build(root, home=None, repo=REPO, deep=True, scanner="auto", hash_files=True, previous=None, checks=None, today=None, warn=None):
    """previous: the existing manifest (a dict); archives that are gone from the disk are carried over from it (`present: false`, dated `today`). checks: {archive
    name: dict} merged into the entry's on_disk_check (the result of comparing the archive with the data). warn(text): messages about odd archive layouts"""
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(os.path.join(root, "nuscenes_full")):
        raise SystemExit("%s is not a nuScenes data root (no nuscenes_full below it): nothing is written" % root)
    home = os.path.expanduser(home or "~")
    today = today or time.strftime("%Y-%m-%d")
    ds, files = datasets(root, hash_files)
    previous = previous or {}
    archives = scan_archives(root, previous.get("archives"), today, warn)
    occ = occ3d(root, repo, hash_files, previous.get("occ3d"), today)
    apply_checks(archives, occ["files"], checks)
    m = {"schema": SCHEMA, "about": ABOUT,
         "root": {"windows": "D:\\Auto-research\\NuScenes", "wsl": "/mnt/d/Auto-research/NuScenes"},
         "licence": "nuScenes is for non-commercial use (the LICENSE of the data is under files); downloads only with Mirko's OK",
         "source": {"bucket": "motional-nuscenes", "listing": "data/NuScenes/tools/list_nuscenes_bucket.py", "etag_check": "data/NuScenes/tools/verify_s3_etag.py"},
         "datasets": ds, "files": files, "file_lists": file_lists(root, hash_files), "archives": archives, "data_markers": data_markers(root),
         "occ3d": occ, "views": views(home, root, repo), "links": LINKS, "generated": today}
    tool = os.path.join(HERE, "make_manifest.py")
    gen = {"tool": "data/NuScenes/tools/make_manifest.py", "tool_sha256": sha256_file(tool) if os.path.isfile(tool) else None}
    if deep:
        m["folders"], gen["folder_scanner"] = scan_folders(root, scanner)
        if hash_files:
            m["data_identity"] = data_identity(m)
    m["generator"] = gen
    return m


def load_checks(path):
    """{archive name: result dict} from a JSON file {name: {...}}, a JSON / JSONL file of results that carry an "archive" key, or a folder of <name>.json results
    (what verify_archive.ps1 writes). The "archive" key becomes the name."""
    def norm(obj):
        obj = dict(obj)
        name = obj.pop("archive")
        obj.setdefault("method", "data/NuScenes/tools/verify_archive.ps1: every member against the data folder (existence, size, bytes)")
        return os.path.basename(name), obj
    if os.path.isdir(path):
        items = []
        for fn in sorted(os.listdir(path)):
            if fn.endswith(".json"):
                with open(os.path.join(path, fn), encoding="utf-8") as f:
                    items.append(json.load(f))
        return dict(norm(o) for o in items)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        data = json.loads(text)
    except ValueError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]          # JSONL
    if isinstance(data, dict) and "archive" in data:
        data = [data]
    if isinstance(data, list):
        return dict(norm(o) for o in data)
    if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        return data
    raise SystemExit("--archive-check %s: expected {name: result}, results with an 'archive' key (JSON / JSONL) or a folder of them" % path)


def diff(old, new, path="", partial=False):
    """differences between two manifests (dict trees); partial: a key that the new one lacks (not computed in --quick) is not a difference"""
    out = []
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new)):
            if not path and k in IGNORE:
                continue
            p = path + "/" + k
            if k not in new:
                if partial and k in ("sha256", "data_identity"):   # --quick computes no hashes, --no-deep no folder fingerprints
                    continue
                out.append("%s: in the manifest, not on disk now" % p)
            elif k not in old:
                out.append("%s: on disk now, not in the manifest" % p)
            else:
                out += diff(old[k], new[k], p, partial)
    elif old != new:
        out.append("%s: manifest %s, disk %s" % (path, str(old)[:60], str(new)[:60]))
    return out


def check(manifest_path, root, home=None, repo=REPO, deep=True, scanner="auto", quick=False):
    with open(manifest_path, encoding="utf-8") as f:
        old = json.load(f)
    new = build(root, home, repo, deep=deep, scanner=scanner, hash_files=not quick, previous=old)
    if not deep:
        old = {k: v for k, v in old.items() if k != "folders"}
    return diff(old, new, partial=quick or not deep)


def dump(m):
    return json.dumps(m, indent=1, sort_keys=True, ensure_ascii=True) + "\n"


# ---- selftest -------------------------------------------------------------------------------------------------------------------------------------------
def _w(path, data=b""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode("utf-8"))


def selftest():
    tmp = tempfile.mkdtemp(prefix="manifest_selftest_")
    A1, A2, R4, G = "v1.0-trainval01_blobs_camera.tgz", "v1.0-trainval02_blobs_camera.tgz", "v1.0-trainval04_blobs_radar.tgz", "downloads/occ3d/gts.tar.gz"
    T0, T1, T2 = "2031-05-01", "2031-05-02", "2031-05-03"
    try:
        root, home, repo = os.path.join(tmp, "D", "NuScenes"), os.path.join(tmp, "home"), os.path.join(tmp, "repo")
        zips = os.path.join(tmp, "D", "NuScenes_zips")
        p_a1, p_r4 = os.path.join(root, "downloads/nuscenes_camera/p01", A1), os.path.join(root, "downloads/nuscenes_radar_456", R4)
        os.makedirs(os.path.join(repo, "data", "NuScenes", "tools"))
        _w(os.path.join(repo, "data", "NuScenes", "tools", "get_occ3d_labels.sh"),
           'bash "$here/get_gdrive_file.sh" 1AAA "$root/downloads/occ3d/annotations.json"\nbash "$here/get_gdrive_file.sh" 1BBB "$root/downloads/occ3d/gts.tar.gz"\n')
        _w(p_a1, b"x" * 10)
        _w(p_a1 + ".unpacked")
        _w(os.path.join(root, "downloads/nuscenes_camera/part_01.log"),
           "18:59:44 download %s (etag abc123-2)\n20:55:11 verify %s\nOK %s\n20:58:04 unpack %s\n21:02:15 unpack to /mnt/d/x/nuscenes_full (only: sweeps)\n"
           "21:03:30 unpacked %s\n21:04:25 done %s\n" % ((A1,) * 6))
        _w(p_r4, b"y" * 5)
        _w(os.path.join(zips, "nuscenes_camera/p02", A2), b"w" * 7)
        _w(os.path.join(zips, "nuscenes_camera/part_02.log"),
           "10:00:00 download %s (etag def456-3)\nOK %s\n10:05:00 unpack to /mnt/d/x/nuscenes_full (only: sweeps/CAM_FRONT)\n10:06:00 unpacked %s\n"
           "10:09:00 unpack to /mnt/d/x/nuscenes_full (only: samples/CAM_BACK sweeps/CAM_BACK)\n10:10:00 unpacked %s\n" % ((A2,) * 4))
        _w(os.path.join(root, "manifests/nuscenes_full.tsv"), "a\t1\nb\t2\n")
        _w(os.path.join(root, "nuscenes_full/v1.0-trainval/scene.json"), json.dumps([{"name": "scene-1"}, {"name": "scene-2"}]))
        _w(os.path.join(root, "nuscenes_full/v1.0-trainval/sample.json"), json.dumps([{}, {}, {}]))
        _w(os.path.join(root, "nuscenes_full/LICENSE"), "licence text")
        _w(os.path.join(root, "nuscenes_full/.v1.0-trainval01_blobs.txt"), "Thank you for downloading v1.0-trainval01_blobs.tgz, contact us at https://www.nuscenes.org")
        _w(os.path.join(root, "nuscenes_full/samples/CAM_FRONT/a.jpg"), b"abc")
        _w(os.path.join(root, "nuscenes_full/samples/CAM_FRONT/b.jpg"), b"abcd")
        _w(os.path.join(root, "nuscenes_full/sweeps/CAM_FRONT/c.jpg"), b"z")
        _w(os.path.join(root, "nuscenes_full/maps/expansion/m.json"), b"{}")
        _w(os.path.join(root, "nuscenes_mini/v1.0-mini/scene.json"), json.dumps([{}]))
        _w(os.path.join(root, "nuscenes_mini/v1.0-mini/sample.json"), json.dumps([{}, {}]))
        _w(os.path.join(root, "nuscenes_mini/samples/CAM_FRONT/e.jpg"), b"e")
        _w(os.path.join(root, "downloads/occ3d/annotations.json"), b"{}")
        _w(os.path.join(root, "downloads/occ3d/gts.tar.gz"), b"tgz")
        _w(os.path.join(root, "occ3d/annotations.json"), b"{}")
        _w(os.path.join(root, "occ3d/gts/scene-0001/tok/labels.npz"), b"npz")
        for v, tgt in (("nuscenes_full", os.path.join(root, "nuscenes_full", "samples")), ("nuscenes", os.path.join(repo, "x"))):
            os.makedirs(os.path.join(home, "data", v))
            os.symlink(tgt, os.path.join(home, "data", v, "samples"))
        kw = dict(home=home, repo=repo, scanner="py")

        def chk(path, **extra):
            return check(path, root, **dict(kw, **extra))

        def refused(fn, why):
            try:
                fn()
            except SystemExit:
                return
            raise AssertionError("must be refused: " + why)

        # --- the log of an archive: successful unpack steps add up, a failed one counts for nothing
        _w(os.path.join(tmp, "x.log"),
           "01:00:00 download A (etag e1)\n01:01:00 unpack to /x (only: sweeps)\n01:02:00 unpacked A\n"
           "02:00:00 download B (etag e2)\n02:01:00 unpack to /x\n02:02:00 unpacked B\n"
           "03:00:00 download C (etag e3)\n03:01:00 unpack to /x (only: samples)\ntar (child): Cannot connect to D: resolve failed\n"
           "04:00:00 download D (etag e4)\n04:01:00 unpack to /x (only: a)\n04:02:00 unpacked D\n04:03:00 unpack to /x (only: b)\n04:04:00 unpacked D\n"
           "05:00:00 download E (etag e5)\n05:01:00 unpack to /x (only: a)\n05:02:00 unpacked E\n05:03:00 unpack to /x\n05:04:00 unpacked E\n")
        pl = parse_log(os.path.join(tmp, "x.log"))
        assert [pl[k].get("members") for k in "ABCDE"] == ["sweeps", "all", None, "a b", "all"] and pl["C"]["etag"] == "e3", pl

        # --- the first manifest
        m1, m2 = build(root, today=T0, **kw), build(root, today=T0, **kw)
        assert dump(m1) == dump(m2), "two builds of the same disk differ"
        assert m1["schema"] == SCHEMA and m1["generated"] == T0 and re.fullmatch(r"[0-9a-f]{64}", m1["data_identity"]), (m1["schema"], m1["generated"])
        a = m1["archives"]
        assert a[A1] == {"kind": "camera", "path": "downloads/nuscenes_camera/p01/" + A1, "bytes": 10, "etag": "abc123-2", "etag_verified": True, "unpacked_marker": True,
                         "unpacked_members": "sweeps", "present": True}, a
        assert a[A2]["path"] == "../NuScenes_zips/nuscenes_camera/p02/" + A2 and a[A2]["unpacked_members"] == "samples/CAM_BACK sweeps/CAM_BACK sweeps/CAM_FRONT", a[A2]
        assert a[R4]["etag"] is None and not a[R4]["etag_verified"], "no log, no ETag claimed"
        assert m1["datasets"]["v1.0-trainval"]["scenes"] == 2 and m1["datasets"]["v1.0-trainval"]["samples"] == 3 and m1["datasets"]["v1.0-mini"]["samples"] == 2
        assert m1["datasets"]["v1.0-trainval"]["tables"]["scene.json"]["sha256"] == hashlib.sha256(open(os.path.join(root, "nuscenes_full/v1.0-trainval/scene.json"), "rb").read()).hexdigest()
        assert m1["file_lists"]["manifests/nuscenes_full.tsv"]["lines"] == 2
        want = hashlib.sha256(b"a.jpg\t3\nb.jpg\t4\n").hexdigest()                    # known answer, computed without fingerprint()
        assert m1["folders"]["nuscenes_full/samples/CAM_FRONT"] == {"files": 2, "bytes": 7, "list_sha256": want}, m1["folders"]["nuscenes_full/samples/CAM_FRONT"]
        assert m1["folders"]["occ3d/gts"]["files"] == 1 and m1["folders"]["nuscenes_full/maps"]["files"] == 1
        assert m1["data_markers"] == {"nuscenes_full/.v1.0-trainval01_blobs.txt": "v1.0-trainval01_blobs.tgz"}
        assert m1["occ3d"]["drive_ids"] == {"annotations.json": "1AAA", "gts.tar.gz": "1BBB"} and m1["occ3d"]["scenes"] == 1 and m1["occ3d"]["files"][G]["present"] is True
        assert m1["views"]["~/data/nuscenes_full"]["samples"] == "$ROOT/nuscenes_full/samples" and m1["views"]["~/data/nuscenes"]["samples"] == "$REPO/x"
        out = os.path.join(tmp, "manifest.json")
        _w(out, dump(m1))
        assert chk(out) == [] and chk(out, quick=True) == [] and chk(out, deep=False) == [], "an unchanged disk must check clean (full, quick, no-deep)"

        # --- a changed table / an extra file are seen
        _w(os.path.join(root, "nuscenes_full/v1.0-trainval/scene.json"), json.dumps([{"name": "scene-1"}, {"name": "scene-X"}]))
        d = chk(out)
        assert any("scene.json/sha256" in x for x in d) and not any("scene.json/bytes" in x for x in d) and any("data_identity" in x for x in d), d
        assert chk(out, quick=True) == [], "quick compares sizes only: a same-size change is not seen"
        _w(os.path.join(root, "nuscenes_full/v1.0-trainval/scene.json"), json.dumps([{"name": "scene-1"}, {"name": "scene-2"}]))
        _w(os.path.join(root, "nuscenes_full/samples/CAM_FRONT/d.jpg"), b"d")
        assert any("folders/nuscenes_full/samples/CAM_FRONT" in x for x in chk(out)) and chk(out, deep=False) == [], "--no-deep does not look at folders"
        os.remove(os.path.join(root, "nuscenes_full/samples/CAM_FRONT/d.jpg"))

        # --- the archives are deleted: a difference until the manifest is made again, then clean in every mode
        os.remove(p_r4)
        os.remove(os.path.join(root, "downloads/occ3d/gts.tar.gz"))
        for extra in ({}, {"quick": True}, {"deep": False}):
            d = chk(out, **extra)
            assert any(R4 + "/present" in x for x in d) and any("gts.tar.gz/present" in x for x in d), (extra, d)
        ok = {"differences": 0, "skipped_members": 0, "truncated": False}
        c1, c4, cg = (dict(ok, members=3, files=3, archive_bytes=10, archive_sha256="ab" * 32), dict(ok, members=1, files=1, archive_bytes=5, archive_sha256="cd" * 32),
                      dict(ok, members=1, files=1, archive_bytes=3, archive_sha256=hashlib.sha256(b"tgz").hexdigest()))
        K = lambda c: dict(c, complete=True)                            # what the manifest stores: the result and the derived 'complete'
        m3 = build(root, previous=m1, checks={A1: c1, R4: c4, "gts.tar.gz": cg}, today=T1, **kw)
        r = m3["archives"][R4]
        assert r["present"] is False and r["removed"] == T1 and r["bytes"] == 5 and r["kind"] == "radar" and r["on_disk_check"] == K(c4), r        # identity kept, marked gone, check kept
        e1 = m3["archives"][A1]
        assert e1["on_disk_check"] == K(c1) and e1["sha256"] == "ab" * 32 and e1["present"] is True and "on_disk_check" not in m3["archives"][A2], "a check goes to its own archive only"
        g3 = m3["occ3d"]["files"][G]
        assert g3["present"] is False and g3["sha256"] == m1["occ3d"]["files"][G]["sha256"] and g3["on_disk_check"] == K(cg) and g3["removed"] == T1, g3
        assert "present" not in m3["occ3d"]["files"]["occ3d/annotations.json"] and m3["data_identity"] == m1["data_identity"], "deleting archives is no change of the data"
        part = build(root, previous=m1, checks={A1: dict(c1, files=2, skipped_members=5)}, today=T1, **kw)["archives"][A1]["on_disk_check"]
        assert part["complete"] is False, "a comparison that skipped members does not prove the archive is on disk"
        os.makedirs(os.path.join(zips, "occ3d"), exist_ok=True)
        shutil.move(os.path.join(root, "downloads/occ3d/annotations.json"), os.path.join(zips, "occ3d/annotations.json"))   # moved, not removed
        of = build(root, previous=m1, today=T2, **kw)["occ3d"]["files"]
        assert of["../NuScenes_zips/occ3d/annotations.json"]["present"] is True and "downloads/occ3d/annotations.json" not in of, of
        shutil.move(os.path.join(zips, "occ3d/annotations.json"), os.path.join(root, "downloads/occ3d/annotations.json"))
        out3 = os.path.join(tmp, "manifest3.json")
        _w(out3, dump(m3))
        for extra in ({}, {"quick": True}, {"deep": False}):
            assert chk(out3, **extra) == [], ("a manifest made after the deletion checks clean", extra)
        m3["archives"][R4]["removed"] = "2000-01-01"                                                  # the date of the first miss survives the next build, so does the check
        m4 = build(root, previous=m3, today=T2, **kw)
        assert m4["archives"][R4]["removed"] == "2000-01-01" and m4["archives"][R4]["on_disk_check"] == K(c4) and m4["archives"][A1]["on_disk_check"] == K(c1)
        assert m4["occ3d"]["files"][G]["on_disk_check"] == K(cg) and m4["occ3d"]["files"][G]["removed"] == T1

        # --- a log that covers the archive and shows a failed unpack resets a claim 'all'; a scan that knows nothing must not erase; another file; a download again
        _w(os.path.join(root, "downloads/nuscenes_camera/part_01.log"), "18:59:44 download %s (etag abc123-2)\n21:02:15 unpack to /x\ntar (child): Cannot connect to D: resolve failed\n" % A1)
        prev_all = json.loads(dump(m3))
        prev_all["archives"][A1]["unpacked_members"] = "all"
        e = build(root, previous=prev_all, today=T2, **kw)["archives"][A1]
        assert e["unpacked_members"] is None and e["etag"] == "abc123-2", e
        os.remove(os.path.join(root, "downloads/nuscenes_camera/part_01.log"))
        os.remove(p_a1 + ".unpacked")
        e = build(root, previous=m3, today=T2, **kw)["archives"][A1]
        assert (e["etag"], e["etag_verified"], e["unpacked_members"], e["unpacked_marker"]) == ("abc123-2", True, "sweeps", True) and e["on_disk_check"] == K(c1), e
        _w(p_a1, b"x" * 11)
        e = build(root, previous=m3, today=T2, **kw)["archives"][A1]
        assert e["bytes"] == 11 and e["etag"] is None and "on_disk_check" not in e and "sha256" not in e, e        # another file under the same name takes nothing over
        _w(p_a1, b"x" * 10)
        _w(p_r4, b"y" * 5)                                                                            # downloaded again (same size): present, not removed, not compared yet
        r = build(root, previous=m4, today=T2, **kw)["archives"][R4]
        assert r["present"] is True and "removed" not in r and "on_disk_check" not in r, r
        os.remove(p_r4)
        for bad, why in (({"nope.tgz": c1}, "an unknown archive"), ({A1: dict(c1, archive_bytes=99)}, "a check made on a file of other size"),
                         ({A1: dict(c1, archive_sha256="cd" * 32)}, "a check made on a file with another SHA-256"), ({A1: {"differences": 0}}, "a check that does not name its file"),
                         ({A1: {k: v for k, v in c1.items() if k != "differences"}}, "a check without the number of differences")):
            refused(lambda: build(root, previous=m3, checks=bad, today=T2, **kw), why)

        # --- a vanished DATA file is never carried over as if it had been an archive
        os.remove(os.path.join(root, "occ3d/annotations.json"))
        os.remove(os.path.join(root, "nuscenes_full/LICENSE"))
        assert any("occ3d/annotations.json" in x for x in chk(out3)) and any("LICENSE" in x for x in chk(out3)), chk(out3)
        m8 = build(root, previous=m3, today=T2, **kw)
        assert "occ3d/annotations.json" not in m8["occ3d"]["files"] and "nuscenes_full/LICENSE" not in m8["files"], "a vanished data file must not be carried over"
        _w(os.path.join(root, "occ3d/annotations.json"), b"{}")
        _w(os.path.join(root, "nuscenes_full/LICENSE"), "licence text")

        # --- roots and layouts that do not fit are said, not silently recorded as 'removed'
        refused(lambda: build(os.path.join(tmp, "D", "NuScene"), today=T2, **kw), "a root without nuscenes_full (typo, drive not mounted)")
        warns = []
        _w(os.path.join(root, "downloads/camera_x/p99/stray.tgz"), b"s")
        build(root, previous=m3, today=T2, warn=warns.append, **kw)
        assert any("stray.tgz" in w for w in warns), warns
        os.remove(os.path.join(root, "downloads/camera_x/p99/stray.tgz"))
        _w(os.path.join(zips, "nuscenes_camera/p01", A1), b"x" * 4)                                  # the same name in both places: the larger file is taken, with a warning
        warns = []
        e = build(root, previous=m3, today=T2, warn=warns.append, **kw)["archives"][A1]
        assert e["bytes"] == 10 and any("two places" in w for w in warns), (e, warns)
        os.remove(os.path.join(zips, "nuscenes_camera/p01", A1))
        root2 = os.path.join(tmp, "E", "NuScenes")                                                    # Occ3D files only in ../NuScenes_zips
        _w(os.path.join(root2, "nuscenes_full/v1.0-trainval/scene.json"), "[]")
        _w(os.path.join(root2, "nuscenes_full/v1.0-trainval/sample.json"), "[]")
        _w(os.path.join(tmp, "E", "NuScenes_zips/occ3d/gts.tar.gz"), b"q")
        assert build(root2, today=T0, **kw)["occ3d"]["files"]["../NuScenes_zips/occ3d/gts.tar.gz"]["present"] is True

        # --- the shapes --archive-check reads
        cd = os.path.join(tmp, "checks")
        _w(os.path.join(cd, "a.json"), json.dumps({"archive": A1, "differences": 0}))
        _w(os.path.join(cd, "b.json"), json.dumps({"archive": A2, "differences": 2}))
        assert set(load_checks(cd)) == {A1, A2} and load_checks(cd)[A2]["differences"] == 2 and "method" in load_checks(cd)[A1]
        _w(os.path.join(tmp, "c.jsonl"), json.dumps(dict(c1, archive=A1)) + "\n" + json.dumps(dict(cg, archive="gts.tar.gz")) + "\n")
        assert set(load_checks(os.path.join(tmp, "c.jsonl"))) == {A1, "gts.tar.gz"}
        _w(os.path.join(tmp, "d.json"), json.dumps({A1: c1}))
        assert load_checks(os.path.join(tmp, "d.json")) == {A1: c1}

        # --- the command line: gone archives are carried, --fresh drops them, --check, a missing --out, --check with --archive-check
        cli = [sys.executable, os.path.join(HERE, "make_manifest.py"), "--root", root, "--home", home, "--scanner", "py"]
        o5 = os.path.join(tmp, "cli.json")

        def run(*args):
            return subprocess.run(cli + list(args), capture_output=True, text=True)
        p_g = os.path.join(root, "downloads/occ3d/gts.tar.gz")
        _w(p_r4, b"y" * 5)
        _w(p_g, b"tgz")
        r = run("--out", o5)
        assert r.returncode == 0 and "wrote" in r.stdout and "note: no manifest" in r.stderr and "0 gone" in r.stdout, (r.stdout, r.stderr)
        os.remove(p_r4)
        os.remove(p_g)
        r = run("--out", o5)
        assert r.returncode == 0 and "2 gone" in r.stdout and json.load(open(o5))["archives"][R4]["present"] is False, (r.stdout, r.stderr)
        assert run("--out", o5, "--check").returncode == 0
        r = run("--out", o5, "--archive-check", os.path.join(tmp, "d.json"))
        assert r.returncode == 0 and json.load(open(o5))["archives"][A1]["on_disk_check"] == K(c1), (r.stdout, r.stderr)
        assert run("--out", o5, "--check", "--archive-check", os.path.join(tmp, "d.json")).returncode == 2, "--check with --archive-check is refused"
        _w(os.path.join(tmp, "e.json"), json.dumps({A1: {"differences": 0}}))
        assert run("--out", o5, "--archive-check", os.path.join(tmp, "e.json")).returncode != 0, "a check that does not name its file is refused"
        r = run("--out", o5, "--archive-check", os.path.join(tmp, "c.jsonl"))
        j = json.load(open(o5))
        assert r.returncode == 0 and "on_disk_check" in j["archives"][A1] and "on_disk_check" in j["occ3d"]["files"][G], (r.stdout, r.stderr)
        assert run("--out", o5, "--archive-check", os.path.join(tmp, "nofile.json")).returncode != 0, "a missing --archive-check file is an error"
        r = run("--out", o5, "--fresh")
        assert r.returncode == 0 and R4 not in json.load(open(o5))["archives"], "--fresh drops the archives that are gone"
        print("make_manifest selftest ok: deterministic build; logs (a failed unpack counts for nothing, steps add up); both archive places; a deleted archive stays "
              "(present false, date, SHA-256, on_disk_check) and the next check is clean in every mode; a scan that knows nothing erases nothing, another file under the "
              "same name takes nothing over; checks for a wrong file are refused; a vanished data file is not carried; a wrong root is refused; data_identity unchanged by "
              "a deletion; the command line (carry, --fresh, --check, --archive-check shapes)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="the data root (default %s)" % DEFAULT_ROOT)
    ap.add_argument("--home", default=None, help="home folder whose data/ holds the WSL views (default ~)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--scanner", choices=("auto", "ps", "py"), default="auto")
    ap.add_argument("--no-deep", action="store_true", help="skip the folder fingerprints")
    ap.add_argument("--check", action="store_true", help="compare the disk with --out instead of writing it")
    ap.add_argument("--quick", action="store_true", help="with --check: sizes only, no SHA-256")
    ap.add_argument("--archive-check", metavar="FILE", help="the result of comparing archives with the data (verify_archive.ps1): a folder of <name>.json, a JSON / JSONL file of "
                    "results with an \"archive\" key, or {name: result}; merged into the entry's on_disk_check (refused when it was made on another file: bytes / SHA-256)")
    ap.add_argument("--fresh", action="store_true", help="do not carry archives that are gone from the disk over from the existing --out (this also drops their on_disk_check)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.check and a.archive_check:
        ap.error("--archive-check goes with writing the manifest, not with --check")
    warn = lambda s: print("WARNING: " + s, file=sys.stderr)
    if a.check:
        d = check(a.out, a.root, a.home, deep=not a.no_deep, scanner=a.scanner, quick=a.quick)
        print("manifest check: %d difference(s)" % len(d))
        for x in d[:50]:
            print("  " + x)
        return 1 if d else 0
    previous = None
    if a.fresh:
        pass
    elif os.path.isfile(a.out):
        with open(a.out, encoding="utf-8") as f:
            previous = json.load(f)
    else:
        print("note: no manifest at %s yet: archives that are gone from the disk cannot be carried over" % a.out, file=sys.stderr)
    checks = load_checks(a.archive_check) if a.archive_check else None
    m = build(a.root, a.home, deep=not a.no_deep, scanner=a.scanner, previous=previous, checks=checks, warn=warn)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(dump(m))
    os.replace(tmp, a.out)
    gone = sum(1 for v in m["archives"].values() if not v.get("present", True)) + sum(1 for v in m["occ3d"]["files"].values() if v.get("present") is False)
    print("archives: %d on disk, %d gone (kept from the previous manifest)" % (sum(1 for v in m["archives"].values() if v.get("present", True)), gone))
    print("wrote %s (%d bytes); sha256 %s%s" % (a.out, os.path.getsize(a.out), sha256_file(a.out), "; data_identity " + m["data_identity"][:16] if "data_identity" in m else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
