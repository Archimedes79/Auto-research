#!/usr/bin/env python3
"""make_manifest.py - the manifest of the raw nuScenes data: which dataset version, which archives, which files, on D:.

    python3 datasets/NuScenes/tools/make_manifest.py                  # writes datasets/NuScenes/manifest.json from the files on disk (no network, read-only on D:)
    python3 datasets/NuScenes/tools/make_manifest.py --check          # compares the disk with the committed manifest; exit 1 and the differences when they differ
    python3 datasets/NuScenes/tools/make_manifest.py --selftest       # a fake data root in a temporary folder: determinism, known answers, the check finds changes

Why: the data itself is never in git, so a result must be able to NAME the data it was made from. This manifest is small enough for git (about 25 KB) and pins
the data by what cannot change unnoticed: the S3 ETag and size of every archive that was downloaded (from the download logs), the SHA-256 of the metadata tables
(`v1.0-trainval/*.json`, the dataset version itself), of the map expansion, of the licence and of the stored file lists, and, per sensor folder, the number of
files, their bytes and the SHA-256 of the sorted list "path<TAB>size" (so one missing or extra file shows). A result card names the manifest by its SHA-256
(`sha256sum datasets/NuScenes/manifest.json`); the GT built from this data is named by `autogt/versions.csv` and its `data/autogt/<signature>/index.json`
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

SCHEMA = 1
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_ROOT = "/mnt/d/Auto-research/NuScenes"
DEFAULT_OUT = os.path.join(REPO, "datasets", "NuScenes", "manifest.json")
IGNORE = {"generated", "generator"}                      # never part of the comparison
ARCHIVE_DIRS = (("downloads/nuscenes_camera", "camera"), ("downloads/nuscenes_radar", "radar"), ("downloads/nuscenes_radar_456", "radar"))
ABOUT = ("Identity of the raw nuScenes data on D: and of the files that are derived from it by pointer. Made by datasets/NuScenes/tools/make_manifest.py from the files "
         "on disk (no network); `make_manifest.py --check` compares the disk with it. Paths are relative to the data root; \"$REPO\" and \"$ROOT\" in view targets "
         "stand for the repository and the data root. A result names this file by its SHA-256.")


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
    """{archive name: {etag, verified, members}} from a part log: '<time> download <name> (etag <etag>)', 'OK <name>', '<time> unpack to <dest> (only: <members>)'"""
    info, cur = {}, None
    if not os.path.isfile(path):
        return info
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        m = re.match(r"\d\d:\d\d:\d\d download (\S+) \(etag (\S+)\)", line)
        if m:
            cur = info.setdefault(m.group(1), {})
            cur["etag"] = m.group(2)
            continue
        m = re.match(r"OK (\S+)$", line)
        if m:
            info.setdefault(m.group(1), {})["verified"] = True
            continue
        m = re.match(r"\d\d:\d\d:\d\d unpack to \S+(?: \(only: (.*)\))?\s*$", line)
        if m and cur is not None:
            cur["members"] = m.group(1) or "all"
    return info


def scan_archives(root):
    out = {}
    for sub, kind in ARCHIVE_DIRS:
        base = os.path.join(root, sub)
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
                    lg = logs.get(f, {})
                    e = {"kind": kind, "path": rel(p, root), "bytes": os.path.getsize(p), "etag": lg.get("etag"), "etag_verified": bool(lg.get("verified")),
                         "unpacked_marker": os.path.exists(p + ".unpacked"), "unpacked_members": lg.get("members")}
                    out[f] = e
    return out


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
    for dp, dn, fns in os.walk(path):
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
    text, path = None, os.path.join(repo, "datasets", "NuScenes", "tools", "get_occ3d_labels.sh")
    if os.path.isfile(path):
        text = open(path, encoding="utf-8", errors="replace").read()
    else:
        try:
            text = subprocess.check_output(["git", "-C", repo, "show", "nuscenes/tools-history-2026-11-01:datasets/NuScenes/tools/history/get_occ3d_labels.sh"],
                                           text=True, stderr=subprocess.DEVNULL)
        except Exception:
            return {}
    return {m.group(2): m.group(1) for m in re.finditer(r"get_gdrive_file\.sh\"?\s+(\S+)\s+\"[^\"]*/downloads/occ3d/([^\"]+)\"", text)}


def occ3d(root, repo, hash_files=True):
    out = {"drive_ids": drive_ids(repo), "files": {}}
    for fn in ("annotations.json", "gts.tar.gz"):
        p = os.path.join(root, "downloads", "occ3d", fn)
        if os.path.isfile(p):
            out["files"]["downloads/occ3d/" + fn] = file_entry(p, hash_files)
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


LINKS = {"gt": {"registry": "autogt/versions.csv", "tags": "autogt/v<N>", "store": "data/autogt/<signature>/index.json", "check": "python -m autogt data <dataroot>"},
         "lane_additions": "bev_fusion_lane_center/data/rebuild.json",
         "recipes": "datasets/NuScenes/README.md and datasets/NuScenes/tools/ (link_views.sh builds the views)"}


# ---- build, diff, check ---------------------------------------------------------------------------------------------------------------------------------
def build(root, home=None, repo=REPO, deep=True, scanner="auto", hash_files=True):
    root = os.path.abspath(os.path.expanduser(root))
    home = os.path.expanduser(home or "~")
    ds, files = datasets(root, hash_files)
    m = {"schema": SCHEMA, "about": ABOUT,
         "root": {"windows": "D:\\Auto-research\\NuScenes", "wsl": "/mnt/d/Auto-research/NuScenes"},
         "licence": "nuScenes is for non-commercial use (the LICENSE of the data is under files); downloads only with Mirko's OK",
         "source": {"bucket": "motional-nuscenes", "listing": "datasets/NuScenes/tools/list_nuscenes_bucket.py", "etag_check": "datasets/NuScenes/tools/verify_s3_etag.py"},
         "datasets": ds, "files": files, "file_lists": file_lists(root, hash_files), "archives": scan_archives(root), "data_markers": data_markers(root),
         "occ3d": occ3d(root, repo, hash_files), "views": views(home, root, repo), "links": LINKS,
         "generated": time.strftime("%Y-%m-%d")}
    tool = os.path.join(HERE, "make_manifest.py")
    gen = {"tool": "datasets/NuScenes/tools/make_manifest.py", "tool_sha256": sha256_file(tool) if os.path.isfile(tool) else None}
    if deep:
        m["folders"], gen["folder_scanner"] = scan_folders(root, scanner)
    m["generator"] = gen
    return m


def diff(old, new, path="", partial=False):
    """differences between two manifests (dict trees); partial: a key that the new one lacks (not computed in --quick) is not a difference"""
    out = []
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new)):
            if not path and k in IGNORE:
                continue
            p = path + "/" + k
            if k not in new:
                if partial and k == "sha256":                      # --quick computes no hashes
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
    new = build(root, home, repo, deep=deep, scanner=scanner, hash_files=not quick)
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
    try:
        root, home, repo = os.path.join(tmp, "D", "NuScenes"), os.path.join(tmp, "home"), os.path.join(tmp, "repo")
        os.makedirs(os.path.join(repo, "datasets", "NuScenes", "tools"))
        _w(os.path.join(repo, "datasets", "NuScenes", "tools", "get_occ3d_labels.sh"),
           'bash "$here/get_gdrive_file.sh" 1AAA "$root/downloads/occ3d/annotations.json"\nbash "$here/get_gdrive_file.sh" 1BBB "$root/downloads/occ3d/gts.tar.gz"\n')
        _w(os.path.join(root, "downloads/nuscenes_camera/p01/v1.0-trainval01_blobs_camera.tgz"), b"x" * 10)
        _w(os.path.join(root, "downloads/nuscenes_camera/p01/v1.0-trainval01_blobs_camera.tgz.unpacked"))
        _w(os.path.join(root, "downloads/nuscenes_camera/part_01.log"),
           "18:59:44 download v1.0-trainval01_blobs_camera.tgz (etag abc123-2)\n20:55:11 verify v1.0-trainval01_blobs_camera.tgz\nOK v1.0-trainval01_blobs_camera.tgz\n"
           "20:58:04 unpack v1.0-trainval01_blobs_camera.tgz\n21:02:15 unpack to /mnt/d/x/nuscenes_full (only: sweeps)\n21:04:25 done v1.0-trainval01_blobs_camera.tgz\n")
        _w(os.path.join(root, "downloads/nuscenes_radar_456/v1.0-trainval04_blobs_radar.tgz"), b"y" * 5)
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
        m1, m2 = build(root, **kw), build(root, **kw)
        assert dump(m1) == dump(m2), "two builds of the same disk differ"
        a = m1["archives"]
        assert a["v1.0-trainval01_blobs_camera.tgz"] == {"kind": "camera", "path": "downloads/nuscenes_camera/p01/v1.0-trainval01_blobs_camera.tgz", "bytes": 10, "etag": "abc123-2",
                                                         "etag_verified": True, "unpacked_marker": True, "unpacked_members": "sweeps"}, a
        assert a["v1.0-trainval04_blobs_radar.tgz"]["etag"] is None and not a["v1.0-trainval04_blobs_radar.tgz"]["etag_verified"], "no log, no ETag claimed"
        assert m1["datasets"]["v1.0-trainval"]["scenes"] == 2 and m1["datasets"]["v1.0-trainval"]["samples"] == 3 and m1["datasets"]["v1.0-mini"]["samples"] == 2
        assert m1["datasets"]["v1.0-trainval"]["tables"]["scene.json"]["sha256"] == hashlib.sha256(open(os.path.join(root, "nuscenes_full/v1.0-trainval/scene.json"), "rb").read()).hexdigest()
        assert m1["file_lists"]["manifests/nuscenes_full.tsv"]["lines"] == 2
        want = hashlib.sha256(b"a.jpg\t3\nb.jpg\t4\n").hexdigest()                    # known answer, computed without fingerprint()
        assert m1["folders"]["nuscenes_full/samples/CAM_FRONT"] == {"files": 2, "bytes": 7, "list_sha256": want}, m1["folders"]["nuscenes_full/samples/CAM_FRONT"]
        assert m1["folders"]["occ3d/gts"]["files"] == 1 and m1["folders"]["nuscenes_full/maps"]["files"] == 1
        assert m1["data_markers"] == {"nuscenes_full/.v1.0-trainval01_blobs.txt": "v1.0-trainval01_blobs.tgz"}
        assert m1["occ3d"]["drive_ids"] == {"annotations.json": "1AAA", "gts.tar.gz": "1BBB"} and m1["occ3d"]["scenes"] == 1
        assert m1["views"]["~/data/nuscenes_full"]["samples"] == "$ROOT/nuscenes_full/samples" and m1["views"]["~/data/nuscenes"]["samples"] == "$REPO/x"
        out = os.path.join(tmp, "manifest.json")
        _w(out, dump(m1))
        assert check(out, root, **kw) == [], "an unchanged disk must check clean"
        assert check(out, root, quick=True, **kw) == [], "quick check of an unchanged disk"
        assert check(out, root, deep=False, **kw) == []
        _w(os.path.join(root, "nuscenes_full/v1.0-trainval/scene.json"), json.dumps([{"name": "scene-1"}, {"name": "scene-X"}]))
        d = check(out, root, **kw)
        assert any("scene.json/sha256" in x for x in d) and not any("scene.json/bytes" in x for x in d), d
        assert check(out, root, quick=True, **kw) == [], "quick compares sizes only: a same-size change is not seen"
        _w(os.path.join(root, "nuscenes_full/v1.0-trainval/scene.json"), json.dumps([{"name": "scene-1"}, {"name": "scene-2"}]))
        _w(os.path.join(root, "nuscenes_full/samples/CAM_FRONT/d.jpg"), b"d")
        d = check(out, root, **kw)
        assert any("folders/nuscenes_full/samples/CAM_FRONT" in x for x in d), d
        assert check(out, root, deep=False, **kw) == [], "--no-deep does not look at folders"
        os.remove(os.path.join(root, "nuscenes_full/samples/CAM_FRONT/d.jpg"))
        os.remove(os.path.join(root, "downloads/nuscenes_radar_456/v1.0-trainval04_blobs_radar.tgz"))
        d = check(out, root, **kw)
        assert any("v1.0-trainval04_blobs_radar.tgz" in x and "not on disk" in x for x in d), d
        print("make_manifest selftest ok: deterministic build, archives from the logs (an archive without a log claims no ETag), tables / file lists / Occ3D hashed, folder "
              "fingerprint = known answer, views normalised, the check finds a changed table, an extra file and a missing archive")
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
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.check:
        d = check(a.out, a.root, a.home, deep=not a.no_deep, scanner=a.scanner, quick=a.quick)
        print("manifest check: %d difference(s)" % len(d))
        for x in d[:50]:
            print("  " + x)
        return 1 if d else 0
    m = build(a.root, a.home, deep=not a.no_deep, scanner=a.scanner)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(dump(m))
    os.replace(tmp, a.out)
    print("wrote %s (%d bytes); sha256 %s" % (a.out, os.path.getsize(a.out), sha256_file(a.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
