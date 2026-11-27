"""Cleanup of what GtGenerator generates (reports, caches of outdated generator versions):

  python -m GtGenerator clean [--reports] [--stale-caches DIR] [--move | --delete]

Without --move / --delete it only lists what it would remove and the size (dry run). --move puts the items into
<store root>/_to_delete/autogt_cleanup_<date>/ (the project rule: nothing is deleted without a trace; next to the stores, so a move is a rename on
their disk, never a copy across disks). --delete removes only reports (one command rebuilds them); stored GT (caches) is only moved: a store is
deleted only after the catalog lists it and the owner confirmed the list (autoresearch/METHOD.md, hard rule 5; D154).
- reports: autogt_report_*.pdf in <store root>/reports (what `python -m GtGenerator report` writes by default; other PDFs there are not touched).
  The store root is where.py: <data root>/groundtruth, outside the code package and outside git
- stale-caches DIR: the cache folders `DIR/<signature>` of `generate` / `report` whose signature is not a current one (the default options or
  `--boxes-at annotation`; every edit of the generator changes the signature). A folder is a cache only if its index.json is a record of
  GtGenerator (a dict whose signature, when it has one, is the folder name). A released store is never listed, because an experiment on that
  version needs it (a rebuild takes hours): one whose signature is in versions.csv (D29) or lane_versions.csv, or whose index.json says
  `release: true`. Neither is a folder that is a link (<store root>/<store> -> another disk), nor one whose provisional index changed within the
  last hour (a run may still be writing it)
Never touches code, the current cache, or anything outside these categories.
"""
import contextlib
import datetime
import json
import os
import shutil
import time

from .store import read_index

RUNNING_S = 3600                                               # a folder with a provisional index that changed within this long may have a run writing it


def _running(p):
    """True for a cache folder whose index is provisional (`generate` writes it before the first keyframe, "complete": false until the end)
    and that was written to within the last hour: a run may still be going, so it is not offered as a stale cache"""
    return read_index(p).get("complete") is False and time.time() - os.path.getmtime(p) < RUNNING_S


def _is_cache(p):
    """a folder of stored GT: it has an index.json that is a record of GtGenerator (a dict whose signature, when it has one, is the folder name)"""
    idx = read_index(p)
    return bool(idx) and idx.get("signature", os.path.basename(p)) == os.path.basename(p)


def _protected():
    """the signatures that name a released store: every row of versions.csv and lane_versions.csv (the store of a version whose code makes an older GT is that older row's)"""
    from .versions import table
    out = {r["signature"] for r in table()}
    try:
        from . import lane_gt
        out |= {r["signature"] for r in lane_gt.table()}
    except ImportError:                                                  # numpy / shapely missing: the front rows are all this box can know
        pass
    return out


def _size(p):
    """the bytes of a file or of everything below a folder (the stats run on 16 threads: 148 s -> about 10 s for one lane store of 34k files on the 9p mounts)"""
    if os.path.isfile(p):
        return os.path.getsize(p)
    files = [os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs]
    if len(files) < 64:
        return sum(os.path.getsize(f) for f in files)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as pool:
        return sum(pool.map(os.path.getsize, files, chunksize=256))


def collect(reports=False, stale_caches=None):
    """list of (path, bytes, category "reports" | "stale-cache") the cleanup would remove"""
    from . import where
    items = []
    rep = os.path.join(where.root(), "reports")
    if reports and os.path.isdir(rep):
        items += [(os.path.join(rep, f), "reports") for f in sorted(os.listdir(rep))
                  if f.startswith("autogt_report_") and f.lower().endswith(".pdf")]
    if stale_caches and os.path.isdir(stale_caches):
        from .suite import current_signature
        # kept: the signatures of the current source (default options and annotation-time boxes) and of every released version
        keep = {current_signature(boxes_at=b) for b in ("camera", "annotation")} | _protected()
        for d in sorted(os.listdir(stale_caches)):
            p = os.path.join(stale_caches, d)
            if d not in keep and os.path.isdir(p) and not os.path.islink(p) and _is_cache(p) and read_index(p).get("release") is not True and not _running(p):
                items.append((p, "stale-cache"))
    return [(p, _size(p), c) for p, c in items]


def clean(delete=False, move=False, reports=False, stale_caches=None, verbose=True):
    from . import where
    items = collect(reports, stale_caches)
    total = sum(s for _, s, _ in items)
    say = print if verbose else (lambda *a, **k: None)
    for p, s, c in items:
        say(f"{c:14s} {s / 1e6:9.1f} MB  {p}")
    say(f"{len(items)} items, {total / 1e6:.1f} MB" + ("" if (delete or move) else "  (dry run: add --move or --delete)"))
    if delete and any(c == "stale-cache" for _, _, c in items):
        raise ValueError("stored GT is only moved (--move): a store is deleted only after the catalog lists it and the owner confirmed the list (hard rule 5, D154); "
                         "--delete is for reports")
    if move:
        dest = os.path.join(where.root(), "_to_delete", f"autogt_cleanup_{datetime.date.today().isoformat()}")      # next to the stores: a rename, no copy
        os.makedirs(dest, exist_ok=True)
        for p, _, _ in items:
            target = os.path.join(dest, os.path.basename(p))
            shutil.move(p, target if not os.path.exists(target) else target + f"_{int(time.time())}")
        with open(os.path.join(dest, "README.md"), "a", encoding="utf-8") as f:
            f.write(f"# GtGenerator cleanup {datetime.date.today().isoformat()}\n\nMoved here by `python -m GtGenerator clean --move` "
                    "(reports, caches of other generator versions). To restore, move an item back; to free "
                    "the space, delete this folder once nothing needs it.\n\n" + "".join(f"- {os.path.basename(p)} ({c})\n" for p, _, c in items))
        say("moved to", dest)
    elif delete:
        for p, _, _ in items:
            os.remove(p)                                                                    # reports: files
        say("deleted")
    return items


@contextlib.contextmanager
def _data_root(path):
    """RESEARCH_DATA set to `path` for the duration (where.root() is then <path>/groundtruth), restored afterwards"""
    was = os.environ.get("RESEARCH_DATA")
    os.environ["RESEARCH_DATA"] = path
    try:
        yield
    finally:
        if was is None:
            os.environ.pop("RESEARCH_DATA", None)
        else:
            os.environ["RESEARCH_DATA"] = was


def selftest():
    import tempfile
    from . import lane_gt, where
    from .suite import current_signature
    from .versions import table
    with tempfile.TemporaryDirectory() as b, _data_root(b):                                # reports, stores and _to_delete/ lie below where.root()
        root = os.path.join(b, "groundtruth")
        assert where.root() == root, where.root()
        rep = os.path.join(root, "reports")
        os.makedirs(rep)
        for f in ("autogt_report_v1.0-mini_2026-11-01.pdf", "autogt_quality_overview.pdf", "notes.txt"):
            open(os.path.join(rep, f), "wb").write(b"x" * 100)

        def store(folder, name, index):
            os.makedirs(os.path.join(folder, name))
            json.dump(index, open(os.path.join(folder, name, "index.json"), "w"))
        released = table()[0]["signature"]                                                  # an old released version: kept although it is not current
        cur = [current_signature(boxes_at=x) for x in ("camera", "annotation")]
        for sig in ("old1", released, *cur):
            store(root, sig, {"signature": sig, "complete": True})
        store(root, "rel_by_index", {"signature": "rel_by_index", "release": True, "complete": True})     # a release by its index alone (the lane stores have no row for the folder)
        store(root, "foreign", ["not", "GtGenerator"])                                      # foreign index.json files are no cache: a list,
        store(root, "wrong", {"signature": "other"})                                        # another folder's signature
        store(root, "building", {"signature": "building", "complete": False})               # a provisional index written a minute ago: a run may still be going
        store(root, "stopped", {"signature": "stopped", "complete": False})                 # ... and one that nothing has written to for two hours
        os.utime(os.path.join(root, "stopped"), (time.time() - 7200, time.time() - 7200))
        store(b, "linkedsig", {"signature": "linkedsig", "complete": True})                 # a link NAMED like the signature in its index: only the link check keeps it out
        os.symlink(os.path.join(b, "linkedsig"), os.path.join(root, "linkedsig"))
        assert [os.path.basename(p) for p, _, _ in collect(reports=True)] == ["autogt_report_v1.0-mini_2026-11-01.pdf"]   # the other files there are not touched
        got = collect(stale_caches=root)
        assert [(os.path.basename(p), c) for p, _, c in got] == [("old1", "stale-cache"), ("stopped", "stale-cache")], got
        assert {r["signature"] for r in lane_gt.table()} <= _protected(), "the rows of lane_versions.csv protect their stores"
        try:                                                                                # stored GT is only moved
            clean(verbose=False, delete=True, stale_caches=root)
            raise AssertionError("a store must not be deleted")
        except ValueError as e:
            assert "--move" in str(e) and os.path.isdir(os.path.join(root, "old1")), e
        before = sorted(os.listdir(b))
        clean(verbose=False, move=True, stale_caches=root)
        moved = os.path.join(root, "_to_delete", f"autogt_cleanup_{datetime.date.today().isoformat()}")      # a move stays below the store root (a rename on its disk)
        assert sorted(os.listdir(moved)) == ["README.md", "old1", "stopped"] and not os.path.exists(os.path.join(root, "old1")), os.listdir(moved)
        assert sorted(os.listdir(b)) == before, ("nothing is moved out of the store root", os.listdir(b))
        assert all(os.path.isdir(os.path.join(root, s)) for s in (released, *cur, "rel_by_index", "building")), "released, current and running stores stay"
        clean(verbose=False, delete=True, reports=True)
        assert sorted(os.listdir(rep)) == ["autogt_quality_overview.pdf", "notes.txt"], os.listdir(rep)
