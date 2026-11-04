"""Cleanup of what autogt generates (videos, pictures, old outputs, caches of outdated generator versions):

  python -m autogt clean --videos --pictures --reports --old-outputs --stale-caches DIR | --all-caches DIR [--older-than DAYS] [--move | --delete]

Without --move / --delete it only lists what it would remove and the size (dry run). --move puts the items into
_to_delete/autogt_cleanup_<date>/ (the project rule: nothing is deleted without a trace), --delete removes them.
- videos / pictures: *.mp4 / *.gif and *.png in data/autogt/examples (they are reproducible: `python -m autogt video / view / sheets`)
- reports: autogt_report_*.pdf in data/autogt/reports (what `python -m autogt report` writes by default; the report is self-contained,
  its sample pictures are drawn straight into the PDF and never exist as files; other PDFs there are not touched)
(everything autogt generates lives under data/autogt/, outside the code package autogt/ and outside git)
- old-outputs: _to_delete/autogt_old_outputs (pictures and videos made before the suite existed)
- stale-caches DIR: the cache folders `DIR/<signature>` of `python -m autogt generate` / `build` / `report` whose signature is not a
  current one (the default options or `--boxes-at annotation`; every edit of the generator changes the signature); all-caches DIR: every
  cache folder in DIR, the current one included. A folder is a cache only if its index.json is a record of autogt (a dict whose
  signature, when it has one, is the folder name). The signatures of released GT versions (versions.csv, D29) are left alone, because
  an experiment on that version needs them (a rebuild takes hours): `--include-released` lists them too, and then `--delete` is refused,
  they can only be moved
Never touches code, the current cache, or anything outside these categories.
"""
import datetime
import json
import os
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                                   # the repository root
RUNNING_S = 3600                                               # a folder with a provisional index that changed within this long may have a build running


def _running(p):
    """True for a cache folder whose index is provisional (`build` writes it before the first keyframe, "complete": false until the end)
    and that was written to within the last hour: a build may still be running in it, so it is not offered as a stale cache"""
    try:
        with open(os.path.join(p, "index.json")) as f:
            idx = json.load(f)
    except (OSError, ValueError):
        return False
    return isinstance(idx, dict) and idx.get("complete") is False and time.time() - os.path.getmtime(p) < RUNNING_S


def _is_cache(p):
    """a folder of stored GT: it has an index.json that is a record of autogt (a dict whose signature, when it has one, is the folder name)"""
    try:
        with open(os.path.join(p, "index.json")) as f:
            idx = json.load(f)
    except (OSError, ValueError):
        return False
    return isinstance(idx, dict) and idx.get("signature", os.path.basename(p)) == os.path.basename(p)


def _size(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)


def collect(videos=False, pictures=False, reports=False, old_outputs=False, stale_caches=None, all_caches=None, older_than=None,
            base=BASE, current_signature=None, include_released=False):
    """list of (path, bytes, category) the cleanup would remove; category "released-cache" only with include_released."""
    from .versions import table
    released = {r["signature"] for r in table()}
    items = []
    ex = os.path.join(base, "data", "autogt", "examples")
    if os.path.isdir(ex):
        for f in sorted(os.listdir(ex)):
            low = f.lower()
            if (videos and low.endswith((".mp4", ".gif"))) or (pictures and low.endswith(".png")):
                items.append((os.path.join(ex, f), "videos" if videos and low.endswith((".mp4", ".gif")) else "pictures"))
    rep = os.path.join(base, "data", "autogt", "reports")
    if reports and os.path.isdir(rep):
        items += [(os.path.join(rep, f), "reports") for f in sorted(os.listdir(rep))
                  if f.startswith("autogt_report_") and f.lower().endswith(".pdf")]
    old = os.path.join(base, "_to_delete", "autogt_old_outputs")
    if old_outputs and os.path.isdir(old):
        items.append((old, "old-outputs"))

    def cache_items(root, keep, category):
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d)
            if not os.path.isdir(p) or d in keep or not _is_cache(p) or _running(p):      # only what autogt marked as its cache, and not a
                continue                                                                   # build that may still be running
            if d in released:
                if include_released:
                    items.append((p, "released-cache"))
            else:
                items.append((p, category))
    if all_caches and os.path.isdir(all_caches):
        cache_items(all_caches, set(), "cache")
    if stale_caches and os.path.isdir(stale_caches):
        if current_signature is None:
            from .suite import current_signature as sig_of
            # the signatures of the current source with the default options and with the annotation-time boxes
            current_signature = {sig_of(boxes_at=b) for b in ("camera", "annotation")}
        cache_items(stale_caches, {current_signature} if isinstance(current_signature, str) else set(current_signature), "stale-cache")
    if older_than is not None:
        limit = time.time() - older_than * 86400
        items = [(p, c) for p, c in items if os.path.getmtime(p) < limit]
    return [(p, _size(p), c) for p, c in items]


def clean(delete=False, move=False, base=BASE, **kw):
    items = collect(base=base, **kw)
    total = sum(s for _, s, _ in items)
    for p, s, c in items:
        print(f"{c:14s} {s / 1e6:9.1f} MB  {p}")
    print(f"{len(items)} items, {total / 1e6:.1f} MB" + ("" if (delete or move) else "  (dry run: add --move or --delete)"))
    if delete and any(c == "released-cache" for _, _, c in items):
        raise ValueError("a released GT version (versions.csv) is never deleted, a rebuild takes hours: use --move")
    if move:
        dest = os.path.join(base, "_to_delete", f"autogt_cleanup_{datetime.date.today().isoformat()}")
        os.makedirs(dest, exist_ok=True)
        for p, _, _ in items:
            target = os.path.join(dest, os.path.basename(p))
            shutil.move(p, target if not os.path.exists(target) else target + f"_{int(time.time())}")
        with open(os.path.join(dest, "README.md"), "a", encoding="utf-8") as f:
            f.write(f"# autogt cleanup {datetime.date.today().isoformat()}\n\nMoved here by `python -m autogt clean --move` "
                    "(videos, pictures, reports, old outputs, caches of other generator versions). To restore, move a folder back; to free "
                    "the space, delete this folder once nothing needs it.\n\n" + "".join(f"- {os.path.basename(p)} ({c})\n" for p, _, c in items))
        print("moved to", dest)
    elif delete:
        for p, _, _ in items:
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        print("deleted")
    return items


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as b:
        ex = os.path.join(b, "data", "autogt", "examples")
        os.makedirs(ex)
        os.makedirs(os.path.join(b, "_to_delete", "autogt_old_outputs"))
        for f in ("a.mp4", "b.png", "make_videos.sh"):
            open(os.path.join(ex, f), "wb").write(b"x" * 100)
        os.makedirs(os.path.join(b, "data", "autogt", "reports"))
        rep = os.path.join(b, "data", "autogt", "reports")
        for f, n in (("autogt_report_v1.0-mini_2026-11-01.pdf", 100), ("autogt_quality_overview.pdf", 100), ("notes.txt", 1)):
            open(os.path.join(rep, f), "wb").write(b"x" * n)
        cache = os.path.join(b, "cache")
        from .versions import table
        rel = table()[-1]["signature"]                                                     # a released version: kept unless asked
        for sig in ("old1", "cur", rel):
            os.makedirs(os.path.join(cache, sig))
            json.dump({"signature": sig, "complete": True}, open(os.path.join(cache, sig, "index.json"), "w"))
        os.makedirs(os.path.join(cache, "foreign"))                                         # foreign index.json files are no cache:
        json.dump(["not", "autogt"], open(os.path.join(cache, "foreign", "index.json"), "w"))        # a list,
        os.makedirs(os.path.join(cache, "wrong"))
        json.dump({"signature": "other"}, open(os.path.join(cache, "wrong", "index.json"), "w"))     # another folder's signature
        got = collect(videos=True, base=b)
        assert [os.path.basename(p) for p, _, _ in got] == ["a.mp4"], got                  # the script is never listed
        assert [os.path.basename(p) for p, _, _ in collect(reports=True, base=b)] == ["autogt_report_v1.0-mini_2026-11-01.pdf"]   # reports only
        assert sorted(os.path.basename(p) for p, _, _ in collect(all_caches=cache, base=b)) == ["cur", "old1"]
        got = collect(stale_caches=cache, base=b, current_signature="cur")
        assert [os.path.basename(p) for p, _, _ in got] == ["old1"], got                   # the current cache stays, a released one too
        got = collect(stale_caches=cache, base=b, current_signature="cur", include_released=True)
        assert sorted((os.path.basename(p), c) for p, _, c in got) == sorted([("old1", "stale-cache"), (rel, "released-cache")]), got
        try:
            clean(delete=True, base=b, stale_caches=cache, current_signature="cur", include_released=True)
            raise AssertionError("a released version must not be deleted")
        except ValueError:
            assert os.path.isdir(os.path.join(cache, rel)) and os.path.isdir(os.path.join(cache, "old1"))
        clean(move=True, base=b, videos=True, pictures=True, stale_caches=cache, current_signature="cur")
        assert not os.path.exists(os.path.join(ex, "a.mp4")) and os.path.exists(os.path.join(ex, "make_videos.sh"))
        assert os.path.exists(os.path.join(cache, "cur")) and not os.path.exists(os.path.join(cache, "old1")) and os.path.isdir(os.path.join(cache, rel))
        moved = os.path.join(b, "_to_delete", f"autogt_cleanup_{datetime.date.today().isoformat()}")
        assert sorted(os.listdir(moved)) == ["README.md", "a.mp4", "b.png", "old1"], os.listdir(moved)
        clean(delete=True, base=b, reports=True)
        assert (sorted(os.listdir(rep)) == ["autogt_quality_overview.pdf", "notes.txt"]), os.listdir(rep)
        clean(delete=True, base=b, old_outputs=True)
        assert not os.path.exists(os.path.join(b, "_to_delete", "autogt_old_outputs"))
