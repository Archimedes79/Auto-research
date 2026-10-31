"""Cleanup of what autogt generates (videos, pictures, old outputs, caches of outdated generator versions):

  python -m autogt clean --videos --pictures --reports --old-outputs --stale-caches DIR | --all-caches DIR [--older-than DAYS] [--move | --delete]

Without --move / --delete it only lists what it would remove and the size (dry run). --move puts the items into
bev_fusion/_to_delete/autogt_cleanup_<date>/ (the project rule: nothing is deleted without a trace), --delete removes them.
- videos / pictures: *.mp4 / *.gif and *.png in autogt/examples (they are reproducible: `python -m autogt video / view / sheets`)
- reports: *.pdf in autogt/reports (what `python -m autogt report` writes by default; the report is self-contained, its sample pictures are
  drawn straight into the PDF and never exist as files)
- old-outputs: bev_fusion/_to_delete/autogt_old_outputs (pictures and videos made before the suite existed)
- stale-caches DIR: the cache folders `DIR/<signature>` of `python -m autogt build` / `report` whose signature is not a current
  one (the default options or `--boxes-at annotation`; every edit of the generator changes the signature, GT of older versions
  is never read again); all-caches DIR: every
  cache folder in DIR, the current one included (stored GT is derived data and can always be rebuilt)
Never touches code, the current cache, or anything outside these categories.
"""
import datetime
import os
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                                   # the bev_fusion folder


def _size(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)


def collect(videos=False, pictures=False, reports=False, old_outputs=False, stale_caches=None, all_caches=None, older_than=None,
            base=BASE, current_signature=None):
    """list of (path, bytes, category) the cleanup would remove."""
    items = []
    ex = os.path.join(base, "autogt", "examples")
    if os.path.isdir(ex):
        for f in sorted(os.listdir(ex)):
            low = f.lower()
            if (videos and low.endswith((".mp4", ".gif"))) or (pictures and low.endswith(".png")):
                items.append((os.path.join(ex, f), "videos" if videos and low.endswith((".mp4", ".gif")) else "pictures"))
    rep = os.path.join(base, "autogt", "reports")
    if reports and os.path.isdir(rep):
        items += [(os.path.join(rep, f), "reports") for f in sorted(os.listdir(rep)) if f.lower().endswith(".pdf")]
    old = os.path.join(base, "_to_delete", "autogt_old_outputs")
    if old_outputs and os.path.isdir(old):
        items.append((old, "old-outputs"))
    if all_caches and os.path.isdir(all_caches):
        items += [(os.path.join(all_caches, d), "cache") for d in sorted(os.listdir(all_caches))
                  if os.path.exists(os.path.join(all_caches, d, "index.json"))]          # only what autogt marked as its cache
    if stale_caches and os.path.isdir(stale_caches):
        if current_signature is None:
            from .suite import current_signature as sig_of
            # the signatures of the current source with the default options and with the annotation-time boxes
            current_signature = {sig_of(boxes_at=b) for b in ("camera", "annotation")}
        keep = {current_signature} if isinstance(current_signature, str) else set(current_signature)
        for d in sorted(os.listdir(stale_caches)):
            p = os.path.join(stale_caches, d)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "index.json")) and d not in keep:
                items.append((p, "stale-cache"))
    if older_than is not None:
        limit = time.time() - older_than * 86400
        items = [(p, c) for p, c in items if os.path.getmtime(p) < limit]
    return [(p, _size(p), c) for p, c in items]


def clean(delete=False, move=False, base=BASE, **kw):
    items = collect(base=base, **kw)
    total = sum(s for _, s, _ in items)
    for p, s, c in items:
        print(f"{c:12s} {s / 1e6:9.1f} MB  {p}")
    print(f"{len(items)} items, {total / 1e6:.1f} MB" + ("" if (delete or move) else "  (dry run: add --move or --delete)"))
    if move:
        dest = os.path.join(base, "_to_delete", f"autogt_cleanup_{datetime.date.today().isoformat()}")
        os.makedirs(dest, exist_ok=True)
        for p, _, _ in items:
            target = os.path.join(dest, os.path.basename(p))
            shutil.move(p, target if not os.path.exists(target) else target + f"_{int(time.time())}")
        print("moved to", dest)
    elif delete:
        for p, _, _ in items:
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        print("deleted")
    return items


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as b:
        ex = os.path.join(b, "autogt", "examples")
        os.makedirs(ex)
        os.makedirs(os.path.join(b, "_to_delete", "autogt_old_outputs"))
        for f in ("a.mp4", "b.png", "make_videos.sh"):
            open(os.path.join(ex, f), "wb").write(b"x" * 100)
        os.makedirs(os.path.join(b, "autogt", "reports"))
        open(os.path.join(b, "autogt", "reports", "r.pdf"), "wb").write(b"x" * 100)
        open(os.path.join(b, "autogt", "reports", "notes.txt"), "wb").write(b"x")
        cache = os.path.join(b, "cache")
        for sig in ("old1", "cur"):
            os.makedirs(os.path.join(cache, sig))
            open(os.path.join(cache, sig, "index.json"), "w").write("{}")
        got = collect(videos=True, base=b)
        assert [os.path.basename(p) for p, _, _ in got] == ["a.mp4"], got                  # the script is never listed
        assert [os.path.basename(p) for p, _, _ in collect(reports=True, base=b)] == ["r.pdf"]       # only pdf reports
        assert sorted(os.path.basename(p) for p, _, _ in collect(all_caches=cache, base=b)) == ["cur", "old1"]
        got = collect(stale_caches=cache, base=b, current_signature="cur")
        assert [os.path.basename(p) for p, _, _ in got] == ["old1"], got                   # the current cache stays
        clean(move=True, base=b, videos=True, pictures=True, stale_caches=cache, current_signature="cur")
        assert not os.path.exists(os.path.join(ex, "a.mp4")) and os.path.exists(os.path.join(ex, "make_videos.sh"))
        assert os.path.exists(os.path.join(cache, "cur")) and not os.path.exists(os.path.join(cache, "old1"))
        moved = os.path.join(b, "_to_delete", f"autogt_cleanup_{datetime.date.today().isoformat()}")
        assert sorted(os.listdir(moved)) == ["a.mp4", "b.png", "old1"], os.listdir(moved)
        clean(delete=True, base=b, reports=True)
        assert not os.path.exists(os.path.join(b, "autogt", "reports", "r.pdf")) and os.path.exists(os.path.join(b, "autogt", "reports", "notes.txt"))
        clean(delete=True, base=b, old_outputs=True)
        assert not os.path.exists(os.path.join(b, "_to_delete", "autogt_old_outputs"))
