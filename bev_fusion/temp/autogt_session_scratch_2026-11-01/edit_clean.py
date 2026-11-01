p = "autogt/clean.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''  python -m autogt clean --videos --pictures --old-outputs --stale-caches DIR [--older-than DAYS] [--move | --delete]''',
              '''  python -m autogt clean --videos --pictures --reports --old-outputs --stale-caches DIR | --all-caches DIR [--older-than DAYS] [--move | --delete]''')
s = s.replace('''- old-outputs:''', '''- reports: *.pdf in autogt/reports (what `python -m autogt report` writes by default; the report is self-contained, its sample pictures are
  drawn straight into the PDF and never exist as files)
- old-outputs:''')
s = s.replace('''- stale-caches DIR: the cache folders `DIR/<signature>` of `python -m autogt build` whose signature is not the current one
  (every edit of the generator changes the signature, GT of older versions is never read again)''',
'''- stale-caches DIR: the cache folders `DIR/<signature>` of `python -m autogt build` / `report` whose signature is not the current
  one (every edit of the generator changes the signature, GT of older versions is never read again); all-caches DIR: every
  cache folder in DIR, the current one included (stored GT is derived data and can always be rebuilt)''')
s = s.replace("def collect(videos=False, pictures=False, old_outputs=False, stale_caches=None, older_than=None, base=BASE, current_signature=None):",
              "def collect(videos=False, pictures=False, reports=False, old_outputs=False, stale_caches=None, all_caches=None, older_than=None,\n            base=BASE, current_signature=None):")
s = s.replace('''    old = os.path.join(base, "_to_delete", "autogt_old_outputs")''', '''    rep = os.path.join(base, "autogt", "reports")
    if reports and os.path.isdir(rep):
        items += [(os.path.join(rep, f), "reports") for f in sorted(os.listdir(rep)) if f.lower().endswith(".pdf")]
    old = os.path.join(base, "_to_delete", "autogt_old_outputs")''')
s = s.replace('''    if stale_caches and os.path.isdir(stale_caches):''', '''    if all_caches and os.path.isdir(all_caches):
        items += [(os.path.join(all_caches, d), "cache") for d in sorted(os.listdir(all_caches))
                  if os.path.exists(os.path.join(all_caches, d, "index.json")) or os.path.isdir(os.path.join(all_caches, d))]
    if stale_caches and os.path.isdir(stale_caches):''')
s = s.replace('''        for f in ("a.mp4", "b.png", "make_videos.sh"):
            open(os.path.join(ex, f), "wb").write(b"x" * 100)''', '''        for f in ("a.mp4", "b.png", "make_videos.sh"):
            open(os.path.join(ex, f), "wb").write(b"x" * 100)
        os.makedirs(os.path.join(b, "autogt", "reports"))
        open(os.path.join(b, "autogt", "reports", "r.pdf"), "wb").write(b"x" * 100)
        open(os.path.join(b, "autogt", "reports", "notes.txt"), "wb").write(b"x")''')
s = s.replace('''        got = collect(stale_caches=cache, base=b, current_signature="cur")''', '''        assert [os.path.basename(p) for p, _, _ in collect(reports=True, base=b)] == ["r.pdf"]       # only pdf reports
        assert sorted(os.path.basename(p) for p, _, _ in collect(all_caches=cache, base=b)) == ["cur", "old1"]
        got = collect(stale_caches=cache, base=b, current_signature="cur")''')
s = s.replace('''        assert sorted(os.listdir(moved)) == ["a.mp4", "b.png", "old1"], os.listdir(moved)''', '''        assert sorted(os.listdir(moved)) == ["a.mp4", "b.png", "old1"], os.listdir(moved)
        clean(delete=True, base=b, reports=True)
        assert not os.path.exists(os.path.join(b, "autogt", "reports", "r.pdf")) and os.path.exists(os.path.join(b, "autogt", "reports", "notes.txt"))''')
open(p, "w", encoding="utf-8").write(s)

p = "autogt/cli.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    cl.add_argument("--old-outputs", action="store_true")''', '''    cl.add_argument("--reports", action="store_true")
    cl.add_argument("--old-outputs", action="store_true")
    cl.add_argument("--all-caches", metavar="DIR")''')
s = s.replace('''clean(delete=a.delete, move=a.move, videos=a.videos, pictures=a.pictures, old_outputs=a.old_outputs,
              stale_caches=a.stale_caches, older_than=a.older_than)''', '''clean(delete=a.delete, move=a.move, videos=a.videos, pictures=a.pictures, reports=a.reports, old_outputs=a.old_outputs,
              stale_caches=a.stale_caches, all_caches=a.all_caches, older_than=a.older_than)''')
s = s.replace('''    cl = sub.add_parser("clean"''', '''    r = sub.add_parser("report", help="evaluate the GT of all sequences, store it, write one PDF report with samples")
    r.add_argument("--dataroot", default="~/data/nuscenes")
    r.add_argument("--version", default="v1.0-mini")
    r.add_argument("--out", default=None, help="report PDF (default autogt/reports/autogt_report_<version>_<date>.pdf)")
    r.add_argument("--data", default=None, help="cache directory (default bev_fusion/data/autogt); --no-data: do not store")
    r.add_argument("--no-data", action="store_true")
    r.add_argument("--scenes", nargs="*")
    r.add_argument("--stride", type=int, default=1)
    r.add_argument("--samples", type=int, default=10, help="sample pictures in the report (evenly spread over the sequences)")
    r.add_argument("--holdout", action="store_true", help="add the leave-one-radar-out test of the radar correction")
    cl = sub.add_parser("clean"''')
s = s.replace('''    elif a.cmd == "clean":''', '''    elif a.cmd == "report":
        import datetime
        from .report import report
        out = a.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "autogt", "reports",
                                    f"autogt_report_{a.version}_{datetime.date.today().isoformat()}.pdf")
        data = None if a.no_data else (a.data or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "autogt"))
        report(os.path.expanduser(a.dataroot), a.version, out, data, a.scenes, a.stride, a.holdout, a.samples)
    elif a.cmd == "clean":''')
s = s.replace('''  python -m autogt clean  [--videos] [--pictures] [--old-outputs] [--stale-caches DIR] [--older-than DAYS] [--move | --delete]''',
              '''  python -m autogt report [--version v1.0-mini] [--out REPORT.pdf] [--data DIR | --no-data] [--scenes ..] [--stride N] [--samples N] [--holdout]
      GT evaluation of all sequences in one pass: store the GT (as `build`), run every check, write ONE PDF report (summary, table
      per sequence, legend, sample pictures drawn straight into the PDF - no picture files exist, nothing to clean up but the report)
  python -m autogt clean  [--videos] [--pictures] [--reports] [--old-outputs] [--stale-caches DIR | --all-caches DIR] [--older-than DAYS] [--move | --delete]''')
open(p, "w", encoding="utf-8").write(s)

p = "autogt/selftest.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    from . import clean as cleanmod''', '''    from .report import report                                                        # 7 the report: one PDF, no picture files
    with tempfile.TemporaryDirectory() as d:
        rep = report(root, "v1.0-mini", os.path.join(d, "r.pdf"), os.path.join(d, "data"), scenes=["scene-0061", "scene-1077"], stride=20,
                     samples=2, verbose=False)
        left = sorted(os.listdir(d))
        assert left == ["data", "r.pdf"] and os.path.getsize(os.path.join(d, "r.pdf")) > 20000 and len(rep["samples"]) == 2, left
        assert not [f for _, _, fs in os.walk(d) for f in fs if f.lower().endswith((".png", ".jpg"))], "no picture files"
        assert rep["written"] == len(os.listdir(os.path.join(d, "data", rep["signature"]))) - 0, rep
    from . import clean as cleanmod''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
