p = "autogt/README.md"
s = open(p, encoding="utf-8").read()
s = s.replace("python -m autogt clean [--videos] [--pictures] [--old-outputs] [--stale-caches DIR] [--move|--delete]   # dry run by default",
              "python -m autogt report [--version v1.0-mini] [--holdout] [--samples N]   # ALL sequences in one pass: store the GT (data/autogt/<signature>/), run every check, write ONE PDF\n"
              "                                                            # (autogt/reports/): summary, table per sequence, legend, samples drawn straight into the PDF (no picture files)\n"
              "python -m autogt clean [--videos] [--pictures] [--reports] [--old-outputs] [--stale-caches DIR | --all-caches DIR] [--older-than DAYS] [--move|--delete]\n"
              "                                                            # removes only what autogt generated; dry run by default, --move puts it into _to_delete/")
s = s.replace("| `checks.py`, `view.py`, `video.py`, `encode.py`, `selftest.py` |", "| `checks.py`, `view.py`, `video.py`, `report.py`, `clean.py`, `encode.py`, `selftest.py` |")
open(p, "w", encoding="utf-8").write(s)
print("ok")
