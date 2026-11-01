def sub(path, old, new, n=1):
    s = open(path, encoding="utf-8").read()
    assert s.count(old) == n, (path, old[:60], s.count(old))
    open(path, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
    print("ok", path)
S = "bev_fusion_lane_center/temp/review_session_scratch_2026-11-01/"
d = "bev_fusion_lane_center/experiment_execution/DECISIONS.md"
sub(d, "`package <run>` writes `records/runs/<id>/package.json`", "`package <run>` writes `bev_fusion/results/<id>/package.json` (the result folder of the run, with `provenance.json`; `bundle/` beside them is a copy of the network and loss files and not in git)")
row_old = "| `runs/` | one folder per training run (not in git) - see `runs/README.md` |"
row_new = row_old + "\n| `results/` | the result package of each run (`results/<run id>/`: `package.json`, `provenance.json`; D40) - see `results/README.md` |"
for p in ("bev_fusion/README.md", S + "bf_readme_head.md"):
    sub(p, row_old, row_new)
a_old = "`manifest <run id>` (the provenance of a training run"
a_new = "`package.py` (the result package of a run in `<problem>/results/<run id>/` and the clean-up of derived data and old GT stores, D40), `manifest <run id>` (the provenance of a training run"
for p in ("autoresearch/README.md", S + "ar_readme_head6.md"):
    sub(p, a_old, a_new)
