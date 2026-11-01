S = "bev_fusion_lane_center/temp/review_session_scratch_2026-11-01/"
def sub(path, old, new):
    s = open(path, encoding="utf-8").read()
    assert s.count(old) == 1, (path, old[:50], s.count(old))
    open(path, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
    print("ok", path)
# autoresearch README: package.py + experiment_flow rows
a1 = ("`manifest <run id>` (the provenance of a training run", "`package.py` (the result package of a run in `<problem>/results/<run id>/` and the clean-up of derived data and old GT stores, D40), `manifest <run id>` (the provenance of a training run")
a2 = ("`loop.md` (P3 + P4, pasted after `/loop`) |", "`loop.md` (P3 + P4, pasted after `/loop`), `experiment_flow.md` (one experiment in six steps: plan card, coding, experiment, training, evaluation, store to the result folder) |")
for old, new in (a1, a2):
    sub(S + "ar_readme_head7.md", old, new)
r_old = "| `runs/` | one folder per training run (not in git) - see `runs/README.md` |"
sub(S + "bf_readme_head7.md", r_old, r_old + "\n| `results/` | the result package of each run (`results/<run id>/`: `package.json`, `provenance.json`; D40) - see `results/README.md` |")
g_old = "!/bev_fusion/runs/README.md\n"
sub(S + "gitignore_head3.txt", g_old, g_old + "# the result folder of a run (results/<run id>/: package.json, provenance.json are text records; bundle/ is a copy of code that the run's tag holds)\n/bev_fusion/results/*/bundle/\n")
p = S + "bf_project_head3.json"
s = open(p, encoding="utf-8").read()
for key in ('"root_dirs": ["_to_delete", "data", "evaluation", "experiments", "literature", "models", "plan", "records", "runs", "temp", "tests", "training"]',
            '"folders_with_readme": ["_to_delete", "data", "evaluation", "experiments", "models", "plan", "records", "runs", "temp", "training"]'):
    assert s.count(key) == 1, key[:30]
    s = s.replace(key, key.replace('"records", "runs"', '"records", "results", "runs"'))
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok", p)
