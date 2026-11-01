import json, sys
def sub(path, old, new, n=1):
    s = open(path, encoding="utf-8").read()
    assert s.count(old) == n, (path, old[:60], s.count(old))
    open(path, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
    print("ok", path)
S = "bev_fusion_lane_center/temp/review_session_scratch_2026-11-01/"
line_old = "!/bev_fusion/runs/README.md\n"
line_new = line_old + "# the result folder of a run (results/<run id>/: package.json, provenance.json are text records; bundle/ is a copy of code that the run's tag holds)\n/bev_fusion/results/*/bundle/\n"
for p in (".gitignore", S + "gitignore_head2.txt"):
    sub(p, line_old, line_new)
for p in ("bev_fusion/project.json", S + "bf_project_head.json"):
    s = open(p, encoding="utf-8").read()
    for key in ('"root_dirs": ["_to_delete", "data", "evaluation", "experiments", "literature", "models", "plan", "records", "runs", "temp", "tests", "training"]',
                '"folders_with_readme": ["_to_delete", "data", "evaluation", "experiments", "models", "plan", "records", "runs", "temp", "training"]'):
        assert s.count(key) == 1, (p, key[:30], s.count(key))
        s = s.replace(key, key.replace('"records", "runs"', '"records", "results", "runs"'))
    open(p, "w", encoding="utf-8", newline="\n").write(s)
    print("ok", p)
