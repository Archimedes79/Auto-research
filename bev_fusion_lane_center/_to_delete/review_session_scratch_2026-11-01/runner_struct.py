p = "bev_fusion/project.json"
s = open(p, encoding="utf-8").read()
for old, new in (('"records", "results", "runs", "temp", "tests", "training"]', '"records", "results", "runner", "runs", "temp", "tests", "training"]'),
                 ('"records", "results", "runs", "temp", "training"]', '"records", "results", "runner", "runs", "temp", "training"]')):
    assert s.count(old) == 1, old
    s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
