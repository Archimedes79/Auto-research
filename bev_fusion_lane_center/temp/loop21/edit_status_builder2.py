p = "reports/build/reports/build_status21_report.py"
s = open(p, encoding="utf-8").read()
R = [
 ('d.h1("2  Were the experiments conducted correctly?", new_page=True)', 'd.h1("2  Were the experiments conducted correctly?", new_page=False)'),
 ('d.h1("3  Experiments in brief", new_page=True)', 'd.h1("3  Experiments in brief", new_page=False)'),
 ("It is the input of the map registration and a feature of the builder; it is a pixel-based head.", "Its maps are an input of the builder (next to the BEV features) and of the map registration; it is a pixel-based head."),
 ("so that the lane project trains in about an hour on one 12 GB GPU.", "so that the lane project trains in short runs on one 12 GB GPU."),
]
for a, b in R:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
