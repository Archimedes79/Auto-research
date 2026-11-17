p = "temp/_session_scratch_e291/apply_hunks.py"
s = open(p, encoding="utf-8").read()
a, b = s.index('build("training/data/lane_gt_data.py"'), s.index("# ---", s.index('build("training/data/lane_gt_data.py"')) if False else s.index('build("training/train/jends_train.py"')
seg = s[a:b]
seg = seg.replace("import jprior as JP  # noqa: E402", "import jprior as JPR  # noqa: E402").replace("JP.partners", "JPR.partners").replace("JP.table", "JPR.table").replace("JP.collate", "JPR.collate")
s = s[:a] + seg + s[b:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print(seg.count("JPR"), seg.count("JP."))
