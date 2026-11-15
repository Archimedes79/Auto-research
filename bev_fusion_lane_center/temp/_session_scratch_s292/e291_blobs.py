import io
import re

rows = io.open("temp/e291/readme_rows.md", encoding="utf-8").read()
OUT = "temp/_session_scratch_s292/blobs_e291"
import os
os.makedirs(OUT, exist_ok=True)

# 1. evaluation/README.md: the boot_pooled sentence appended to the jdiag.py row
m = re.search(r"APPEND to the row of `jdiag.py`.*?\n\n(`boot_pooled`.*?)\n\n", rows, re.S)
assert m
sentence = m.group(1).strip()
p = "evaluation/README.md"
L = io.open(p, encoding="utf-8", newline="").read().split("\n")
idx = [i for i, l in enumerate(L) if l.startswith("| `jdiag.py`")]
assert len(idx) == 1
assert L[idx[0]].rstrip().endswith("|")
L[idx[0]] = L[idx[0]].rstrip()[:-1].rstrip() + " " + sentence + " |"
io.open(os.path.join(OUT, "evaluation_README.md"), "w", encoding="utf-8", newline="").write("\n".join(L))

# 2. training/grid/README.md: the jprior.py row after the jgt/qjunction group (after the qjunction.py row), shortened to what is installed
m = re.search(r"(\| `jprior.py` \|.*?\|)\n", rows, re.S)
assert m
row = m.group(1)
cut = row.index(" The data path:")
row = row[:cut] + " Until the freeze of the E343 digest ends the arms read it through STAGED copies of `jends.py`, `lane_gt_data.py`, `jends_train.py`, `jends_eval.py` (`temp/e291/install_plan.md`: the data path, `JEnds(prior=True)`, the dump read-outs and the `--lane-prior` flag are described there and move into this table at the install). | E291 | `python jprior.py` (known answers for every rule; the mutants of the module and of the staged files: `temp/e291/mutate_e291.py`) |"
p = "training/grid/README.md"
L = io.open(p, encoding="utf-8", newline="").read().split("\n")
idx = [i for i, l in enumerate(L) if l.startswith("| `qjunction.py`")]
assert len(idx) == 1 and not any(l.startswith("| `jprior.py`") for l in L)
L.insert(idx[0] + 1, row)
io.open(os.path.join(OUT, "grid_README.md"), "w", encoding="utf-8", newline="").write("\n".join(L))

# 3. tools/maintenance/run_all_tests.sh: the selftest line of jprior.py after the grid/targets.py line
p = "tools/maintenance/run_all_tests.sh"
s = io.open(p, encoding="utf-8", newline="").read()
anchor = 'run "grid targets (cell targets, heat)" grid/targets.py\n'
assert s.count(anchor) == 1
line = 'run "E291 lane-count prior: the corruption (pure function, rates, junction cases, s / sc read-outs), the features (known answers, real peaks only, presence radius), the gated head (zero start, gradient)" grid/jprior.py\n'
io.open(os.path.join(OUT, "run_all_tests.sh"), "w", encoding="utf-8", newline="").write(s.replace(anchor, anchor + line))
print("ok", os.listdir(OUT))
