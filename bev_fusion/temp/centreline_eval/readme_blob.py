"""the blob that commit_paths.py commits for evaluation/README.md: the committed file + only my four rows (the working file is another unit's work in progress, older than HEAD in places)"""
import os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OLD3 = "26 direction arrows, GT cyan / predicted amber), `dashboard` | renders; error colours at the right cells; `../tests/test_centreline_eval.py` (the two panels) |"
NEW3 = "26 direction arrows, GT cyan / predicted amber; gate `../tests/test_centreline_eval.py`), `dashboard` | renders; error colours at the right cells |"
path = os.path.join(ROOT, "evaluation", "README.md")
work = open(path, encoding="utf-8", newline="").read()
if OLD3 in work:
    open(path, "w", encoding="utf-8", newline="").write(work.replace(OLD3, NEW3))

EDITS = [
    ("`BEVPriorSubject` (per-cell GT mean as a constant prediction), the controls | contract checker `selftest-adapter` + `--real` |",
     "`BEVPriorSubject` (per-cell GT mean as a constant prediction), the controls; the 1 m centre-line head (`p[\"sem_1m\"]`, only a model that has it): `predictions_1m` gives its presence "
     "as the layer `centreline` on the 0.5 m grid (sigmoid, each 1 m cell copied to its 2 x 2 cells) and its direction (cos, sin), `task_spec` adds the layer to the Task's rows (a model without the head: "
     "its spec itself, nothing else changes) | contract checker `selftest-adapter` + `--real`; `../tests/test_centreline_eval.py` (the 1 m head: oracle, hand examples, old runs unchanged, 13 mutants) |"),
    ("metrics `seg` (SegMeter counts of the masks fov / seen / out / obj per frame), `curve`",
     "metrics `seg` (SegMeter counts of the masks fov / seen / out / obj per frame; with the 1 m centre-line head also the layer `centreline`: IoU / range bins / dt keys like the lane dividers, "
     "and `centreline:dir` / `angle_mae`, rad, lower is better), `curve`"),
    ("`outputs`, `stages`, `dashboard`",
     "`outputs`, `stages` (a model with the 1 m centre-line head gets two panels more: 25 its presence against the GT within 1 m, 26 direction arrows, GT cyan / predicted amber; "
     "gate `../tests/test_centreline_eval.py`), `dashboard`"),
    ("`SegMeter` (tolerant IoU per layer / range bin / dt, line zero-crossing F1), `DetMeter`",
     "`SegMeter` (tolerant IoU per layer / range bin / dt, line zero-crossing F1, `add_directions`: the mean absolute signed angle of a direction field, `angle_error`), `DetMeter`"),
]
head = subprocess.run(["git", "-C", ROOT, "show", "HEAD:bev_fusion/evaluation/README.md"], capture_output=True, check=True).stdout.decode("utf-8")
for old, new in EDITS:
    assert head.count(old) == 1, (head.count(old), old[:70])
    head = head.replace(old, new)
with open(os.path.join(HERE, "README_eval_mine.md"), "w", encoding="utf-8", newline="") as f:
    f.write(head)
print("ok")
