import subprocess
# 1. training/grid/README.md: the jmaps.py row
p = "training/grid/README.md"
s = open(p, encoding="utf-8").read()
a = "Tests: `python jmaps.py`, `python test_jmaps_dirx.py` (flag off = the head before E319, flag on starts as flag off, gradients, geometry). |"
b = "E332 (D84, simplify): ctx suffixes `+nopaint` (no 0.5 m paint branch: -1.48 of 3.26 GMACs, no `j05`, no paint loss) and `+nozone` (zone / core not supervised, `out['skip']`), `jmaps_train.py --no-paint --no-zone`; the pair head must then run `--no-evidence`. Tests: `python jmaps.py`, `python test_jmaps_dirx.py` (flag off = the head before E319, flag on starts as flag off, gradients, geometry), `python test_jmaps_simplify.py` (the E332 flags: golden numbers flag off, the head minus the branch is bit-identical, the loss drops exactly one term, gradients, the pair head on the reduced head, the cost of the variants). |"
assert a in s
open(p, "w", encoding="utf-8").write(s.replace(a, b))
# 2. run_all_tests.sh
p = "tools/maintenance/run_all_tests.sh"
s = open(p, encoding="utf-8").read()
a = 'run "E319 junction maps head: heading-guided exchange (flag off = before, zero start, gradients, geometry)" grid/test_jmaps_dirx.py\n'
b = a + 'run "E332 junction maps head without auxiliary maps (flags off = before, bit-identical reduction, loss terms, gradients)" grid/test_jmaps_simplify.py\nrun "E330 cost of the junction system per block (parameters, MACs)" jmodel_cost.py selftest\nrun "E331 - E335 adoption rule of a removal arm" jrule_removal.py selftest\n'
assert a in s
open(p, "w", encoding="utf-8").write(s.replace(a, b))
# 3. evaluation/README.md: a blob from HEAD + my rows (the working copy holds other chats' uncommitted rows)
head = subprocess.run(["git", "show", "HEAD:bev_fusion_lane_center/evaluation/README.md"], capture_output=True, text=True, encoding="utf-8").stdout
lines = head.split("\n")
i = next(k for k, l in enumerate(lines) if l.startswith("| `jmaneuver.py`"))
rows = ["| `jmodel_cost.py` | E330 (D84, simplify): parameters and MACs per block of the junction system (frozen pyramid, the maps head by branch: context, 1 m, 0.5 m paint, the exchange block; the pair head by part) on CPU at batch 1 with torch FlopCounterMode, from a pair run's checkpoint or a configuration (`--ctx`, `--d`, `--evidence`): what each removal of E331 - E335 saves. `selftest`. |",
        "| `jrule_removal.py` | E331 - E335 (D84, D85): the adoption rule of a removal arm: topo F1 at most 0.01 below the control of the same seed on both seeds and both read-outs (unsnapped = differentiable, snapped = the rule step, `--readouts attach` once E320 is the reference), the guards of the cards held; exit 0 = ADOPT. `selftest`. |"]
lines[i:i] = rows
open("temp/e330/evaluation_README_blob.md", "w", encoding="utf-8", newline="").write("\n".join(lines))
print("ok", i)
