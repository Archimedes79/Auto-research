cd /mnt/c/Auto-research/bev_fusion_lane_center
python3 - <<'EOF'
import importlib.util, os, sys
os.environ["BFLC_ROOT"] = os.getcwd()
def load(p, n):
    spec = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(spec); sys.modules[n] = m; spec.loader.exec_module(m); return m
a = load("training/grid/test_lane_past.py", "t1")
for name, old, new in a.MUTANTS:
    c = open("training/data/lane_past_data.py", encoding="utf-8").read().count(old)
    if c != 1: print("lane_past", c, name)
b = load("training/grid/test_jpast.py", "t2")
for name, old, new in b.MUTANTS:
    c = open("training/grid/jpast.py", encoding="utf-8").read().count(old)
    if c != 1: print("jpast", c, name)
t = load("training/grid/test_jpast_train.py", "t3")
for mod, name, old, new in t.MUTANTS:
    path = t.TRAIN if mod == t.T0 else t.EVAL
    c = open(path, encoding="utf-8").read().count(old)
    if c != 1: print("train/eval", c, name)
print("checked", len(a.MUTANTS), len(b.MUTANTS), len(t.MUTANTS))
EOF
