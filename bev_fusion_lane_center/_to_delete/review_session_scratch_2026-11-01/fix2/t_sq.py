import glob, sys
sys.path.insert(0, "../autoresearch")
from evalsuite.store import RunStore
from evalsuite.quantitative import same_question
a = glob.glob("runs/t8_ctl_s0/eval/suite/quant/eval_n*"); b = glob.glob("runs/t8_bevfeat_s0/eval/suite/quant/eval_n*")
print(a, b)
A, B = RunStore.open(a[0]) if hasattr(RunStore, "open") else RunStore(a[0]), RunStore.open(b[0]) if hasattr(RunStore, "open") else RunStore(b[0])
try:
    same_question(A, B); print("plain: accepted")
except ValueError as e:
    print("plain: refused:", str(e)[:160])
same_question(A, B, ignore={"source": ("feature_cache", "control")}); print("with ignore: accepted")
