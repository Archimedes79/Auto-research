import os, sys
H = os.path.dirname(os.path.abspath(__file__))
R = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(R, "evaluation", "tests"))
sys.path.insert(0, os.path.join(H, "evaluation"))
import jtest_eval as J
assert J.__file__.startswith(H), J.__file__
sys.path.insert(1, os.path.join(R, "evaluation", "tests"))
import test_jtest_eval as T
T.J = J
fails = []
for k, f in sorted(vars(T).items()):
    if k.startswith("test_") and k != "test_e321_artefacts_unchanged":
        try:
            f()
        except BaseException as e:
            fails.append(k)
print("FAILED:", fails)
