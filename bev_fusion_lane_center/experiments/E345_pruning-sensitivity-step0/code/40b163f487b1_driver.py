import os, sys
R = os.environ["BFLC_ROOT"]
H = os.environ["MUT_DIR"]
sys.path.insert(0, os.path.join(R, "evaluation", "tests"))
sys.path.insert(0, H)
import jtest_eval as J
assert J.__file__.startswith(H), J.__file__
sys.path.insert(1, os.path.join(R, "evaluation", "tests"))
import test_jtest_eval as T
T.J = J
only = [x for x in os.environ.get("MUT_TESTS", "").split(",") if x]
fails = []
for k, f in sorted(vars(T).items()):
    if k.startswith("test_") and k != "test_e321_artefacts_unchanged" and (not only or k in only):
        try:
            f()
        except BaseException as e:
            fails.append(k)
            break
print("FAILED:", fails)
