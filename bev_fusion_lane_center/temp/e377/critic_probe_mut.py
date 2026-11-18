"""critic probe: two textual mutants of near_none that the E377 test may not catch"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training", "grid"))
import test_jtoken as TJ
MUTS = {"heading test on |cos| (opposite twins counted)": [("(cos >= NEAR_COS)", "(cos.abs() >= NEAR_COS)")],
        "anchor without the valid mask (padding with a type counts)": [("asg = valid & (ytype != 0)", "asg = (ytype != 0)")],
        "radius 2.5 m": [("d2 <= NEAR_R ** 2", "d2 <= 2.5 ** 2")],
        "cos >= 0.3": [("(cos >= NEAR_COS)", "(cos >= 0.3)")]}
for name, edits in MUTS.items():
    mod = TJ._mutant(*edits)
    try:
        TJ._check_near(mod, 0.25)
        TJ._check_near(mod, 1.0)
        print("SURVIVES:", name)
    except AssertionError:
        print("caught:", name)
