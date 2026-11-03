"""which mutants does the REAL-GT part of the gate kill on its own (the unit checks switched off)?"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests"))
import test_centreline_eval as T

T.unit_checks = lambda ck: None
samples = T.GC.pick(8, "15efbde2d1", 0)
for label, patch in T.mutants():
    undo = patch()
    q = T.Fails(loud=False)
    try:
        T.run_all(q, samples)
        res = f"killed [{q[0][:80]}]" if q else "SURVIVED by the real-GT part"
    except Exception as e:                                         # noqa: BLE001
        res = f"killed (exception {type(e).__name__})"
    finally:
        undo()
    print(f"{label:60s} {res}")
