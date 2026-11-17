"""throw-away: textual mutants of jtoken_eval (score's --ref-cell plumbing, paired_junction's draws, sweep's junction / long-link fields) run against the edited tests; each must make its test fail.
    jte_mutants.py [first index] [last index]"""
import inspect
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "evaluation", "tests")))
import test_jtoken_eval as T  # noqa: E402

E = T.E
MUTANTS = [
    ("score", 'decode_variant(f, v, lt0, kt0, g0, ltl0), "rec", kt0)', 'decode_variant(f, v, lt0, a.keep_thr, g0, ltl0), "rec", a.keep_thr)', "every", "ref keep = the variant's (decode and writer)"),
    ("score", 'decode_variant(f, v, lt0, kt0, g0, ltl0)', 'decode_variant(f, v, lt0, a.keep_thr, g0, ltl0)', "every", "ref keep = the variant's (decode only)"),
    ("score", '"rec", kt0)', '"rec", a.keep_thr)', "every", "ref keep = the variant's (writer only: e shifted)"),
    ("score", 'decode_variant(f, v, lt0, kt0, g0, ltl0)', 'decode_variant(f, v, lt0, kt0, not a.no_none_gate, ltl0)', "every", "ref gate = the variant's"),
    ("score", 'decode_variant(f, v, lt0, kt0, g0, ltl0)', 'decode_variant(f, v, ltl0, kt0, g0, lt0)', "every", "ref lt0 / ltl0 swapped"),
    ("paired_junction", "B=1000, seed=0):", "B=1000, seed=1):", "numbers", "paired_junction seed 1"),
    ("paired_junction", "B=1000, seed=0):", "B=500, seed=0):", "numbers", "paired_junction B 500"),
    ("paired_junction", "frames]), B, seed)", "frames]), B, 1)", "numbers", "paired_junction draws with seed 1 in the body"),
    ("sweep", 'c["lane"]["junc"][0]', 'c["lane"]["junc"][1]', "sweep", "junction points per frame = GT points"),
    ("sweep", 'c["lane"]["junc"][0]', 'c["lane"]["junc"][2]', "sweep", "junction points per frame = matched points"),
    ("sweep", 'res["nodes"].get(f"link_long_{k}")', 'res["nodes"].get(f"link_long{k}")', "sweep", "node_link_long key typo (None)"),
]

lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, len(MUTANTS) - 1)
result = []
for i, (fn, old, new, test, name) in enumerate(MUTANTS):
    if not lo <= i <= hi:
        continue
    orig = getattr(E, fn)
    src = inspect.getsource(orig)
    assert src.count(old) == 1, (fn, old)
    exec(compile(src.replace(old, new), E.__file__, "exec"), E.__dict__)
    try:
        getattr(T, "test_" + test)()
        verdict = "SURVIVED"
    except Exception as e:  # noqa: BLE001
        verdict = f"killed ({type(e).__name__}: {str(e)[:160]})"
        tb = [x for x in traceback.extract_tb(e.__traceback__) if x.filename.endswith("test_jtoken_eval.py")]
        verdict += f" at test line {tb[-1].lineno}" if tb else ""
    finally:
        setattr(E, fn, orig)
    result.append(f"{i} {name} [{test}]: {verdict}")
    print("MUTANT " + result[-1], flush=True)
print("\n".join(["SUMMARY"] + result))
