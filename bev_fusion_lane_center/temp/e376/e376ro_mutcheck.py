"""throw-away mutation check of the critic's items: each textual mutant of the scratch read-out must make its test fail"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = sys.argv[:1]
import e376ro_test_jtoken_eval as T  # noqa: E402

E = T.E
MUTANTS = [("ref run read at the record decode (item 2)", E.score, 'score_dump(rr, a, dec, frames, log=quiet)', 'score_dump(rr, a, "rec", frames, log=quiet)', T.test_argmax_keep),
           ("same_tokens deleted (item 3)", E.read_ref, "    same_tokens(r, rr)\n", "", T.test_ref_run),
           ("no floor rule (item 1)", E.check_argmax, "        if keep_thr > KEEP_THR:", "        if False:", T.test_argmax_keep),
           ("concentration: G without the class gate (item 4)", E.head_concentration, '((y > 0)[:, None] & out["cand"])[0]', '((y > 0)[:, None] | (out["cand"] & False))[0]', T.test_anatomy),
           ("zero-mass guard removed (nit)", E.group_concentration, "ok = rows & (mass > 0)", "ok = rows", T.test_anatomy)]
bad = 0
for name, fn, old, new, test in MUTANTS:
    with T.patched((E, fn.__name__, T._mutant(fn, old, new))):
        try:
            test()
        except Exception as e:  # noqa: BLE001
            print(f"KILLED  {name}: {type(e).__name__}: {str(e)[:120]}")
            continue
    bad += 1
    print(f"SURVIVED {name}")
print("all mutants killed" if not bad else f"{bad} mutants survived")
