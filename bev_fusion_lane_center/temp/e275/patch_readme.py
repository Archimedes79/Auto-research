p = "evaluation/README.md"
s = open(p, encoding="utf-8").read()
old = "`test_verdict275.py` (the E275 rule on fake cards: KEEP / TIE / CUT / INCONCLUSIVE, every guard, the seed check);"
assert s.count(old) == 1
s = s.replace(old, old + " `test_qgraph_score_glue.py` (the GPU-only glue of `qgraph_run_eval.score` on a toy lane, CPU: cards per read-out, kill criterion, reproduction gate, curve, dump);")
old = "(one card per read-out in `runs/e275/<run>.<spec>/`, the kill criterion of the card, a reproduction check against the stored card)"
assert s.count(old) == 1
s = s.replace(old, "(`--exp` required; one card per read-out in `runs/e275/<run>.<spec>/`, the kill criterion of the card, `--repro` = a reproduction gate against the E274 sweep, a check against the stored card)")
open(p, "w", encoding="utf-8").write(s)
print("readme ok")
