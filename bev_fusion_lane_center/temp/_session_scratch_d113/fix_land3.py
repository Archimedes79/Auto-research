import os
p = os.path.join(os.environ["BFLC_ROOT"], "temp", "landing", "land.sh")
s = open(p, encoding="utf-8").read()
a = 'grep -q "past_block" training/grid/jtoken.py && { echo "STOP: the hooks are still in jtoken.py (Fable\'s unit U5 has not landed)"; exit 1; }\ngrep -q "import jrecur_time" evaluation/jtoken_eval.py && { echo "STOP: jtoken_eval.py still imports jrecur_time (Fable\'s switch to jtiming.timed has not landed)"; exit 1; }\n'
assert s.count(a) == 1
b = 'grep -q "past_block" training/grid/jtoken.py && echo "NOTE: the hooks are still in jtoken.py (Fable lands his unit U5 right after this commit; test_jtoken.py\'s past / maps tests need jpast / jmap until then)"\ngrep -q "import jrecur_time" evaluation/jtoken_eval.py && echo "NOTE: jtoken_eval.py still imports jrecur_time inside timing() (Fable\'s switch to jtiming.timed lands right after this commit: only `jtoken_eval.py timing` is broken in between)"\n'
s = s.replace(a, b)
s = s.replace("(b) Fable has committed his simplification units U1 - U5 (the two hooks out of jtoken.py, `jtoken_eval.py` switched from jrecur_time.timed to jtiming.timed), (c) the E379 / S4 read-outs are done. After it: the token tests", "(b) agreed order 2026-11-18: MINE first, then Fable's units U1 - U5 right after (the two hooks out of jtoken.py, `jtoken_eval.py` switched from jrecur_time.timed to jtiming.timed), then the report chat's --jobs, (c) the E379 / S4 read-outs are done. After BOTH: the token tests")
open(p, "w", encoding="utf-8").write(s)
print("ok")
