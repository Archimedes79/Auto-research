import io
p = "experiments/E363_e2e-stack-on-faithful-graphs/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
a = "## 6. Summary"
assert s.count(a) == 1
txt = ("**Promotion (2026-11-16 15:55, after the result):** the scratch tools `make_run.py`, `faithful.py` and `surgery_check.py` named above were promoted into `evaluation/jstack.py` (subcommands `surgery`, `dump`, `check-nodes`, `check-arrays`, `selftest`) with a selftest on toy tensors (22 hand mutants of its pure parts killed), a README row and a short code critic pass (three should-fix items taken: the node comparison cannot pass without comparing, NaN fails, the `dump` patch is restored and cannot double-wrap, the array check compares key sets). `check-arrays` and `check-nodes` (CPU) were re-run on the real E363 folders with the scratch versions' results; **the promoted `dump` was NOT re-run** (the GPU window had closed at 15:30; the E363 dumps were made by the scratch version): the first real use of `jstack.py dump` must be preceded by the fixture dump (`surgery b <seed>`, `dump`, `check-arrays`: it must reproduce the stored E324 s dump bit for bit). The recorded deviation of this card is closed for these three tools; `temp/e358/` and `temp/e364/` scripts remain scratch (named in their cards)." + nl + nl)
s = s.replace(a, txt + a)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
