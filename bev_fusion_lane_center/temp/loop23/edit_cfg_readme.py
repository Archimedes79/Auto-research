p = "training/configs/README.md"
s = open(p, encoding="utf-8").read()
a = "`start` composes the base"
b = ("**Split argument:** `trainsuite start` always passes `--split train` to `run.py`, and this project's `splits` (protected, `project.json`) are val / test / confirm, so the launcher refuses it. Add `--run-arg=--split --run-arg=val` "
     "(the later `--split` wins; a training run that scores VAL at its end is a val run; it was tried end to end with the smoke `e271_cfg_smoke`, 2026-11-03). Adding `train` to `splits` would need a grant under a decided D-entry.\n\n`start` composes the base")
assert s.count(a) == 1
s = s.replace(a, b)
a2 = "--id e271_a_s1 --preset seed1 --hyp"
assert s.count(a2) == 1
s = s.replace(a2, "--id e271_a_s1 --preset seed1 --run-arg=--split --run-arg=val --hyp")
open(p, "w", encoding="utf-8").write(s)
print("ok")
