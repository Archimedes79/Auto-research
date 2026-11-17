p = "temp/_session_scratch_e291/apply_hunks.py"
s = open(p, encoding="utf-8").read()
old1 = '    specs = JP.readouts(a.prior_mode or "clean", a.prior_noise) if m.je.prior is not None else []'
new1 = '    pmode, pnoise = getattr(a, "prior_mode", None), getattr(a, "prior_noise", 0.0)                                                              # (a caller with its own Namespace has neither)\n    specs = JP.readouts(pmode or "clean", pnoise) if m.je.prior is not None else []'
old2 = 'assert m.je.prior is not None or (a.prior_mode is None and not a.prior_noise)'
new2 = 'assert m.je.prior is not None or (pmode is None and not pnoise)'
assert old1 in s and old2 in s
s = s.replace(old1, new1).replace(old2, new2)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
