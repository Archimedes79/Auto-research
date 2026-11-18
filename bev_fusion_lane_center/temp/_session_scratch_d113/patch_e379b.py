import os
p = os.path.join(os.environ["BFLC_ROOT"], "experiments", "E379_ensemble-distilled-head", "experiment.md")
s = open(p, encoding="utf-8").read()
a = "To build: the trainer option `--distill-from` / `--distill-weight` (Fable's `jtoken_train.py` / `jtoken.py`:"
b = "To build: the trainer option `--distill-from` / `--distill-weight` (Fable's `jtoken_train.py` / `jtoken.py`; the lead's pointer: reuse the machinery of the existing `--teacher` option of E369 / E370 (`make_loss(..., teacher=...)` builds the target dict `tb` from `jtoken.rule_teacher` and passes it to the loss; here `tb` would hold the teachers' soft distributions instead, a new value of the same code path), the flag name `--distill-from` stays new so that the RECIPE keys of a run say which it is, to be agreed with Fable:"
assert s.count(a) == 1
open(p, "w", encoding="utf-8").write(s.replace(a, b))
print("ok")
