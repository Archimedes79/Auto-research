def rd(p):
    return open(p, encoding="utf-8").read()


def sub(s, a, b):
    assert s.count(a) == 1, a[:80]
    return s.replace(a, b)


p = "training/grid/README.md"
s = rd(p)
s = sub(s, "the pair head must then run `--no-evidence`. Tests: `python jmaps.py`,", "the pair head must then run `--no-evidence`. E339 (D84, simplify): `cwd` (channel-wise distillation, spatial softmax per channel, KL teacher -> student) and `distill_loss` (CWD on the entry / exit heat logits + the heat-weighted heading cosine to a frozen wider head on the same pyramid), `jmaps_train.py --teacher <maps run> [--distill-w 1.0]`, default off. Tests: `python jmaps.py` (incl. the distillation: self-CWD 0, a width-32 student follows a width-64 teacher, no gradient to the teacher),")
open(p, "w", encoding="utf-8", newline="\n").write(s)
p = "training/README.md"
s = rd(p)
s = sub(s, "E319 `--dirx`: the heading-guided exchange block), `qnode_pilot.py`", "E319 `--dirx`: the heading-guided exchange block; E339 `--teacher`: a narrower head taught by a wider one, channel-wise distillation), `qnode_pilot.py`")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
