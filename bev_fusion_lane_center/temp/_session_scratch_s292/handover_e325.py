p = "HANDOVER.md"
s = open(p, encoding="utf-8").read()
def sub(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
sub("E331 evidence removal: not adopted.", "E331 evidence removal and E332 a (maps without the paint branch, which includes E331's removal): not adopted, the loss is the evidence (unsnapped -0.019 / -0.011).")
sub("E325 step 1 (twin + `wire_choice`) done, head not built.", "E325 head BUILT and critic-answered (ca1308d8): VAL cross-validation head +0.015 / +0.024 over the rule on real choices; READY + ACCEPTED written.")
sub("= scheduler E332 a, E333, E334 (READY + ACCEPTED markers); `day15` done. Watch `temp/day14/day14.log`.", "= scheduler E333, E334 (E331, E332 done); `temp/day16/day16.sh` (E325.w_s0/1 + scoring + `temp/e325/verdict_e325.py`, waits for 22:00). `day15` done. Watch `temp/day14/day14.log`, `temp/day16/day16.log`.")
sub("E292 CPU diagnosis (`temp/e292/predfull_*`); E325 pilot / head only with code + critic (GPU night); E299 arms low priority;", "after the E325 night: result section, ledger, HANDOVER (verdict by `verdict_e325.py`); E299 arms low priority;")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print(len(s.encode("utf-8")))
