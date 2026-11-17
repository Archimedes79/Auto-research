p = "experiments/E325_learned-wiring-head/experiment.md"
s = open(p, encoding="utf-8").read()
a = "**Status: card only, no code, nothing started.**"
assert s.count(a) == 1
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(a, "**Status (2026-11-14 07:30): code built (step 1: the twin and the read-out path, step 2: the head, its data, training and read-out), two code critic passes answered (section 5), nothing trained at full size; the night run `E325.w_s0` / `E325.w_s1` follows through `temp/day16/day16.sh` (READY + ACCEPTED written with the commit of step 2).**"))
