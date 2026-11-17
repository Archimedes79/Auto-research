"""append the per-head decode choices of E367 to the card (before the VAL score of those heads); usage: python3 put_choice.py '<text line>'"""
import sys
p = "experiments/E367_e365-capacity-balance-schedule-arms/experiment.md"
s = open(p, encoding="utf-8").read()
line = sys.argv[1]
a = "## 3. Summary\n(empty until the runs exist)"
if a in s:
    s = s.replace(a, "## 3. Summary\n**Decode choices per head (each written BEFORE that head's VAL score; the rule of E366 on the head's own 291-frame TRAIN sample, `temp/e366/choose.py <run>`):**\n- " + line + "\n\n(results follow)")
else:
    b = "\n\n(results follow)"
    assert s.count(b) == 1
    s = s.replace(b, "\n- " + line + b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
