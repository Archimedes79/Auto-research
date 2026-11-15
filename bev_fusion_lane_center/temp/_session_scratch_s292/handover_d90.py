p = "HANDOVER.md"
s = open(p, encoding="utf-8").read()
a = "4. **E299** time card v2 (critic REVISE answered) + **step 0 DONE**"
assert s.count(a) == 1
b = "4. **E299** time card v2 (critic REVISE answered; D90 2026-11-14: temporal fusion only RECURSIVE, one ego-warped state tensor, no second-input form: v2 arm r is the card of record, E336 only a status line and not run) + **step 0 DONE**"
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(a, b))
print("ok")
