p = "HANDOVER.md"
s = open(p, encoding="utf-8").read()
a = "Follow-up: **E337** (card v1, critic running; step 0 CPU: attached ends with the snap's geometry, the gap +0.0069 is geometry or wiring?)."
assert s.count(a) == 1
s = s.replace(a, "Follow-up: **E337** (card v2 after the critic, answered from stored cards: the gap +0.0069 / +0.0062 is GEOMETRY, the same delivered piece is wired for 96 % of the ends; attached ends recover 25 % / 34 % of the snap's geometry; the sharpening lever is a stub, NOT scheduled: decide after E292 and the E325 head).")
open(p, "w", encoding="utf-8", newline="\n").write(s)
