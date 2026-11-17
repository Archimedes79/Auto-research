import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("VAL rehearsal = chain day26 (since 04:57, `temp/day26/*.done`), then log row", "VAL rehearsal = day26 (dumps) then day27 (rest, 10 GB gate, RSS in `temp/day27/`), then log row")
rep("day26 = E343 rehearsal.", "day26 / day27 = E343 rehearsal. E348.w_s1 crashed at 5038 (ENOMEM), the lead resumed it (card section 3).")
rep("day23 = u_s1 post-processing, day24 = E348,", "day23 = u_s1 post, day24 = E348,")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
