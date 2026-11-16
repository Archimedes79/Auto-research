import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
a = s[s.index("CPU rehearsal FAILED"):s.index("+ grant + ONE run (D95).") + len("+ grant + ONE run (D95).")]
b = "rehearsals: CPU FAILED the pre-registered tolerance, GPU branch (day28) seed 0 PASS, seed 1 FAIL by ONE number (vertices/100 m 4.9e-3 vs 3e-3; stack exact, residual = B's CPU dump): by the card's letter the check FAILED, NO TEST started; the lead / Mirko decide (kill, or a recorded amendment; card section 2); log row drafted (temp/_session_scratch_s292/e343_row.txt)."
s = s.replace(a, b, 1)
a2 = "1. E343 (Mirko's OK): rehearsal result, log row, ONE TEST run;"
assert a2 in s
s = s.replace(a2, "1. E343: decision on the failed check, then log row + ONE TEST run;", 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
