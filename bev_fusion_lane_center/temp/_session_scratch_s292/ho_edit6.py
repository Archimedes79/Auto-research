import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("**E343** (TEST of E324 s): code critic done, a coder closes the test gaps (VAL only); then critic, criterion 2, log row + grant, ONE run; R1 - R3 confirmed by Mirko (D95).",
    "**E343** (TEST of E324 s): code committed (abe68bfd, 3 critic passes, 38 tests); FREEZE: the 71 files of its code digest (jtest_eval, jwire*, jends*, qends_eval ...) must not change until the TEST run ends; VAL rehearsal = chain day26 (since 04:57, `temp/day26/*.done`), then log row + grant + ONE run (R1 - R3 confirmed, D95).")
rep("**Next:** 1. E343 (Mirko's OK): code round 4, VAL rehearsal day26, log row + ONE TEST run; 2. E350 phase A commit (phase B after E343's TEST: digest); E345 results;",
    "**Next:** 1. E343 (Mirko's OK): rehearsal result, log row, ONE TEST run; 2. E350 phase A committed (10518e70), phase B after E343's TEST; E345 results;")
rep("day26 = E343 rehearsal (not started).", "day26 = E343 rehearsal.")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
