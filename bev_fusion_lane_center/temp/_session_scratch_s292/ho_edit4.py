import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("day21 (E345); day22.sh is a STUB (file-name collision with the lead's chain at 01:23), E318.u_s1 restarted by day25 (01:29), post-processing day23, day24 = E348 chain.",
    "E318.u_s0 scored 03:17, u_s1 queued behind E348 (till ~05:45); E319 s1 gate NOT met (kp +0.0082 < +0.01): no pair stage, s0 pending. day21 = E345, day22.sh a STUB, day23 = u_s1 post-processing, day24 = E348, day26 = E343 rehearsal (not started).")
rep("**Next:** E343 code + checks; E345 results; old-chain step 1 (D97); dead-code batches; results E318, E319, E348; E349 critic + build; E350 (E325 head on E324 s): card v1, critic running.",
    "**Next:** 1. E343 (Mirko's OK): code round 4, VAL rehearsal day26, log row + ONE TEST run; 2. E350 phase A commit (phase B after E343's TEST: digest); E345 results; D97; dead code; E318 / E319 / E348 results; E349.")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
