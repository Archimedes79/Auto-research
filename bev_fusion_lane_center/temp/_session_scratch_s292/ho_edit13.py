import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("E319 s1 gate NOT met (kp +0.0082 < +0.01): no pair stage, s0 pending.", "E319 CLOSED at the gate (both seeds: maps gain < +0.01, no pair stage).")
rep("E318.u_s0 scored 03:17, u_s1 queued behind E348 (till ~05:45);", "E318.u_s1 done 06:44, post-processing day23, verdict after it;")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
