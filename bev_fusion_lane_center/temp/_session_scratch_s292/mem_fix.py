p = "/mnt/c/Users/mmeut/.claude/projects/C--Auto-research/memory/lane-night-d107-2026-11-16.md"
s = open(p, encoding="utf-8").read()
a = "quiet-host ms (D106 'faster') was the open fact (watcher)."
assert a in s
s = s.replace(a, "quiet-host ms measured at 04:42 (QUIET): n1 - E324 s = -0.14 ms [-0.56, +0.21], NOT faster, so D106 does not choose it and the champion stays E324 s (a MACs-only narrowing is Mirko's case; the junction layer is launch-bound, narrowing keeps every kernel). E351 quiet eager (12 frames): E312 15.13, E341 14.29, E324 s 20.58, E348 19.86 ms (E341 / E348 -0.8 ms).")
b = "the trained stage (build 2 + ~6 slot hours) was not started."
assert b in s
s = s.replace(b, "channel-level anchor (E345 semantics) T2 only -0.04 / -0.02 on B plain: the whole-group collapse is a granularity effect; ranked vs random +0.254 (seed 0) / +0.010 (seed 1); the trained stage (build 2 + ~6 slot hours) was not started. The chain ran 22 stages by 04:26, earlier than the 07:30 estimate, so the quiet-host window opened at 04:37 instead of 05:00 (QUIET_NOW=1 of the watcher).")
open(p, "w", encoding="utf-8").write(s)
print("ok")
