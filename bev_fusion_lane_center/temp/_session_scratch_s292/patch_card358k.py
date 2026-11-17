import io
p = "experiments/E358_wiring-head-on-attached-ends/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)
rep("the ms of the host-side candidate lists are NOT measured. **D109 / D106 line:** 'simpler AND faster?' is NO on parameters (+545) and unmeasured on ms;", "**the ms of the wiring stage, measured 2026-11-16 13:55 (`E358.time_wiring`, run.py job, host QUIET: load 1.8 / 1.5 on the 12 cores, no other job of the problem; `temp/e358/time_wiring.py`, `temp/e358/timing_wiring.json`; host side, torch with ONE thread; per VAL frame, 873 frames x 5 rounds, order alternating, median over rounds, paired difference with a bootstrap over frames; the read-out of B's graph and the pairs are not in it; equality first: curves 11729 / 11880 and wired-at-both-ends 6210 / 6412 equal the stored cards for both arms): rule wiring 4.27 / 4.29 ms per frame, learned head 6.29 / 6.36 ms, so the head costs +2.02 ms [+1.93, +2.11] / +2.06 ms [+1.98, +2.15] per frame (seed 0 / seed 1; 13.4 / 13.6 pairs per frame); the head's cost is the candidate lists with their attributes about 1.0 ms and the forward of the 545-parameter head about 3.8 - 3.9 ms (torch call overhead on 27 ends, not arithmetic: a batched or numpy forward would be a separate, unmeasured optimisation), `add_connectors` with the chosen pieces 1.36 ms (the rule's own piece scan inside `add_connectors` is what the rule arm pays instead).** **D109 / D106 line:** 'simpler AND faster?' is NO on both counts: +545 parameters and about +2.0 ms per frame on the host (about +10 % of the 20.6 ms of E324 s measured in E351 3b; the GPU part is unchanged);")
rep("the first end-to-end replacement of a named rule step (R1) on the read-out that D109 reports as the system.", "the first end-to-end replacement of a named rule step (R1) on the read-out that D109 reports as the system, at a price of about +2.0 ms per frame on the host (measured, QUIET).")
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
