p = "experiments/E320_soft-attach-replaces-snap/experiment.md"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:60], s.count(old))
    s = s.replace(old, new)
rep("- **Start** (not started; day12b starts it after day11):", "- **Start** (not started; **Friday 2026-11-13 08:30, after E316**: E315's pair stage runs at about 2.3 s/step on a contended CPU and E316 (6000 steps) follows, so the day chain 11 is done at about 06:15 and the start rule of the next chain is 'start by 05:00, else Mon - Fri 08:30 - 15:30'; the result is expected at about 11:00 - 11:30):")
rep("the run E320.a_s<S> has not started.", "the run E320.a_s<S> has not started and is planned for Friday 2026-11-13 08:30 (after E316).")
open(p, "w", encoding="utf-8").write(s)
