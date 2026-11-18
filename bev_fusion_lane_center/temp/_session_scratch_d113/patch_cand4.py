import os
p = os.path.join(os.environ["BFLC_ROOT"], "experiments", "E379_next_candidates.md")
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
rep("| one head trained on the GT loss + KL to the mean of two frozen champion-recipe heads; inference unchanged | +0.005 (window -0.002 .. +0.010) | 40 / 50 / 10 | 1.5 | 0.0013 |", "| one head trained on the GT loss + KL on the successor distributions to the mean of two frozen champion-recipe heads; inference unchanged (v2 after the critic: odds 25 %, cost 2.4 slot-hours for the rule-bound part, about 5.0 with the controls) | +0.005 (window -0.002 .. +0.010) | 25 / 65 / 10 | 2.4 | 0.0005 |")
rep("(The cost of the ensemble itself: a second head, about +3.7 ms on the 16.7 ms path at batch 1, 20.4 ms against E324 s 19.7 ms: the reason for E379 to distil.)", "(The cost of the ensemble itself: a second head in sequence adds about 5.2 ms to the 16.6 ms path (the stored NOT-QUIET timing: whole 16.57 - B' 11.37), about 21.8 ms against E324 s 19.7 ms; stacked in one forward it should be less: measured by the job E379.time_e2 FIRST, the lead decides e2 against d by cost, D106. The critic (cards mode, 2026-11-17 ~21:00) also found where the gain lives: averaging only the link probabilities gives +0.0091, averaging only p_none -0.0002: the KD of the card covers the successor softmax only.)")
rep("a third head per seed would show whether the gain keeps growing (E378's x_s0 / x_s1 heads are the next ones);", "a third same-recipe head per seed would show whether the gain keeps growing (none exists: E378's x heads use `--pos-loss max`, another recipe, and are no third member);")
open(p, "w", encoding="utf-8").write(s)
print("ok")
