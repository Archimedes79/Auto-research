def sub(path, pairs):
    s = open(path, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:60])
        s = s.replace(a, b)
    open(path, "w", encoding="utf-8").write(s)
sub("experiments/E322_soft-wiring/experiment.md", [
 ("wiring BOTH tied pieces (a `k = 2` weighted wiring, the first member of the next family card)", "wiring BOTH tied pieces (a `k = 2` weighted wiring: D88 strikes it as a read-out variant; it would only come back as a differentiable term)"),
 ("and the family card on the number of pieces per end (k = 2 weighted) when it is wanted; the E316 pair stage (6000 steps) stays a stack candidate.", "(the family card on the number of pieces per end, k = 2 weighted, is struck by D88: a read-out variant is a diagnostic, not a solution); the E316 pair stage (6000 steps) stays a stack candidate."),
])
sub("experiments/E323_duplicates-operating-point/experiment.md", [
 ("Both are read-out only and CPU; a variant with k = 2 weighted pieces is the next member (not planned here).", "Both are read-out only and CPU, that is MEASUREMENTS that explain a defect of an end-to-end candidate (D88: read-outs and counts are diagnostics, not experiments that deliver a system); a variant with k = 2 weighted pieces was named here as a next member and is struck by D88."),
])
sub("experiments/E325_learned-wiring-head/experiment.md", [
 ("anything about ties (wiring both co-located places is a follow-up), a dustbin target, or k = 2 pieces per end;", "anything about ties, a dustbin target, or k = 2 pieces per end (wiring both co-located places is a read-out variant, struck by D88; it could come back only as a differentiable term);"),
])
s = open("HANDOVER.md", encoding="utf-8").read()
a = "Mirko 2026-11-13 evening:"
assert s.count(a) == 1
s = s.replace(a, "**D88** (Mirko ~10:05: \"Only E2E differentiable solutions are searched for\"): every improvement / simplification candidate is an end-to-end differentiable solution; NO new rule step on the output (threshold, de-duplication rule, wiring variant, snap variant, operating point) is proposed as a solution; read-outs and counts stay as MEASUREMENTS that explain a defect of an E2E candidate (E323 type), not as experiments that deliver a system; zero-parameter soft interfaces whose hard argmax equals the existing rule are allowed. **Named output selections of today, to be named in every report until their E2E twins are measured: the one-piece wiring (snap), the pair probability threshold p >= 0.1, the top-K / candidate selections, the close-node step, Douglas-Peucker in the scorer.** Struck from the roadmap by D88: 'k pieces per end' and operating-point variants (diagnostics only). " + a, 1)
open("HANDOVER.md", "w", encoding="utf-8").write(s)
print("ok")
