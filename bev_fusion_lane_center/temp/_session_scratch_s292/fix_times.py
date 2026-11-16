import re
def sub(path, pairs):
    s = open(path, encoding="utf-8").read()
    for a, b in pairs:
        assert a in s, (path, a)
        s = s.replace(a, b)
    open(path, "w", encoding="utf-8").write(s)
sub("experiments/E345_pruning-sensitivity-step0/experiment.md", [
    ("Card of 2026-11-14 19:30, revised 20:30 (first critic pass: REVISE) and 20:55 (second pass", "Card of 2026-11-14 19:05, revised 19:20 (first critic pass: REVISE) and 19:36 (second pass"),
    ("First pass 20:30 REVISE, second pass 20:55 REVISE (this revision)", "First pass (19:20) REVISE, second pass (19:36) REVISE (this revision)")])
sub("experiments/E343_test-e324s/experiment.md", [
    ("Card of 2026-11-14 19:50, revised 20:10 (first critic pass: REVISE) and 20:45 (second pass", "Card of 2026-11-14 19:10, revised 19:17 (first critic pass: REVISE) and 19:30 (second pass"),
    ("First critic pass (cards) 20:10: REVISE. Second pass 20:45: REVISE (small)", "First critic pass (cards) 19:17: REVISE. Second pass 19:30: REVISE (small)")])
sub("experiments/E344_e2e-completion-plan/experiment.md", [
    ("(2026-11-14 20:40; he can correct each", "(2026-11-14 19:27; he can correct each"),
    ("(started 2026-11-14 20:40 after", "(started 2026-11-14 19:27 after")])
print("ok")
