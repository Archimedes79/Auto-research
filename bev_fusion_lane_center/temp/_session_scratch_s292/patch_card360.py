import io
p = "experiments/E360_matrix-nms-in-the-e2e-stack/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)
rep("lie within 0.02 of c's (stored references: B +0.009 / +0.008, c2 +0.010 / +0.010)", "lie within 0.01 of c's (stored references: B +0.009 / +0.008, c2 +0.010 / +0.010; tolerance tightened from 0.02 to 0.01 at 09:25 on the second critic's remark that 0.02 is twice the signal; the `kill` text is not part of the rule block)")
rep("(i) `numbers.base` of c and of a equal the stored graph-level cards", "(i) the WHOLE `numbers.base` block of c and of a equals the stored graph-level cards (the witnesses ck / ak must carry exactly the base block and the curves of c / a)")
rep("(v) arm t does not reproduce E338a", "(v) arm t is missing or does not reproduce E338a (numbers, curves, wired_both_ends, long_links_dropped, ends_within_1m, the counts file)")
rep("Low confidence: c2 plain has 29.2", "Note: at the PRIMARY read-out the hard step is worth more than at the E312 connectors (E338a 0.6989 against the base 0.6346 of c2 with the step: +0.064, not +0.054), so the c2-plain bracket may lie higher than assumed; the dup_frac, junction F1 and lanes-found predictions are rough extrapolations of the graph-level differences. Low confidence: c2 plain has 29.2")
rep("**Status: card v2, second critic pass not yet done (section 4); nothing run. CPU only: a replay of stored dumps (no training, no GPU).**", "**Status: card v2.1 (09:25): second critic pass done, G2 YES (section 4). CPU only: a replay of stored dumps (no training, no GPU). The chain of nine scoring jobs was started at 09:19 by an accident (a 'dry run' flag does not pass from Git Bash through `wsl`), before the second critic pass answered; no number had been looked at, the commands are the ones of section 0, only text of the kill clause (stop-rule tolerance) and the verdict script were tightened afterwards (the rule block and the arms are unchanged).**")
rep("A second pass on the card text is required before the first scoring (the G2 gate is not given before it).", "**Second pass 2026-11-16 09:22 (the same critic, verification of the answers): G2 YES.** Findings 1 - 7 answered; remarks taken: the symmetry tolerance 0.02 -> 0.01, the stop rules on the whole base block, the witnesses on c's / a's base blocks and the curves, a missing arm t is a failed stop rule, t compares the counts file (all in `temp/e360/verdict_e360.py`, selftest 14 planted cases); noted: the prediction windows equal the bracket half-width (an honest, weakly informative hit rule), the connector gains come from E312's connectors (PRIMARY gains are larger), `decide.py` must not be used (its output would say KEEP on guards alone).")
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
