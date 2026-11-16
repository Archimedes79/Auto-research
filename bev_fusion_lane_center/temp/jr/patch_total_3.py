"""one-off patch of build_total_report.py (2026-11-13 10:00): D87 (E2E replacements without an alternative are kept and refined, not cut)."""
import os

p = os.path.join(os.environ["BFLC_ROOT"], "temp", "jr", "build_total_report.py")
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


# the experiment table: D87 status next to the written token
rep('VNAME = {', 'D87_KEPT = {"E320": "the only differentiable replacement of the snap geometry; topo F1 ties the snap; the duplicates are to be fixed (E324)", "E322": "the only end-to-end replacement of the output selection; to be built at the equal-accuracy bar (the gate G0 stays as its result)"}\nVNAME = {')
rep("            cf = f\"<br><span class='cap'>conflict: {esc(e['conflict'])}</span>\" if e.get(\"conflict\") else \"\"",
    "            cf = f\"<br><span class='cap'>conflict: {esc(e['conflict'])}</span>\" if e.get(\"conflict\") else \"\"\n            if e['id'] in D87_KEPT:\n                vp += f\"<br><span class='cap'><b>D87: E2E-required, kept and refined</b> - {esc(D87_KEPT[e['id']])}</span>\"")
# the wording of the cut (pre-registered token stays)
rep("Read E273 with care:", "<b>D87 (Mirko, 2026-11-13):</b> an experiment that is the end-to-end replacement of a rule step with no other end-to-end alternative is not cut by a missed bound or guard: the written token stays in the card and in this table, the experiment is kept and refined (marked D87). Read E273 with care:")
rep("So the attachment of E320 (geometry) could recover at most about a third of the gap, and it did (table below).", "So the attachment of E320 (geometry) could recover at most about a third of the gap, and it did (table below; by D87 E320 is kept and refined, not cut: the duplicates are the defect to remove).")
rep("a learned head stays designed with an equal-accuracy bar and is the one question left for Mirko (end-to-end replacement of the output selection against a simple system).</div>", "the question the card carried for Mirko (end-to-end replacement of the output selection against a simple system) is answered by D87: E2E has priority where no E2E alternative exists, so the learned wiring head is built at the equal-accuracy bar (topo F1 >= c - 0.005 on both seeds) and its price in parameters and ms is reported.</div>")
rep('"MEASURED: CUT by the duplicates guard (dup_frac +0.0184 on seed 0, +0.0093 on seed 1; topo F1 ties the snap); E323: not the curve count"', '"MEASURED: written token CUT by the duplicates guard (dup_frac +0.0184 on seed 0, +0.0093 on seed 1; topo F1 ties the snap); E323: not the curve count; D87: kept and refined, follow-up E324 (the duplicates)"')
rep('"gated card: G0 (oracle selection) shows no headroom; closed at the gate; the head stays designed (equal-accuracy bar), one question for Mirko: E2E replacement of the output selection against simplicity"', '"G0 (oracle selection) shows no accuracy headroom; D87: not closed, the learned head is built at the equal-accuracy bar (topo F1 >= c - 0.005), price in params / ms reported"')
rep("E320 (learned attachment) is measured (CUT by a guard, a tie with the snap), E322 (soft wiring) closed at its gate G0, E317 / E318 / E319 are held (D85).</li>", "E320 (learned attachment) is measured (written token CUT by a guard, a tie with the snap; D87: kept and refined), E322 (soft wiring): no accuracy headroom, by D87 built at the equal-accuracy bar, E317 / E318 / E319 are held (D85).</li>")
open(p, "w", encoding="utf-8").write(s)
print("patched")
