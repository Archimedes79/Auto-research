"""Scratch (run once): E363, the end-to-end stack on faithful graphs, into the report (third E2E row, R2 row, day table, table A, chapter 7)."""
import os
R = os.environ["BFLC_ROOT"]
D = os.path.join(R, "results", "Ergebnis_Spurmitte_2026-11-15", "parts")

STACK = ("**E2E stack: learned wiring + soft layer instead of the close-node step (E363): 0.6820 / 0.6854**, -0.0092 / -0.0094 against the rule-wired PRIMARY (inside the band 0.010, not equal), "
         "the duplicate guard broken by the layer on both seeds, the layer's ms unmeasured")


def load(fn):
    return open(os.path.join(D, fn), encoding="utf-8").read()


def save(fn, t):
    open(os.path.join(D, fn), "w", encoding="utf-8").write(t)


def rep(t, old, new):
    assert t.count(old) == 1, (t.count(old), old[:80])
    return t.replace(old, new)


def line_starting(t, head, new):
    L = t.split("\n")
    idx = [i for i, l in enumerate(L) if l.startswith(head)]
    assert len(idx) == 1, (head, idx)
    L[idx[0]] = new
    return "\n".join(L)


# ---------------- 00a_start.md
t = load("00a_start.md")
t = rep(t, "**E360, the layer against the end-to-end control (CPU replay): token INVALID by its written stop rule (iii), nothing adopted; topo F1 +0.0162 / +0.0244 over the end-to-end control without the step, junction F1 +0.087 / +0.100, the duplicate guard broken (+0.0178 / +0.0122); it recovers 52 % / 71 % of the -0.031 / -0.034 that removing the hard step costs and stays -0.015 / -0.010 below the rule version** (with the tolerance 0.02 instead of the 0.01 written before the run it would be a TIE). Follow-up: the stack of the layer and the wiring head; none in the champion |",
        "**E363, the layer on faithful graphs against the end-to-end control: TIE, kept and refined (D87), only because the duplicate guard (0.010) is broken on both seeds** (this supersedes E360, which scored the layer on stale attached ends and was INVALID by its stop rule): topo F1 +0.0142 [+0.0090, +0.0198] / +0.0232 [+0.0183, +0.0283], junction F1 +0.088 / +0.101, vertices per 100 m -6.1 / -7.4, dup_frac +0.0161 / +0.0117; it recovers 46 % / 67 % of the -0.031 / -0.034 that removing the hard step costs and stays -0.017 / -0.011 below the rule version of the same graph. **In the stack (R1 learned head + R2 soft layer, both rule steps replaced): topo F1 0.6820 / 0.6854**, -0.0092 / -0.0094 against the rule-wired PRIMARY. A candidate under refinement (the layer's duplicate defect); nothing adopted, none in the champion; the layer's ms unmeasured |")
t = rep(t, "in two forms of equal weight:", "in three forms of equal weight:")
t = rep(t, "so it is Mirko's case (chapter 7);",
        "so it is Mirko's case (chapter 7); and **the strictest candidate, the E2E stack, replaces R1 and R2 at once: learned wiring + soft Matrix-NMS layer instead of the close-node step, E363: 0.6820 / 0.6854**, -0.009 against the rule-wired PRIMARY (inside the band, not equal), the layer's duplicate guard broken, the layer's ms unmeasured;")
t = rep(t, "| **end-to-end read-out, rule wiring = PRIMARY (the control)** |",
        "| **end-to-end read-out, E2E stack (E363)** | learned attached ends on the graph of the soft Matrix-NMS layer (no hard close-node step), wiring by the learned head W, no snap geometry | **0.6820 / 0.6854**; -0.0092 / -0.0094 against the rule-wired PRIMARY, +0.0155 / +0.0242 over the control without a layer; the layer's duplicate guard broken (dup_frac +0.0161 / +0.0117 over the control, bound 0.010); the layer's ms unmeasured, the head +545 parameters and about +2.0 ms host side | the strictest E2E candidate under refinement (R1 and R2 both replaced); nothing adopted |\n"
        "| **end-to-end read-out, rule wiring = PRIMARY (the control)** |")
t = rep(t, "and **0.691 / 0.695** (end-to-end read-out with the rule wiring, PRIMARY, the control: no extra cost);",
        "and **0.691 / 0.695** (end-to-end read-out with the rule wiring, PRIMARY, the control: no extra cost) and **0.682 / 0.685** (E2E stack, E363: learned wiring + soft layer instead of the close-node step, -0.009 against the rule-wired one, duplicate guard broken, layer ms unmeasured);")
t = rep(t, "and 0.691 / 0.695 with the rule wiring (the control, no extra cost);",
        "and 0.691 / 0.695 with the rule wiring (the control, no extra cost); the E2E stack replacing the close-node step as well: 0.682 / 0.685 (E363);")
t = line_starting(t, "| E360 Matrix-NMS layer instead of the hard close-node step",
        "| E360 Matrix-NMS layer instead of the hard close-node step, against the end-to-end control (CPU replay of stored cards, the attached ends are B's) | topo F1 +0.0162 [+0.0116, +0.0216] / +0.0244 [+0.0196, +0.0293]; junction F1 +0.087 / +0.100; duplicate guard broken (+0.0178 / +0.0122 against 0.010); recovers 52 % / 71 % of the -0.031 / -0.034 that removing the hard step costs | INVALID by the written stop rule (iii), the symmetry of the attached ends (tolerance 0.01; with 0.02 it would be a TIE); superseded by E363 on faithful graphs | superseded: see the next row |\n"
        "| E363 the E2E stack on faithful graphs: the pair run of E324 s re-dumped with the node head of B' (c2 plain, and a with the Matrix-NMS layer); read out with the rule wiring and with the learned head of E358 | layer a against the control c (no layer, no step): topo F1 +0.0142 [+0.0090, +0.0198] / +0.0232 [+0.0183, +0.0283], junction F1 +0.0884 / +0.1007, vertices per 100 m -6.08 / -7.36, F1@1.0 -0.0017 / -0.0027, lanes found -0.0054 / -0.0068, dup_frac +0.0161 / +0.0117 (BROKEN, bound 0.010); the rule version of the same graph 0.6975 / 0.6955, the layer recovers 46 % / 67 % of the -0.0310 / -0.0343 that removing the hard step costs (a - t -0.0168 / -0.0112). **The stack (layer + head): 0.6820 / 0.6854**: head gain on the layer graph +0.0013 [+0.0008, +0.0019] / +0.0010 [+0.0004, +0.0018]; -0.0092 / -0.0094 against E324 s PRIMARY, -0.0111 / -0.0110 against E358 w, +0.0155 / +0.0242 over the control, +0.0204 / +0.0208 over the strict attach-only line | TIE, kept and refined (D87): the duplicate guard is the only broken clause; stop rules (i) - (vii) all held (fixture bit for bit, node check maximum difference 0.0) | not simpler or faster: the layer has no parameters but its ms were never measured, the head +545 parameters and about +2.0 ms host side; a candidate, nothing adopted |")
t = rep(t, "the stack of the Matrix-NMS layer and the wiring head (a CPU card, result expected the same afternoon, otherwise in HANDOVER.md), the ms of the wiring stage (quiet host, running), D102",
        "the E2E stack of the Matrix-NMS layer and the wiring head (E363, TIE; see chapter 7: the stack against W-only as the reported result), D102")
save("00a_start.md", t)

# ---------------- 00_summary.md
t = load("00_summary.md")
t = rep(t, "and **0.691 / 0.695** (rule wiring, PRIMARY, the control: no extra cost) / 0.698 / 0.701 |",
        "and **0.691 / 0.695** (rule wiring, PRIMARY, the control: no extra cost) and **0.682 / 0.685** (E2E stack: learned wiring + soft layer instead of the close-node step, E363: -0.009, duplicate guard broken, layer ms unmeasured) / 0.698 / 0.701 |")
i = t.index("the Matrix-NMS layer against the end-to-end control (E360)")
j = t.index("D110 (the change of yardstick)", i)
t = t[:i] + ("the Matrix-NMS layer on faithful graphs against the end-to-end control (E363, superseding E360) gains +0.0142 / +0.0232 topo F1 and +0.088 / +0.101 junction F1, recovers 46 % / 67 % of what removing the hard close-node step costs, and breaks the duplicate guard on both seeds: TIE, kept and refined. **The stack of the learned head and the layer, which replaces two named hard steps (R1 wiring rule and R2 close-node step), reads out at 0.6820 / 0.6854, -0.009 against the rule-wired end-to-end read-out** (inside the band, not equal; the layer's ms unmeasured). "
                  ) + t[j:]
save("00_summary.md", t)

# ---------------- 03_results.md
t = load("03_results.md")
t = line_starting(t, "| E360 | the Matrix-NMS layer",
        "| E360 | the Matrix-NMS layer (no parameters) instead of the hard close-node step (R2), read out against the END-TO-END control of its own level (CPU replay of stored cards; the attached ends are B's) | a minus c: topo F1 +0.0162 [+0.0116, +0.0216] / +0.0244 [+0.0196, +0.0293], junction F1 +0.0874 / +0.1004, dup_frac +0.0178 / +0.0122 (BROKEN), removing the hard step without a replacement costs -0.0311 / -0.0343, the layer recovers 52 % / 71 % | INVALID by the stop rule (iii) (symmetry of the attached ends; with tolerance 0.02 it would have been TIE); superseded by E363 on faithful graphs | see E363 |\n"
        "| E363 | the E2E stack on faithful graphs: the pair run of E324 s re-dumped with the node head of B' (c2 plain, a with the Matrix-NMS layer), read out at the end-to-end read-out with the rule wiring and with the learned head of E358 (a transfer of the head) | a - c: topo F1 +0.0142 [+0.0090, +0.0198] / +0.0232 [+0.0183, +0.0283], junction F1 +0.0884 / +0.1007, vert100_dp -6.08 / -7.36, F1@1.0 -0.0017 / -0.0027, lanes found -0.0054 / -0.0068, dup_frac +0.0161 / +0.0117 (BROKEN); c 0.6665 / 0.6612, a 0.6807 / 0.6843, the rule version of the same graph t 0.6975 / 0.6955; stack aw 0.6820 / 0.6854 (aw - a +0.0013 / +0.0010, intervals above 0); aw against E324 s PRIMARY -0.0092 / -0.0094, against E358 w -0.0111 / -0.0110, over c +0.0155 / +0.0242, over attach-only +0.0204 / +0.0208 | TIE, kept and refined (D87), only because the duplicate guard is broken; stop rules (i) - (vii) held; predictions hit | the stack replaces R1 and R2 at about -0.009 topo F1 against the rule-step read-out; the layer's ms unmeasured, the head +545 parameters and about +2.0 ms host side; a candidate, nothing adopted |")
t = rep(t, "| E324 s attached ends + learned wiring head W (E358) | 0.6931 / 0.6964 |",
        "| E324 s E2E stack: layer instead of the close-node step + learned wiring head W (E363; faithful graph of B', the layer's duplicate guard broken) | 0.6820 / 0.6854 | 0.7733 / 0.7759 (layer graph a, rule wiring) | - | 0.1805 / 0.1612 (layer graph a, rule wiring) | 0.8737 / 0.8781 (layer graph a, rule wiring) | 23.31 / 23.69 (layer graph a, rule wiring) | - |\n"
        "| E324 s attached ends + learned wiring head W (E358) | 0.6931 / 0.6964 |")
save("03_results.md", t)

# ---------------- 08_open.md
t = load("08_open.md")
t = line_starting(t, "| R2 B's close-node step",
        "| R2 B's close-node step (1.5 m) | E277 soft step: CUT by its guards; E279 (R 1.25 m, pyramid unfrozen): KEEP on B alone, not transferred; E292 Matrix-NMS layer against a rule-step system: TIE. **E363, the layer on faithful graphs against the end-to-end control: TIE, kept and refined (D87), only because the duplicate guard is broken on both seeds; it supersedes E360 (INVALID by a stop rule, on stale attached ends); nothing adopted; the open point is the layer's duplicate defect** | E363: topo F1 +0.0142 / +0.0232 over the end-to-end control without the step, junction F1 +0.088 / +0.101, vertices per 100 m -6.1 / -7.4, dup_frac +0.0161 / +0.0117 (bound 0.010), recovers 46 % / 67 % of the -0.031 / -0.034 that removing the hard step costs, -0.017 / -0.011 below the rule version; ms never measured. E279 against the hard step, graph only: junction F1 -0.048 / -0.031, vertices +2.1 / +0.8 |")
t = line_starting(t, "**Ranked next cards**",
        "**The stack (E363) and the ranked next cards** (the lane lead's list; none is a decision, nothing starts after 15:30 today). **The E2E stack (learned wiring + soft layer instead of the close-node step) reads out at 0.6820 / 0.6854 topo F1: R1 and R2, two named hard steps, are replaced at a price of -0.0092 / -0.0094 against the rule-wired E2E read-out (0.6912 / 0.6948), -0.0111 / -0.0110 against the learned-wiring read-out (0.6931 / 0.6964), and +0.0155 / +0.0242 above the end-to-end control without a duplicate suppression.** Remaining named steps: R3 - R8, T1 - T3, the candidate gate, the cover drop, the layer's threshold e >= 0.5. Next: (1) the layer's duplicate defect (its decay parameters or a one-to-one term, D87); (2) the pair run (or only its attachment) re-trained on B''s nodes; (3) ms of the layer and the stack on an idle GPU, and a batched or numpy forward of the wiring head (its host stage is torch call overhead on about 27 ends); (4) E362, the oracle-node head with 16000 steps (a text draft only, not started); (5) the question whether to hold E317 (open).")
t = rep(t, "8. **WSL memory (a system setting).**",
        "8. **The stack against W-only as the reported E2E result (E363).** The stack replaces R1 and R2 (learned wiring + soft layer, no hard close-node step, no distance rule) and reads out at 0.6820 / 0.6854: -0.009 topo F1 against the rule-wired read-out and -0.011 against the learned-wiring read-out, with the layer's duplicate guard broken on both seeds (+0.0161 / +0.0117 over the control, bound 0.010) and the layer's ms unmeasured. The lane lead's default until he answers: report the stack as the strictest E2E candidate under refinement and keep the learned-wiring read-out (0.6931 / 0.6964) as the headline until the layer's duplicate defect is refined. His case; D110 (the yardstick) is still open.\n9. **WSL memory (a system setting).**")
save("08_open.md", t)

# ---------------- 09_rebuild.md
t = load("09_rebuild.md")
t = rep(t, "| **E324 s, end-to-end read-out, rule wiring**",
        "| E324 s, E2E stack (E363: learned wiring + soft Matrix-NMS layer instead of the close-node step) | 0.682 / 0.685 | - | - | a candidate under refinement: -0.009 against the rule-wired one, duplicate guard broken, layer ms unmeasured |\n"
        "| **E324 s, end-to-end read-out, rule wiring**")
t = rep(t, "| **E362** | E361 with 16000 steps |",
        "| **E363** (D111) | the E2E stack on faithful graphs: the layer of E292 and the head of E358 together, no hard close-node step, no distance rule | TIE, kept and refined (D87): layer against the control +0.0142 / +0.0232, duplicate guard broken; stack 0.6820 / 0.6854 | `E363_e2e-stack-on-faithful-graphs` |\n"
        "| **E362** | E361 with 16000 steps |")
save("09_rebuild.md", t)
print("ok")
