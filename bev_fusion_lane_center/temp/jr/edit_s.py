"""Scratch (run once): afternoon edits of 00_summary.md."""
import os
P = os.path.join(os.environ["BFLC_ROOT"], "results", "Ergebnis_Spurmitte_2026-11-15", "parts", "00_summary.md")
t = open(P, encoding="utf-8").read()
def rep(old, new):
    global t
    assert t.count(old) == 1, (t.count(old), old[:70])
    t = t.replace(old, new)

rep("| **E324 s, system of record** (+ learned attachment, geometry loss, 6000 steps) | **0.691 / 0.695** (PRIMARY) / 0.698 / 0.701 |",
    "| **E324 s, system of record** (+ learned attachment, geometry loss, 6000 steps) | **0.693 / 0.696** (attached ends + learned wiring head, E358; the rule-wired control PRIMARY 0.691 / 0.695) / 0.698 / 0.701 |")
rep("The named hard steps on the output remain (R1 snap wiring,",
    "The named hard steps on the output remain (R1 snap wiring, now a learned head in the reported read-out but still an argmax per end,")
rep("**Token design (D104):** step 0 on the ground truth (E355): half of the nodes are free, a minimum node distance by merging costs 0.03 junction F1. The first trained stage (E357, ONE pair head on perfect nodes) failed by its written rule on both seeds (topo F1 0.43 / 0.39, section-free read-out at p >= 0.5, against 0.70 for a nearest-node rule): the design stops; the system we have stays.",
    "**Token design (D104):** step 0 on the ground truth (E355): half of the nodes are free, a minimum node distance by merging costs 0.03 junction F1. The first trained stage (E357, ONE pair head on perfect nodes) failed by its written rule on both seeds (topo F1 0.43 / 0.39, read-out at p >= 0.5, against 0.70 for a nearest-node rule). The node-type head with a per-source softmax on the same oracle nodes (E359, 3000 steps: topo F1 0.772 / 0.763; E361, 10000 steps: 0.832 / 0.830 against the half-gap line 0.834, junction-link F1 0.687 / 0.688 against 0.75) fails the written bars, E361 by a hair; E362 (16000 steps) is a text draft only. The design stays outside the system (section 3 F, chapter 7).")
rep("- **Next compute card:**",
    "- **End-to-end work of the afternoon (D111; the computer shuts down at 17:30 today by Mirko's order):** the learned wiring head on the attached ends (E358) is KEEP+: end-to-end read-out 0.6931 / 0.6964 against the rule-wired 0.6912 / 0.6948 (+0.0019 / +0.0016, intervals above 0, every guard held; +545 parameters, ms not measured); the Matrix-NMS layer against the end-to-end control (E360) gains +0.0162 / +0.0244 topo F1 and +0.087 / +0.100 junction F1 but is INVALID by its written stop rule (it would be a TIE with a looser tolerance) and breaks the duplicate guard, recovering 52 % / 71 % of what removing the hard close-node step costs. D110 (the change of yardstick) is still an open question; nothing is applied.\n- **Next compute card:**")
open(P, "w", encoding="utf-8").write(t)
print("ok")
