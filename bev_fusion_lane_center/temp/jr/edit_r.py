"""Scratch (run once): afternoon edits of 03_results.md (E358, E360, E359 / E361, token design section F)."""
import os
P = os.path.join(os.environ["BFLC_ROOT"], "results", "Ergebnis_Spurmitte_2026-11-15", "parts", "03_results.md")
t = open(P, encoding="utf-8").read()


def rep(old, new):
    global t
    assert t.count(old) == 1, (t.count(old), old[:70])
    t = t.replace(old, new)


# five lines
rep("E324 s 0.691 / 0.695 (snapped 0.698 / 0.701; attach-only 0.662 / 0.665).",
    "E324 s 0.693 / 0.696 with the learned wiring head (E358; the rule-wired control 0.691 / 0.695; snapped 0.698 / 0.701; attach-only 0.662 / 0.665).")
# table A: new rows after the E355 + E357 row
i = t.index("| E355 + E357 |")
j = t.index("\n", i)
rows = (
    "\n| E358 | the learned wiring head W (545 parameters) on the ATTACHED ends of E324 s instead of the nearest-piece rule (R1), trained by imitation of a rule-derived label on frozen records | end-to-end read-out topo F1 0.6931 / 0.6964 against the rule-wired control 0.6912 / 0.6948: +0.0019 / +0.0016 [+0.0012, +0.0027] / [+0.0009, +0.0023]; F1@1.0, lanes found, vertices unchanged, dup_frac +0.0008 / +0.0013, junction F1 +0.0009 / +0.0013; head seeds 1 and 2 within 0.0003; agreement with the oracle label on the real choices 0.787 / 0.782 against the rule 0.764 / 0.748 | KEEP+ (by the card's rule, by hand: D87 replay; the paired interval above 0 on both seeds, every guard held); prediction +0.001 (window 0.003) HIT | R1 on the reported read-out becomes a learned argmax; learned by imitation, no output gradient reaches the head, the candidate gate and the cover drop stay hard (named); +545 parameters, at most 0.12 MMACs, ms not measured; live check of the TRAIN pass: four attempts, three REFUSED (CPU; GPU; GPU at batch size 8: the two stored dumps were made with different loader settings), the fourth, a faithful two-loader replay allowed once by the lead, ACCEPTED |"
    "\n| E360 | the Matrix-NMS layer (no parameters) instead of the hard close-node step (R2), read out against the END-TO-END control of its own level (CPU replay of stored cards; the attached ends are B's) | a minus c, PRIMARY: topo F1 +0.0162 [+0.0116, +0.0216] / +0.0244 [+0.0196, +0.0293], junction F1 +0.0874 / +0.1004, dup_frac +0.0178 / +0.0122 (bound +0.010: BROKEN on both), F1@1.0 -0.0001 / -0.0016, lanes found -0.0020 / -0.0040, vertices per 100 m (DP basis) -6.11 / -7.40; the own-keypoint witness gives the same gain (+0.0162 / +0.0239); removing the hard step without a replacement costs -0.0311 / -0.0343, the layer recovers 52 % / 71 %, and stays -0.0149 / -0.0099 below the rule version of c2 | INVALID by the stop rule (iii) (symmetry of the attached ends; tolerance 0.01 written before the run, with 0.02 the token would have been TIE); nothing cut, nothing adopted | R2 leaves the delivered path only in a stack with a faithful node head (open card); price against the rule version -0.015 / -0.010; ms not measured |"
    "\n| E359 + E361 | T0 of the token design, again on oracle nodes: a node-type output (long-out, long-in, short-out) plus a per-source softmax over the candidate successors; E361 = E359 with 10000 instead of 3000 steps | E359: topo F1 0.7721 / 0.7626, gap share g1 0.267 / 0.232, junction-link F1 (G2) 0.531 / 0.502; E361: topo F1 0.8316 / 0.8297, g1 0.491 / 0.484, G2 0.687 / 0.688 (bars: g1 0.5 / half-gap line 0.834, G2 0.75; E357: 0.4255 / 0.3928, g1 -1.04 / -1.16, G2 0.108 / 0.093) | E359: FAIL on both seeds; E361: FAIL by the letter, by a hair (section F) | not a candidate: oracle nodes only, TEST untouched; E362 (16000 steps) is a text draft, not started |"
)
t = t[:j] + rows + t[j:]
# B1: head row above PRIMARY row
rep("| **E324 s PRIMARY (attached ends, end-to-end read-out)** |",
    "| **E324 s, attached ends + learned wiring head W (E358; the reported end-to-end read-out)** | **0.6931 / 0.6964** | 0.7653 / 0.7648 | 0.3047 / 0.3063 | 0.1905 / 0.1605 | 0.8777 / 0.8786 | 19.2 / 19.8 | - |\n"
    "| E324 s PRIMARY (attached ends, rule wiring: the control of E358) |")
rep("| **E324 s PRIMARY (attached ends + nearest-piece wiring)** | **0.6911 / 0.6948** |",
    "| E324 s attached ends + learned wiring head W (E358) | 0.6931 / 0.6964 | 0.7653 / 0.7648 | 0.3047 / 0.3063 | 0.1905 / 0.1605 | 0.8777 / 0.8786 | 19.2 / 19.8 | - |\n"
    "| E324 s PRIMARY (attached ends + nearest-piece wiring), the control | 0.6911 / 0.6948 |")
rep("The headline row is **E324 s PRIMARY, the end-to-end read-out**",
    "The headline row is **E324 s attached ends + learned wiring head, the reported end-to-end read-out** (E358; its numbers are those of the E358 card, the control row PRIMARY is the VAL card of E324 s: the two sources differ in the 4th decimal, 0.6912 against 0.6911 on seed 0)")
# table C
rep("| E324 s (system of record) | 2.741 | 8.202 (incl. node head 0.243 and Attach) |",
    "| E324 s (system of record; with the learned wiring head W: +545 parameters, at most +0.12 MMACs, ms of the host-side wiring stage not measured) | 2.741 | 8.202 (incl. node head 0.243 and Attach) |")
rep("**0.6911 / 0.6948** (PRIMARY) / 0.6980 / 0.7010 |", "**0.6931 / 0.6964** (with the head W; rule wiring PRIMARY 0.6911 / 0.6948) / 0.6980 / 0.7010 |")
# section F
F = open(os.path.join(os.environ["BFLC_ROOT"], "temp", "jr", "section_f.md"), encoding="utf-8").read()
rep("## Sources\n", F + "\n## Sources\n")
rep("E351 - E357 (E355 step 0 table", "E351 - E362 (E355 step 0 table")
open(P, "w", encoding="utf-8").write(t)
print("ok")
