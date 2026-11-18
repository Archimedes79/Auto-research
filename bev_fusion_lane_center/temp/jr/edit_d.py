"""Scratch (run once): afternoon edits of 00a_start.md (E358, E360, E359 / E361, D111)."""
import os
P = os.path.join(os.environ["BFLC_ROOT"], "results", "Ergebnis_Spurmitte_2026-11-15", "parts", "00a_start.md")
t = open(P, encoding="utf-8").read()
def rep(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:70])
    t = t.replace(old, new)

rep("**champion E324 s 0.691 / 0.695 (snapped, with a rule step: 0.698 / 0.701)**",
    "**champion E324 s 0.693 / 0.696 (attached ends + the learned wiring head, E358; rule wiring 0.691 / 0.695; snapped, with a rule step: 0.698 / 0.701)**")
rep("| **E324 s (champion)** | **0.691 / 0.695** (PRIMARY: attached ends, nearest-piece wiring, no snap geometry); snapped, with a rule step: 0.698 / 0.701 | ends learned soft (attachment); hard: nearest-piece wiring, thresholds",
    "| **E324 s (champion)** | **0.693 / 0.696** (end-to-end read-out: attached ends + the learned wiring head W of E358, no snap geometry); the same with the rule wiring (control, PRIMARY) 0.691 / 0.695; snapped, with a rule step: 0.698 / 0.701 | ends learned soft (attachment); wiring learned but an argmax per end (+545 parameters, ms not measured); hard: thresholds")
rep("End-to-end read-out +0.047 / +0.047 over E312's unsnapped one;", "End-to-end read-out (rule wiring) +0.047 / +0.047 over E312's unsnapped one;")
rep("| E355, E357 | one token set",
    "| E358 | the learned wiring head W (545 parameters) on E324 s's attached ends instead of the nearest-piece rule | end-to-end read-out 0.6931 / 0.6964 against the rule-wired 0.6912 / 0.6948 (+0.0019 / +0.0016, intervals above 0) | KEEP+: the reported read-out |\n"
    "| E360 | the Matrix-NMS layer instead of the hard close-node step, against the end-to-end control (CPU replay of stored cards) | topo F1 +0.0162 / +0.0244 over the control without the step, but the duplicate guard broken; -0.015 / -0.010 below the rule version | INVALID by its stop rule (would be TIE with a looser written tolerance); nothing adopted |\n"
    "| E355, E357 | one token set")
rep("| E357 FAIL, the design stops; the system we have stays |",
    "| E357 FAIL (T0 of the token design, on oracle nodes) |\n"
    "| E359, E361 | the same oracle-node test with a node-type output and a per-source softmax over the successors; E361 = 10000 steps | topo F1 0.772 / 0.763 (E359), 0.832 / 0.830 (E361) against 0.70 for the best simple rule and the half-gap line 0.834; junction-link F1 0.531 / 0.502 and 0.687 / 0.688 against the 0.75 bar | both FAIL by the written bars, E361 by a hair; E362 (16000 steps) is a text draft only, not started; the token design stays outside the system (section 3 F) |")
open(P, "w", encoding="utf-8").write(t)
print("ok")
