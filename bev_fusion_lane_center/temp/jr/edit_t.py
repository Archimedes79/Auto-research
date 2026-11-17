"""Scratch (run once): E358 wiring-stage ms (fifth timing protocol), equal prominence of the two E2E read-outs, open decision W."""
import os
R = os.environ["BFLC_ROOT"]
D = os.path.join(R, "results", "Ergebnis_Spurmitte_2026-11-15", "parts")

COST = ("about +2 ms per frame on the host side (wiring stage 4.27 / 4.29 ms with the rule, 6.29 / 6.36 ms with the head: +2.02 [+1.93, +2.11] / +2.06 [+1.98, +2.15] ms, "
        "seed 0 / 1; torch with one thread, per VAL frame, quiet host, paired: a fifth timing protocol, never compared with the GPU numbers)")


def edit(fn, pairs):
    p = os.path.join(D, fn)
    t = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert t.count(old) == 1, (fn, t.count(old), old[:80])
        t = t.replace(old, new)
    open(p, "w", encoding="utf-8").write(t)


edit("00a_start.md", [
    ("an end-to-end result, not an accuracy result; ms not measured, +545 parameters, at most 0.12 MMACs.",
     "an end-to-end result, not an accuracy result; the cost: +545 parameters, at most 0.12 MMACs, " + COST + ". By D106 (simpler AND faster) the learned wiring is a variant with more parts and a slower host stage: Mirko's case, see chapter 7."),
    ("wiring learned but an argmax per end (+545 parameters, ms not measured); hard: thresholds",
     "the wiring is a rule (control) or a learned argmax per end (+545 parameters, about +2 ms per frame on the host side, a separate protocol); hard: thresholds"),
    ("not simpler (+545 parameters, at most 0.12 MMACs), ms of the host-side wiring stage not measured yet; the gain is end-to-end-ness, not accuracy",
     "not simpler (+545 parameters, at most 0.12 MMACs) and not faster: " + COST + "; the junction forward is 20.6 ms in the 12-frame night protocol, not like-for-like, so no percentage is given; the gain is end-to-end-ness, not accuracy"),
    ("the lead has decided to report it that way, +545 parameters, ms not measured)",
     "the lead's default until he answers is to report the learned-wiring read-out as the E2E result and to name the cost next to it, +545 parameters and about +2 ms per frame on the host side; he can revert it with one word)"),
    # equal prominence in the read-out table
    ("| **end-to-end read-out with the learned wiring head (E358)** | learned attached ends, wiring by the learned head W (545 parameters, argmax per end), no snap geometry | **0.6931 / 0.6964** | **the result reported as the system** (D109; the lead's decision after E358 KEEP+) |\n"
     "| end-to-end read-out, rule wiring = PRIMARY (the control) | learned attached ends, nearest-piece snap wiring (a named hard step), no snap geometry | 0.6912 / 0.6948 | the control of E358 and of every new card (D109) |",
     "| **end-to-end read-out, learned wiring (E358)** | learned attached ends, wiring by the learned head W (argmax per end), no snap geometry | **0.6931 / 0.6964**; +545 parameters, about +2.0 ms per frame on the host side | reported as the E2E result by the lane lead's default (one hard step less; D88 / D109); by D106 a variant with more parts and a slower host stage: Mirko's case |\n"
     "| **end-to-end read-out, rule wiring = PRIMARY (the control)** | learned attached ends, nearest-piece snap wiring (a named hard step), no snap geometry | **0.6912 / 0.6948**; no extra cost | the control of E358 and of every new card (D109) |"),
    ("Since D109 the result reported as the system is the **end-to-end read-out of E324 s, now with the learned wiring head (E358): topo F1 0.6931 / 0.6964** (VAL, seed 0 / seed 1); the same read-out with the rule-wired nearest-piece step (PRIMARY, the control) is 0.6912 / 0.6948;",
     "Since D109 the result reported as the system is the **end-to-end read-out of E324 s**, in two forms of equal weight: **with the learned wiring head (E358): topo F1 0.6931 / 0.6964** (VAL, seed 0 / seed 1; +545 parameters, about +2.0 ms per frame on the host side) and **with the rule-wired nearest-piece step (PRIMARY, the control): 0.6912 / 0.6948** (no extra cost). The lane lead's default until Mirko answers: report the learned wiring (it replaces a rule step by a learned one, D88 / D109) and name its cost next to it; by D106 it is a more-parts-and-slower variant, so it is Mirko's case (chapter 7);"),
    ("**0.693 / 0.696** (end-to-end read-out: attached ends + the learned wiring head W of E358, no snap geometry); the same with the rule wiring (control, PRIMARY) 0.691 / 0.695;",
     "**0.693 / 0.696** (end-to-end read-out with the learned wiring head W of E358: +545 parameters, about +2 ms per frame on the host side) and **0.691 / 0.695** (end-to-end read-out with the rule wiring, PRIMARY, the control: no extra cost);"),
    ("**champion E324 s 0.693 / 0.696 (attached ends + the learned wiring head, E358; rule wiring 0.691 / 0.695; snapped, with a rule step: 0.698 / 0.701)**",
     "**champion E324 s 0.693 / 0.696 with the learned wiring head (E358; +545 parameters, about +2 ms per frame on the host side) and 0.691 / 0.695 with the rule wiring (the control, no extra cost); snapped, with a rule step: 0.698 / 0.701**"),
])
edit("00_summary.md", [
    ("+545 parameters, ms not measured); the Matrix-NMS layer",
     "+545 parameters and about +2 ms per frame on the host side, a separate timing protocol: by D106 a variant with more parts and a slower host stage, Mirko's case); the Matrix-NMS layer"),
    ("| **0.693 / 0.696** (attached ends + learned wiring head, E358; the rule-wired control PRIMARY 0.691 / 0.695) / 0.698 / 0.701 |",
     "| **0.693 / 0.696** (learned wiring head, E358: +545 parameters, about +2 ms per frame host side) and **0.691 / 0.695** (rule wiring, PRIMARY, the control: no extra cost) / 0.698 / 0.701 |"),
])
edit("03_results.md", [
    ("+545 parameters, at most 0.12 MMACs, ms not measured; live check",
     "+545 parameters, at most 0.12 MMACs, " + COST + "; live check"),
    ("with the learned wiring head W: +545 parameters, at most +0.12 MMACs, ms of the host-side wiring stage not measured)",
     "with the learned wiring head W: +545 parameters, at most +0.12 MMACs, about +2 ms per frame on the host side in a fifth protocol, not added to the GPU column)"),
    ("**Four timing protocols, never mixed in one comparison**", "**Five timing protocols, never mixed in one comparison** (the fifth, E358's host-side wiring stage: rule 4.27 / 4.29 ms, learned head 6.29 / 6.36 ms per VAL frame, +2.02 [+1.93, +2.11] / +2.06 [+1.98, +2.15] ms, torch with one thread, quiet host, paired; it is not part of the GPU forward below)"),
])
edit("08_open.md", [
    ("+545 parameters, at most 0.12 MMACs; ms of the host-side wiring stage not measured yet (running on a quiet host); earlier",
     "+545 parameters, at most 0.12 MMACs, " + COST + "; earlier"),
    ("(2) the ms of the wiring stage on a quiet host (running);", "(2) optimise the wiring head's host stage (the forward is torch call overhead on about 27 ends, not arithmetic; a batched or numpy forward is an unmeasured optimisation);"),
    ("the lane lead has decided to report it that way (+545 parameters, ms not measured, the D106 line \"simpler AND faster\" is NO on parameters).",
     "see the open decision W below."),
    ("7. **WSL memory (a system setting).**",
     "7. **W: learned wiring against rule wiring as the reported E2E result (E358).** Learned wiring: +0.0019 / +0.0016 topo F1 and one hard step less (a learned argmax instead of a distance rule), against +545 parameters and about +2.0 ms per frame on the host side (4.27 / 4.29 ms to 6.29 / 6.36 ms; a fifth timing protocol, not comparable with the GPU numbers; the junction forward is 20.6 ms in the 12-frame night protocol, not like-for-like, so no percentage is given). By D106 (simpler AND faster) the learned wiring is a more-parts-and-slower variant: his case. The lane lead's default until he answers: report the learned-wiring read-out (0.6931 / 0.6964) as the E2E result and name the cost next to it; the rule-wired read-out (0.6912 / 0.6948, no extra cost) stays as the control; he can revert it with one word.\n8. **WSL memory (a system setting).**"),
])
print("ok")
