"""Scratch (run once): E362 (node-type head, 16000 steps) done: token INCONCLUSIVE; machine problems of the day."""
import os
R = os.environ["BFLC_ROOT"]
D = os.path.join(R, "results", "Ergebnis_Spurmitte_2026-11-15", "parts")


def load(fn):
    return open(os.path.join(D, fn), encoding="utf-8").read()


def save(fn, t):
    open(os.path.join(D, fn), "w", encoding="utf-8").write(t)


def rep(t, old, new):
    assert t.count(old) == 1, (t.count(old), old[:80])
    return t.replace(old, new)


INFRA = ("seed 0 had 3 and seed 1 had 9 CUDA \"unknown error\" interruptions (resumed from checkpoints by a supervisor chain; the runs are not bit-reproducible), "
         "and one GPU stall from 18:18 to 18:38 with the Windows driver event `nvlddmkm` 13 (about 100 per minute) that ended when the experiments chat killed its own two trainings; "
         "no rollback and no non-finite loss; with two trainings on the card (12 GB nearly full) this is a recurring problem of the day, not a result issue")

# ---- 00a_start
t = load("00a_start.md")
t = rep(t, "E362 (16000 steps) is a text draft only, not started; the token design stays outside the system (section 3 F) |",
        "E362 (16000 steps) INCONCLUSIVE (topo F1 0.8427 / 0.8367, over the half-gap line but not clear, junction-link F1 0.711 / 0.713); the token design stays outside the system (section 3 F) |")
t = rep(t, "| E362 the same head with 16000 steps | a text draft, not started | - | - |",
        "| E362 the same head with 16000 steps (Mirko's D104 question) | gap share g1 0.533 / 0.511, topo F1 0.8427 / 0.8367 (half-gap line 0.834: over it on both seeds, but not clear by the card's own criterion g1 >= 0.54); junction-link F1 (G2) 0.711 / 0.713 (bar 0.75, fail line 0.50); against E361 topo F1 +0.011 / +0.007 (3000 to 10000 steps gave +0.06, 10000 to 16000 +0.01: the schedule is nearly saturated); the type head is the weak output again (long-out F1 0.853 / 0.845 against 0.90, recall 0.82 / 0.81, AUC 0.98), the softmax works (top-1 0.905 / 0.909) | INCONCLUSIVE by the written order of E359 (FAIL does not apply, PASS needs G2 0.75) | not a candidate (oracle nodes only); T1 (predicted nodes) is Mirko's decision (D105 / D111) |\n\n**Infrastructure of the day:** " + INFRA + ".")
t = rep(t, "(a) continue the oracle-node stage longer (E362, 16000 steps; at best this moves the strict-bar verdict to inconclusive, the half-gap line is within noise now; a text draft, not started);",
        "(a) continue the oracle-node stage longer (E362, 16000 steps, is done: INCONCLUSIVE, +0.011 / +0.007 topo F1 over E361, the schedule is nearly saturated, the type head is the weak output);")
t = rep(t, "Default: (c) unless he wants (a) or (b).", "Default: stop the oracle-node stage (it no longer fails but does not pass cleanly, and longer training is saturated); T1 only on his word.") if "Default: (c) unless he wants (a) or (b)." in t else t
save("00a_start.md", t)

# ---- 00_summary
t = load("00_summary.md")
t = rep(t, "E361 by a hair; E362 (16000 steps) is a text draft only.",
        "E361 by a hair. E362 (16000 steps, Mirko's D104 question) is INCONCLUSIVE: topo F1 0.8427 / 0.8367 (half-gap line 0.834, over it but not clear), junction-link F1 0.711 / 0.713 against 0.75; longer training is nearly saturated and the type head is the weak output, so predicted nodes (T1) are his decision.")
save("00_summary.md", t)

# ---- 03_results
t = load("03_results.md")
t = rep(t, "E359: FAIL on both seeds; E361: FAIL by the letter, by a hair (section F) | not a candidate: oracle nodes only, TEST untouched; E362 (16000 steps) is a text draft, not started |",
        "E359: FAIL on both seeds; E361: FAIL by the letter, by a hair; E362 (16000 steps): INCONCLUSIVE, g1 0.533 / 0.511, topo F1 0.8427 / 0.8367, G2 0.711 / 0.713 (section F) | not a candidate: oracle nodes only, TEST untouched |")
t = rep(t, "## F. Token design: three oracle-node tests, and what is left (E355 / E357 / E359 / E361)", "## F. Token design: four oracle-node tests, and what is left (E355 / E357 / E359 / E361 / E362)")
t = rep(t, "Three steps, each read out on the same frames", "Four steps, each read out on the same frames")
t = rep(t, "| E361 | E359, 10000 steps | 10000 | 0.832 / 0.830 | 0.491 / 0.484 | 0.687 / 0.688 | FAIL by the letter, by a hair |\n",
        "| E361 | E359, 10000 steps | 10000 | 0.832 / 0.830 | 0.491 / 0.484 | 0.687 / 0.688 | FAIL by the letter, by a hair |\n"
        "| E362 | E359, 16000 steps | 16000 | 0.843 / 0.837 | 0.533 / 0.511 | 0.711 / 0.713 | INCONCLUSIVE |\n")
t = rep(t, "**Slips, in one line.**",
        "**E362: 16000 steps (written order of E359: INCONCLUSIVE).** Gap share g1 0.533 / 0.511 (topo F1 0.8427 / 0.8367): over the half-gap line 0.834 on both seeds, but not clear by the card's own criterion (g1 >= 0.54 on both seeds; the band 0.50 - 0.54 is read as inside the noise, and this is the third look at the same bar on VAL, with the schedule chosen after E361's near miss). The strict link-level G2 is 0.711 / 0.713 (bar 0.75, fail line 0.50): FAIL does not apply, PASS does not apply. Against E361: topo F1 +0.011 / +0.007, g1 +0.042 / +0.027; 3000 to 10000 steps had given +0.060 / +0.067, so the schedule is close to saturated; the training loss fell to 0.077 / 0.080 (E361: 0.102 / 0.110), a hint that the head starts to fit the training scenes (not tested). The limiting output is again the TYPE head: long-out F1 0.853 / 0.845 against the bar 0.90 (recall 0.82 / 0.81, precision 0.89), AUC 0.983 / 0.981 (met); the softmax works (oracle-source top-1 0.905 / 0.909 against 0.85). Trend over the four steps: g1 -1.04 / -1.16, 0.27 / 0.23, 0.49 / 0.48, 0.53 / 0.51; type recall 0.66 / 0.64, 0.79 / 0.79, 0.82 / 0.81. Reported next to it, labelled \"chosen after E361, informed by it, not independent\": the multi-successor decode g1 0.547 / 0.521, the source-level F1 0.797 / 0.800, the argmax cap 0.811 reached to 0.76, the degraded read-out (17 % of the nodes removed) 0.795 / 0.786 against 0.648 for the best rule. Predictions hit except the training loss (below the window). For Mirko, one sentence: the oracle-node stage no longer fails but does not pass cleanly; longer training is saturated, the type head is the weak output; T1 (predicted nodes) is his decision (D105 / D111).\n\n"
        "**Slips, in one line.**")
t = rep(t, "no roll-back or non-finite step in either log. The E357 prediction",
        "no roll-back or non-finite step in either log. E362's two runs (17:26 to 20:37) were interrupted by the GPU: " + INFRA + ". The E357 prediction")
t = rep(t, "What is open for Mirko (continue the oracle-node stage longer as E362, start T1, or stop the token design) is in chapter 7. Sources: the cards E357 (section 3), E359 and E361 (section 3),",
        "What is open for Mirko (start T1 on predicted nodes, or stop the token design; more oracle-node training is saturated) is in chapter 7. Sources: the cards E357 (section 3), E359, E361 and E362 (section 3),")
save("03_results.md", t)

# ---- 08_open
t = load("08_open.md")
t = rep(t, "(4) E362, the oracle-node head with 16000 steps (a text draft only, not started); (5)", "(4) E362, the oracle-node head with 16000 steps, is done (INCONCLUSIVE, section F of chapter 3); (5)")
t = rep(t, "Options: (a) continue the oracle-node stage longer (E362, 16000 steps; at best this moves the strict-bar verdict to inconclusive, the half-gap line is within noise now);",
        "E362 (16000 steps, his D104 question) is done: INCONCLUSIVE (g1 0.533 / 0.511, topo F1 0.8427 / 0.8367, over the half-gap line but not clear; junction-link F1 0.711 / 0.713 against 0.75; +0.011 / +0.007 over E361: longer training is saturated, the type head is the weak output). Options: (a) continue the oracle-node stage longer (saturated: not recommended);")
t = rep(t, "Default: (c) unless he wants (a) or (b).", "Default (the lane lead's): stop the oracle-node stage, it is saturated; T1 (predicted nodes) only on his word.")
save("08_open.md", t)

# ---- 09_rebuild
t = load("09_rebuild.md")
t = rep(t, "| text draft only, not started | `E362_node-type-head-16000-steps` |",
        "| INCONCLUSIVE: g1 0.533 / 0.511, topo F1 0.8427 / 0.8367, junction-link F1 0.711 / 0.713; the head's two runs had 3 and 9 GPU \"unknown error\" interruptions and one GPU stall (resumed from checkpoints, not bit-reproducible) | `E362_node-type-head-16000-steps` |")
save("09_rebuild.md", t)
print("ok")
