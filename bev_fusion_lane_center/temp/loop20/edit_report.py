p = "reports/build/reports/build_week3_report.py"
t = open(p, encoding="utf-8").read()


def rep(a, b, cnt=1):
    global t
    assert t.count(a) == cnt, (t.count(a), a[:70])
    t = t.replace(a, b)


rep('the band of 0.010 is about 3 SDs. Lanes found SD', 'the band of 0.010 is about 4 SDs. Lanes found SD')
rep('("0 of 3", "ideas that beat the plain fine-tune this week", "exact distance map (E237), clamp 8 m (E244 arm b), per-range thresholds (E246 killed at step 0); the Poisson loss (E255) see section 5"),',
    '("0 of 4", "ideas that beat the plain fine-tune this week", "exact distance map (E237), clamp 8 m (E244 arm b), per-range thresholds (E246 killed at step 0), Poisson set loss (E255: Q -0.073 at an equal vertex budget)"),')
rep('Seed SD of Q about 0.003 (SD of the paired difference 0.005) and of lanes found about 0.005 were measured on paired fine-tunes (E240). The 0.010 band is 3 SDs for Q; "\n    "a single pair cannot see anything below about 0.01.",',
    'Seed SD of Q about 0.0025 (SD of the paired difference 0.0043) and of lanes found about 0.004 were measured on four paired fine-tunes (E240). The 0.010 band is 4 SDs for Q; "\n    "a single pair cannot see anything below about 0.01.",')
rep('Lanes found +0.06 and R@1 far +0.05 in every pair, Q +0.006 (TIE).', 'Lanes found +0.06 and R@1 far +0.05 in every one of four pairs, Q +0.006 (TIE).')
rep('"<b>Four loss and threshold ideas of the cycle were killed or tied</b> (clamp, per-range thresholds, soft targets not trained, the Poisson set loss): the far-recall gap is not a threshold or a clamp problem.",',
    '"<b>Four loss and threshold ideas of the cycle were killed or tied</b> (clamp, per-range thresholds, soft targets not trained, the Poisson set loss): the far-recall gap is not a threshold or a clamp problem. '
    'The Poisson set likelihood looked fine at its own 18.7 vertices per 100 m and lost 0.073 Q once held to the same 14.3 vertices as the controls (E255): <b>the matched vertex budget decides such verdicts</b>.",')
rep('"<b>The research framework held, and it also hurt in specific places</b>: the derived-state tools (D16) worked, a stalled launcher could be resumed, and a shared machine was the largest source of lost time (section 6).",',
    '"<b>The research framework held, and it also hurt in specific places</b>: the derived-state tools (D16) worked, a stalled launcher could be resumed, and a shared machine was the largest source of lost time: the planned timing repeat (E253) could not run at all because another chat started a 7-hour training in its window (section 6).",')
rep('the seed floor is estimated from three or four pairs', 'the seed floor is estimated from four pairs')
rep('"One seed pair is a hint, not a rule: no early-kill threshold was adopted.")',
    '"With the four cross-pairings of two seeds each the cut test at 50 % passed (j1rc above j1eq by 0.05 - 0.08 lanes found in 4 of 4, no wrong cut), but the effect is as large at 25 % as at 50 % (it overstates the final +0.055 by up to a third) and Q never separates; "\n'
    '    "four pairings that share runs only bound the wrong-cut rate to about 0.5, so the rule stays a candidate for a 12-pairing replay and covers loss-weight type effects only (E241).")')

# section 5
rep('''    ["3", "E255", "Poisson set loss, full schedule, compared at a matched vertex budget", "1 training + grids", "see 5.3"],
    ["4", "E253", "timing repeat on a quiet machine with a machine guard", "45 min exclusive", "see 5.4"]],''',
    '''    ["3", "E255", "Poisson set loss, full schedule, compared at a matched vertex budget", "1 training + grids", "see 5.3"],
    ["4", "E256 (E253 blocked)", "the planned timing repeat on a quiet machine (E253) could not run: another chat's 7-hour training started in its window. Replaced by: do the D4 candidates keep their order in each city?", "CPU 10 min", "see 5.4 / 5.5"]],''')
rep('''d.p("Pair 3 had been cancelled the night before when the shared machine slowed training twenty-fold; the stalled ids were resumed through the launcher. It was pinned to the stored distance map with <code>LCR_DIST_DIR</code> so that the four pairs stay comparable "
    "(the loader default is now the exact map). The table in 4.1 and the tiles of section 1 already include it when it finished.")''',
    '''d.p("Pair 3 had been cancelled the night before when the shared machine slowed training twenty-fold; the stalled ids were resumed through the launcher (E240.a3, E240.c3). It was pinned to the stored distance map with <code>LCR_DIST_DIR</code> so that the four pairs stay comparable "
    "(the loader default is now the exact map). Pair 3: Q 0.7391 (j1rc) against 0.7316 (j1eq), lanes found 0.7393 against 0.6815. With four pairs the tool still says "
    + TIE("TIE") + " (Q a - c +0.0062, prediction +0.006 hit); the seed SDs moved by at most 0.0006 against the three-pair estimate (Q 0.0030 / 0.0020, paired 0.0043), so the floor is stable. The table in 4.1 and the tiles of section 1 include the pair. "
    "The intermediate checkpoints of pair 3 gave the E241 cross-pairings (section 4.1).")''')
rep('''    d.p(f"Decision by the card's rule: Q a - c = {m143['e25_j1phd2_s0']['Q'] - 0.5 * (m143['e24_j1ex_s0']['Q'] + m143['e24_j1ex_s1']['Q']):+.4f} (prediction -0.045 ± 0.020).")
else:
    d.p("E255 did not finish in this report's window; see the experiment file.")
d.fig(F + "w3_matched_vertex.png", "E255 / E244: Q against the vertex budget on the tuning subset (left) and the first, cut attempt at its 50 % checkpoint (right).")''',
    '''    d.p(f"{NEG('CUT')} by <code>decide.py</code>: Q a - c = {m143['e25_j1phd2_s0']['Q'] - 0.5 * (m143['e24_j1ex_s0']['Q'] + m143['e24_j1ex_s1']['Q']):+.4f} (rule: CUT at -0.020 on one seed; the topo F1 guard is violated by -0.127). "
        "The prediction -0.045 ± 0.020 missed by -0.028: the loss was worse than the half-schedule reading. The procedure: sweep tau and the vertex radius on 250 tuning frames, interpolate each curve to 14.3 vertices per 100 m, take the best tau, evaluate on VAL 873 and report the 623 untouched frames (Q 0.6519 against 0.7226 / 0.7241: the tuning inflated neither arm). "
        "The stored cards of all three models were reproduced first (layer 1 passed).")
    d.p("<b>Why:</b> at its own settings the Poisson arm sits at 18.7 vertices per 100 m with Q 0.723 - a quarter more vertices than the controls' 15.1 - because the set likelihood spreads the validity scores. Squeezed to 14.3 it loses 0.07 Q while the controls lose 0.006 over the same squeeze: thinning removes true lanes, not duplicates. "
        "K193 is closed: a matching-free likelihood needs a calibrated score before a vertex budget can be set. One seed is enough for the CUT rule (the gap is about 17 paired SDs).")
else:
    d.p("E255 did not finish in this report's window; see the experiment file.")
d.fig(F + "w3_matched_vertex.png", "E255 / E244: Q against the vertex budget on the tuning subset (left; thin lines = other tau values, dashed line 14.3) and the first, cut attempt at its 50 % checkpoint (right).")''')
rep('''d.h2("5.4  Timing on a quiet machine (E253)")
d.fig(F + "w3_timing.png", "Replicate medians per configuration, ms per frame: the invalid session of loop 19 (left) and the repeat with the machine guard (right).")
d.p("<i>Result text of E253: see the experiment file.</i>")''',
    '''d.h2("5.4  Timing on a quiet machine (E253) - blocked")
d.p(f"{INFO('not run')} The repeat needs an exclusive machine for 45 minutes (idle GPU, load average below 3, no other chat's job). The BEV-system chat had announced a 7 - 9 hour training right after the window and started it while loop 20 still ran its trainings; it asked that no timing run alongside it, "
    "and the guard of the driver (<code>temp/loop20/run_e253.sh</code>: GPU use, WSL load, Windows CPU, repair rounds) would have voided every replicate. The card stays planned (item E253.t); the timing numbers of this report are still the stored loop-18 values and the ordering of the invalid E243 session. "
    "This is the clearest case for a machine lease between chats (N1): an exclusive window cannot be agreed by message alone when two chats plan the same hours.")
d.h2("5.5  The D4 candidates by city (E256, replaces the timing iteration)")
d.p("E252 showed that all models score higher in Boston than in Singapore. D4 chooses between candidates on pooled numbers, so we asked whether their order holds in each city (CPU, 10 minutes). Rule: a pair that is ordered one way in Boston and the other way in Singapore by more than 0.010 in both cities is a flip.")
d.table([
    ["fast_recall (j1rc 192 / 96)", "0.7757", "0.7061", "+0.070", "0.774 / 0.702", "15.0 / 15.0"],
    ["recall (rc1 + pruning 0.4)", "0.7700", "0.7137", "+0.056", "0.803 / 0.736", "15.6 / 15.5"],
    ["control (E94 trace network)", "0.7661", "0.7119", "+0.054", "0.830 / 0.773", "21.7 / 21.9"],
    ["champion (j1 + pruning 0.3)", "0.7625", "0.7101", "+0.052", "0.734 / 0.660", "14.9 / 14.8"],
    ["fast (j1, 192 / 96)", "0.7515", "0.7064", "+0.045", "0.709 / 0.636", "15.0 / 15.0"]],
    header=["candidate", "Q Boston", "Q Singapore", "gap", "lanes found B / S", "vertices B / S"], num=range(1, 6))
d.p(f"{OK('ranking confirmed')} No pair flips (prediction hit). What the split adds: the pooled advantage of fast_recall over fast (+0.011) is a Boston effect (+0.024 against -0.000 in Singapore), while the lanes-found gain (+0.065 in both cities) is the same everywhere. "
    "So D4 may cite 'lanes found +0.06 at equal Q' as evidence; the order of the five Q values inside the 0.012 pooled spread is carried by 10 Boston scenes and is not established. Two cities only; the vertex budgets are not matched here (the control sits at 21.7).")''')

# section 6 framework
rep('''    ["changed", "matched-vertex procedure as code''',
    '''    ["hurt: exclusive windows", "the timing window (E253) was agreed by message with the BEV-system chat and then overrun by its training start: the iteration was replaced by a CPU one (E256). A lease (N1) is the fix, not more messages. The cut test of E241 (cross-pairings) had no tool support and was decided by hand: a subcommand of <code>decide.py</code> is open (N15)"],
    ["changed", "matched-vertex procedure as code''')

# section 8
rep('The recall clamp (E244 arm b), per-range thresholds (E246), the first Poisson attempt (E244 arm a, partial), a quick timing session on a shared machine (E243).",',
    'The recall clamp (E244 arm b), per-range thresholds (E246), the Poisson set loss (E244 arm a partial, E255 final: -0.073 Q at 14.3 vertices), a quick timing session on a shared machine (E243) and the planned quiet-machine repeat (E253, blocked).",')
rep('Two cities in VAL; 22 scenes; the bands rest on three or four seed pairs.', 'Two cities in VAL; 22 scenes; the bands rest on four seed pairs (the standard error of an SD with n = 4 is about 40 %).')
rep('A machine lease between chats before the next timing session (N1) and the metric-specific noise bands as a DECISIONS entry once four pairs exist.',
    'A machine lease between chats before the next timing session (N1; E253 waits for the BEV chat\'s training to end), and the metric-specific noise bands (Q 0.0025, lanes found 0.004, R@1 far 0.003, topo F1 0.001) as a DECISIONS entry now that four pairs exist.')

# section 9
rep('''    ["E255 card", "'one seed is enough for CUT at -0.010'", "that is about 2 paired SDs; CUT now needs -0.020 on one seed"],''',
    '''    ["E255 card", "'one seed is enough for CUT at -0.010'", "that is about 2 paired SDs; CUT now needs -0.020 on one seed"],
    ["E240 summary (loop 19)", "seed SD from three pairs (Q 0.0036 / 0.0022; 8.6 and 6.9 paired SDs)", "four pairs: Q 0.0030 / 0.0020, paired 0.0043; lanes found 10.1 and R@1 far 7.7 paired SDs"],
    ["E256 draft (author)", "'pooled advantage of fast_recall over fast +0.0197'", "+0.0111 (0.7392 against 0.7281), found before the reviewer pass"],
    ["E255 draft (author)", "'about 20 paired SDs'; native point described as the E244 reading", "17 paired SDs (0.0726 / 0.0043); the native point is the registry card (18.71 vertices, Q 0.7227)"],''')
# section 10
rep('["E240", "seed noise of the recall fine-tune", TIE("TIE")],', '["E240", "seed noise of the recall fine-tune, four pairs", TIE("TIE")],')
rep('["E241", "early-kill replay of checkpoints", INFO("inconclusive by design")],', '["E241", "early-kill replay of checkpoints", OK("KEEP: worth a 12-pairing replay")],')
rep('["E253", "timing repeat with a machine guard", INFO("see 5.4")],', '["E253", "timing repeat with a machine guard", INFO("blocked, planned")],\n    ["E256", "D4 candidates by city", OK("ranking confirmed")],')
rep('["E255", "Poisson set loss, full schedule, matched vertex budget", INFO("see 5.3")],', '["E255", "Poisson set loss, full schedule, matched vertex budget", NEG("CUT")],')
open(p, "w", encoding="utf-8").write(t)
print("ok")
