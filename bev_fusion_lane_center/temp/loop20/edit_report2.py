p = "reports/build/reports/build_week3_report.py"
t = open(p, encoding="utf-8").read()


def rep(a, b, cnt=1):
    global t
    assert t.count(a) == cnt, (t.count(a), a[:80])
    t = t.replace(a, b)


# --- 4.1 table with a vertex column
rep('''                 f3(a["R@1_far"]), f3(c["R@1_far"]), f3(a["topo_F1"]), f3(c["topo_F1"])])''',
    '''                 f3(a["R@1_far"]), f3(c["R@1_far"]), f3(a["topo_F1"]), f3(c["topo_F1"]), f"{a['vert100']:.2f} / {c['vert100']:.2f}"])''')
rep('''rows.append(["mean", f3(qrc["mean"]), f3(qeq["mean"]), f"{mean(dq):+.4f}", f3(lrc["mean"]), f3(leq["mean"]), f"{mean(dl):+.3f}", "", "", "", ""])
rows.append(["SD", f4(qrc["sd"]), f4(qeq["sd"]), "", f4(lrc["sd"]), f4(leq["sd"]), "", "", "", "", ""])''',
    '''vrc = [card(f"e24_j1rc_s{s}")["vert100"] for s in seeds]
veq = [card(f"e24_j1eq_s{s}")["vert100"] for s in seeds]
rows.append(["mean", f3(qrc["mean"]), f3(qeq["mean"]), f"{mean(dq):+.4f}", f3(lrc["mean"]), f3(leq["mean"]), f"{mean(dl):+.3f}", "", "", "", "", f"{mean(vrc):.2f} / {mean(veq):.2f}"])
rows.append(["sample SD (n - 1)", f4(qrc["sd"]), f4(qeq["sd"]), "", f4(lrc["sd"]), f4(leq["sd"]), "", "", "", "", "", ""])''')
rep('''header=["", "Q j1rc", "Q j1eq", "diff", "lanes j1rc", "lanes j1eq", "diff", "R@1far rc", "R@1far eq", "topo rc", "topo eq"], num=range(1, 11), hl=(nS,), widths=None)''',
    '''header=["", "Q j1rc", "Q j1eq", "diff", "lanes j1rc", "lanes j1eq", "diff", "R@1far rc", "R@1far eq", "topo rc", "topo eq", "vert100 rc / eq"], num=range(1, 12), hl=(nS,), widths=None)''')
rep('"Prediction of the card: Q +0.006, hit.")',
    '"Prediction of the card: Q +0.006, hit. Every SD in this report is a sample SD (n - 1). j1rc carries +0.16 vertices per 100 m on average (+0.04 / +0.22 / +0.34 / +0.06 per pair), worth about 0.002 of the Q gap at the E236 slope. "\n'
    '    f"Q is positive in all four pairs (+0.006, standard error 0.002, paired t about 2.9 on 3 degrees of freedom): below the decision band of 0.010, so a TIE by the rule, but not shown to be zero.")')

# --- section 1
rep('Lanes found +0.06 and R@1 far +0.05 in every one of four pairs, Q +0.006 (TIE).',
    'Lanes found +0.05 to +0.07 and R@1 far +0.04 to +0.05 in every one of four pairs; Q +0.006 +- 0.002 (standard error), positive in all pairs but below the 0.010 band (TIE by the rule, not shown to be zero).')
rep('TEST was not touched this week (one check on loop 18 is logged in',
    'TEST was not touched in loops 19 and 20 (the one check of loop 18, on 25 October, is logged in')
rep('f"j1rc minus j1eq, mean of {nS} pairs: lanes found higher in every pair, Q a TIE (tool band 0.010)"',
    'f"j1rc minus j1eq, mean of {nS} pairs: lanes found higher in every pair, Q +{mean(dq):.3f} below the tool band 0.010 (TIE)"')
rep('The 0.010 band is 4 SDs for Q; "\n    "a single pair cannot see anything below about 0.01.",',
    'The 0.010 band is 4 seed SDs and 2.3 paired SDs for Q; "\n    "a single pair cannot see anything below about 0.01.",')

# --- E241 sentence
rep('(it overstates the final +0.055 by up to a third) and Q never separates; "',
    '(+0.065 to +0.078 against a final +0.052 to +0.058: 20 to 50 % more; part of it is the +0.4 to +0.6 vertices that j1rc carries at 25 - 75 %) and Q stays between +0.001 and +0.012 without a stable sign of separation; "')

# --- 4 intro
rep('Twelve cards, refereed before the first run; the full test suite passed 14 of 14 before and after. 30 runs, 9.5 GPU hours in the loop table; "\n    "10 of the 30 runs are failed or cancelled launchers and a machine that other chats shared (section 6).")',
    'Twelve cards, refereed before the first run; the full test suite passed 14 of 14 before and after. The loop table shows 33 runs and 11.3 GPU hours (it also counts the first loop-20 runs, D17); "\n    "7 are failed launchers, on a machine that other chats shared (section 6).")')

# --- 4.4
rep('The lowest interval bound is +0.002: close to zero in one cell.', 'The lowest interval bound of the two new draws is +0.002; one cell of the stored draw reaches 0.0.')

# --- section 5 intro
rep('The t1 training of the BEV-system chats had finished at 14:58, "\n    "so the loop started at once. Each iteration is a card with a referee verdict before its run, a written decision rule, and a summary.")',
    'The t1 training of the BEV-system chats had finished, "\n    "so the loop started at once. Each iteration is a card with a written decision rule and a summary; the cards of E252, E255 and E253 went through a referee before their runs, the E256 card (a CPU follow-up that cannot change a model) did not.")')

# --- 5.1 offset claim
rep('so an offset of that size would be visible. "', 'so an offset of 0.5 m or more would be clearly visible (the 0.3 m shift, 0.0225, is below the null SD of the city gap). "')

# --- 5.2
rep('the seed SDs moved by at most 0.0006 against the three-pair estimate (Q 0.0030 / 0.0020, paired 0.0043), so the floor is stable.',
    'the Q seed SDs moved by at most 0.0006 against the three-pair estimate (0.0036 / 0.0022 became 0.0030 / 0.0020; the paired SD 0.0052 became 0.0043), so the Q floor is stable; the lanes-found SDs moved by up to 0.0007.')

# --- 5.3
rep('Squeezed to 14.3 it loses 0.07 Q while the controls lose 0.006 over the same squeeze: thinning removes true lanes, not duplicates. "\n        "K193 is closed: a matching-free likelihood needs a calibrated score before a vertex budget can be set. One seed is enough for the CUT rule (the gap is about 17 paired SDs).")',
    'Squeezed from 18.7 to 14.3 vertices it loses 0.07 Q (0.016 per vertex removed), while the controls lose 0.007 over their smaller squeeze from 15.1 to 14.3 (0.0085 per vertex): about twice the cost per vertex, and lanes found falls 0.640 to 0.551. '
    '"\n        "Our reading is that the set likelihood spreads the validity scores so that thinning removes true lanes; no score-calibration analysis was made. K193 is closed as a loss swap; a calibration step is untested. One seed is enough for the CUT rule (the gap is 3.6 times the CUT threshold, about 17 paired SDs of the E240 contrast).")')
rep('(Q 0.6519 against 0.7226 / 0.7241: the tuning inflated neither arm)', '(Q 0.6519 against 0.7226 / 0.7241: the two tuned scalars did not overfit the 250 frames; the held-out frames come from the same 22 scenes)')

# --- 5.5
rep('d.p(f"{OK(\'ranking confirmed\')} No pair flips (prediction hit). What the split adds:',
    'd.p(f"{TIE(\'no flip, order not established\')} No pair flips, but the rule cannot fire easily: the whole Singapore spread is 0.008, below the 0.010 flip threshold. What the split adds:')
rep('is a Boston effect (+0.024 against -0.000 in Singapore), while the lanes-found gain (+0.065 in both cities) is the same everywhere. "\n    "So D4 may cite \'lanes found +0.06 at equal Q\' as evidence;',
    'comes from Boston (+0.024 against -0.000 in Singapore; one model pair, 10 Boston scenes, scene intervals about +-0.04: no more than the seed-to-seed spread of E240), while the lanes-found gain (+0.065 in both cities) is the same everywhere. "\n    "So D4 may cite \'lanes found +0.06\' as evidence (Q is higher in Boston only);')

# --- 6
rep('the referee found real flaws in 3 of 3 loop-20 cards (null and scale for E252, an unequal matching rule for E255, an unusable load guard for E253)',
    'the referee found real flaws in all 3 refereed loop-20 cards (null and scale for E252, an unequal matching rule for E255, an unusable load guard for E253)')

# --- 7 footnote
rep('stated at each system\'s own operating point (14.8 - 15.5 vertices per 100 m); at a matched 14.3',
    'stated at each system\'s own operating point (14.8 - 15.5 vertices per 100 m; the control sits at 21.8, so its row is not at the same budget); at a matched 14.3')

# --- 8
rep('and the metric-specific noise bands (Q 0.0025, lanes found 0.004, R@1 far 0.003, topo F1 0.001) as a DECISIONS entry now that four pairs exist.',
    'and metric-specific noise bands as a DECISIONS entry (D18) now that four pairs exist: seed SDs Q 0.0025, lanes found 0.004, R@1 far 0.003, topo F1 0.001; a band is about 2 paired SDs (Q 0.009, lanes found 0.012), not 1 SD.')
rep('"The seed SD mixes runs', '"The seed SD mixes runs') if False else None
rep('"Tuning of tau and vtx_r on VAL subsets (declared), one or two seeds per arm.",',
    '"Tuning of tau and vtx_r on VAL subsets (declared), one or two seeds per arm; the four seeds of E240 were trained at three code commits and under different machine loads.",')

# --- 9
rep('"four pairs: Q 0.0030 / 0.0020, paired 0.0043; lanes found 10.1 and R@1 far 7.7 paired SDs"', '"four pairs: Q 0.0030 / 0.0020, paired 0.0043; lanes found 10.1 and R@1 far 7.6 paired SDs (exact; the loop-19 ratios were 8.6 and 7.0, not 6.9)"')
rep('["E241 summary", "\'Q overstates at 25 %\'", "within one paired SD of the E240 mean: Q is not separable from noise at a quarter of the schedule; the lanes-found prediction at 50 % missed on size"],',
    '["E241 summary", "\'Q overstates at 25 %\'", "within about one paired SD of the E240 mean in pair 2; in pair 3 (cross-pairing rc2-eq3 +0.0109) just above it; Q is not separable from noise at a quarter of the schedule; the lanes-found prediction at 50 % missed on size"],\n'
    '    ["this report, first draft", "\'ranking confirmed\' (E256), \'would be visible\' (E252), \'controls lose 0.006 over the same squeeze\' (E255), \'TEST not touched this week\', \'seed SDs moved by at most 0.0006\'", "found by the numbers reviewer: the E256 rule cannot fire on a 0.008 spread; only a 0.5 m shift is clearly visible; the controls lose 0.007 over a squeeze of 0.8 vertices, the Poisson arm 0.07 over 4.4; the loop-18 TEST check of 25 October lies in the week; the Q paired SD moved by 0.0009"],')

# --- 10
rep('["E256", "D4 candidates by city", OK("ranking confirmed")],', '["E256", "D4 candidates by city", TIE("no flip, order not established")],')
open(p, "w", encoding="utf-8").write(t)
print("ok")
