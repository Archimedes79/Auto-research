p = "reports/build/reports/build_week3_report.py"
t = open(p, encoding="utf-8").read()


def rep(a, b):
    global t
    assert t.count(a) == 1, (t.count(a), a[:80])
    t = t.replace(a, b)


rep("the planned timing repeat (E253) could not run at all because another chat started a 7-hour training in its window (section 6).",
    "the planned timing repeat (E253) had to wait five hours for another chat's 7-hour training to end (section 6), and the session was valid once it ran.")
rep('"so the loop started at once. Each iteration is a card',
    '"so the loop started at once. Each iteration is a card')
rep('["4", "E256 (E253 blocked)", "the planned timing repeat on a quiet machine (E253) could not run: another chat\'s 7-hour training started in its window. Replaced by: do the D4 candidates keep their order in each city?", "CPU 10 min", "see 5.4 / 5.5"]],',
    '["4", "E253 (after a wait: E256)", "timing repeat on a quiet machine with a machine guard; it had to wait five hours for another chat\'s 7-hour training, and E256 (do the D4 candidates keep their order in each city?) used the wait", "45 min exclusive; CPU 10 min", "see 5.4 / 5.5"]],')

# 5.4 block
i = t.index('d.h2("5.4  Timing on a quiet machine (E253) - blocked")')
j = t.index('d.h2("5.5')
new54 = '''d.h2("5.4  Timing on a quiet machine (E253)")
d.p("The repeat needs the machine alone for 45 minutes. The BEV-system chat's 7-hour training ran in the planned window, so the run waited until it reported the end (07:19 on 27 October) and started at 07:20. "
    "Six configurations, six rounds in round-robin order, a fresh process per replicate, 200 timed frames each; the guards of the driver (GPU use, WSL load average, Windows CPU) voided no replicate.")
d.table([
    ["j1 192 / 96, pruning 0.3 (a)", "18.52", "0.44", "17.97 - 19.17", "7.8"],
    ["f1 288 / 144, pruning 0.3 (c)", "18.75", "0.29", "18.55 - 19.34", "7.5"],
    ["j1eq, pruning 0.4", "18.49", "0.21", "18.15 - 18.72", "7.7"],
    ["j1rc, pruning 0.4", "21.14", "0.39", "20.74 - 21.82", "8.9"],
    ["rc1 288 / 144, pruning 0.4", "20.81", "0.26", "20.46 - 21.20", "8.8"],
    ["f1 with time recursion (as trained)", "24.85", "0.76", "24.71 - 26.69", "7.8"]],
    header=["configuration (ms per frame, idle machine)", "median", "replicate SD", "range of 6", "assembly ms"], num=range(1, 5))
d.p(f"{OK('session valid')} 6 of 6 replicates per configuration, replicate SD at most 0.6 ms in 5 of 6 configurations (the recursion run has 0.76 ms), the stored loop-18 values reproduced within 1.0 ms. "
    f"{TIE('TIE')} j1 minus f1 = -0.23 ms (per-round differences -0.83 to +0.21; prediction -0.3 hit; the rule needs -1.0 ms for a speed claim): <b>j1 is not a speed variant</b>. "
    "The recall weight costs +2.65 ms (j1rc against j1eq; prediction +2.2) and the recursion +6.1 ms, as in the invalid first session, but all absolute values are 2 - 3 ms lower on the quiet machine. "
    "The chain therefore runs at 18.5 - 18.8 ms without recursion and 24.9 ms with it.")
d.fig(F + "w3_timing.png", "Replicate medians per configuration, ms per frame: the invalid session of loop 19 (left, shared machine) and the repeat with the machine guard (right).")
'''
t = t[:i] + new54 + t[j:]
rep('d.h2("5.5  The D4 candidates by city (E256, replaces the timing iteration)")', 'd.h2("5.5  The D4 candidates by city (E256, used the wait for the quiet machine)")')

rep("the timing window (E253) was agreed by message with the BEV-system chat and then overrun by its training start: the iteration was replaced by a CPU one (E256).",
    "the timing window (E253) was agreed by message with the BEV-system chat and then overrun by its training start: the run waited five hours and a CPU iteration (E256) used the time.")
rep('["champion (D7 f1 + fork pruning 0.3)", "0.735", "0.741", "0.695", "0.709", "0.409", "24 (17.5 without recursion)"],',
    '["champion (D7 f1 + fork pruning 0.3)", "0.735", "0.741", "0.695", "0.709", "0.409", "24.9 (18.8 without recursion; E253)"],')
rep('["recall (rc1 + pruning 0.4)", "0.740", "0.742", "0.767", "0.771", "0.428", ""],', '["recall (rc1 + pruning 0.4)", "0.740", "0.742", "0.767", "0.771", "0.428", "20.8 (E253)"],')
rep('"0.739", "0.737", "0.736", "0.727", "0.447", "20 - 22"],', '"0.739", "0.737", "0.736", "0.727", "0.447", "21.1 (E253)"],')
rep("a quick timing session on a shared machine (E243) and the planned quiet-machine repeat (E253, blocked).", "and a quick timing session on a shared machine (E243, invalid).")
rep("(N1; E253 waits for the BEV chat's training to end)", "(N1; E253 only got its quiet window because the BEV chat reported the end of its training)")
rep('["E253", "timing repeat with a machine guard", INFO("blocked, planned")],', '["E253", "timing repeat with a machine guard", TIE("TIE, session valid")],')
# 4.6 pointer
rep('Absolute numbers are quoted from the stored loop-18 timings (champion 24.2 ms with recursion, 17.5 ms without).")',
    'The repeat on a quiet machine is in 5.4 (E253): same ordering, absolute values 2 - 3 ms lower.")')
open(p, "w", encoding="utf-8").write(t)
print("ok")
