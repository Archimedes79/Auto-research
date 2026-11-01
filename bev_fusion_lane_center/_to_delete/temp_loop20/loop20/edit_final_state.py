def sub(p, pairs):
    t = open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert a in t, (p, a[:60])
        t = t.replace(a, b)
    open(p, "w", encoding="utf-8").write(t)
sub("experiment_execution/PLAN.md", [
    ("BLOCKED: the BEV chat's 7-hour training (t1_r4_soa) runs in the window; item E253.t stays planned; replaced by I4b |", "done (E253, 07:20 after the BEV chat's training ended): session valid, TIE, j1 - f1 -0.23 ms (prediction -0.3 hit), recall +2.65 ms, recursion +6.1 ms |"),
    ("shutdown NOT done (the BEV chat's training t1_r4 runs)", "E253 ran 07:20 - 07:38 on a quiet machine"),
])
sub("HANDOVER.md", [
    ("# HANDOVER - 2026-10-27 02:30", "# HANDOVER - 2026-10-27 08:00"),
    ("**E253 timing is blocked** (t1_r4 runs; item E253.t planned; run it when no other chat's job is up: `run.py --id E253.t ... -- bash temp/loop20/run_e253.sh`, then `analyse_e253.py`).", "E253 timing ran 07:20 after the BEV chat's training ended: valid session, TIE (j1 - f1 -0.23 ms), j1rc +2.65 ms over j1eq, recursion +6.1 ms."),
    ("1. After t1_r4 ends: E253 (45 min exclusive); the 12-pairing E241 replay only if D4 needs it.", "1. The 12-pairing E241 replay only if D4 needs it; nothing else is queued."),
    ("**no shutdown while `pgrep -f t1_r4` matches**", "t1_r4 ended 07:19, the BEV chat now does CPU work only"),
])
