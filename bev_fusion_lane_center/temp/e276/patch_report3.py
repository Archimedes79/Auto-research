p = "temp/e276/e276_build_report.py"
s = open(p, encoding="utf-8").read()
# 1. verdict of E277 parsed from the log
s = s.replace("\ndef long_stats():", '''
def verdict_of(tag):
    """the "verdict" of the last `== verdict277 ...<tag>` block of verdicts.log"""
    i = verd.rfind(tag)
    if i < 0:
        return None
    j = verd.find('"verdict": "', i)
    return None if j < 0 else verd[j + 12: verd.find('"', j + 12)]


V277 = verdict_of("== verdict277 u against")
V279 = verdict_of("== verdict277 for E279")
W = [card("e279", f"e279_w_s{s}.plain_t0.5") for s in (0, 1)]


def long_stats():''', 1)
# 2. short answer bullet 2
i0 = s.index('f"<li><b>E277 puts that step inside the network</b>')
i1 = s.index('"</li>"', i0) + len('"</li>"')
new = '''f"<li><b>E277 puts that step inside the network (trained, soft)</b>: topology F1 {m2(U, 'topo_F1')}, junction F1 {m2(U, 'junction_F1')} (the hard step: {m2(H, 'topo_F1')} / {m2(H, 'junction_F1')}), so it reproduces the step with no rule step left, but on one seed it loses more lane recall than the guards allow (lanes found {m2(U, 'lanes_found')} against {m2(B, 'lanes_found')}): by the rule written before the run the verdict is <b>{V277 or 'pending'}</b>. A smaller radius (1.25 m, E279) was queued: " + ("verdict " + V279 if V279 else "<span class='pending'>running, result pending</span>") + ".</li>"'''
s = s[:i0] + new + s[i1:]
# 3. section 3 paragraph
old = s[s.index('("<p class=\'pending\'>The trained heads are not scored yet'):s.index('("<div class=\'note\'><pre>" + html.escape(verd[-3200:])')]
s = s.replace(old, '''("<p class='pending'>The trained heads are not scored yet.</p>" if not all(U) else "<p>Rule (card E277, written before the run): KEEP when the soft step lifts topology F1 by 0.02 and junction F1 by 0.03 over B on both seeds, with every guard held (F1 at 1 m not below B by more than 0.01, lanes found by more than 0.03, the assignment guards) and a topology F1 within 0.01 of the hard step. Result: <b>" + (V277 or "pending") + "</b>. Both gains hold on both seeds and the hard step on top of the soft one removes no further node; the guards fail on seed 0 only (lanes found -0.039, F1 at 1 m -0.015); seed 1 holds all of them. Whether the guard values of the hard step are the right ones for a trained step is your call; the hard step stays the read-out meanwhile. E279 repeats it at a radius of 1.25 m (same rule): " + (("verdict " + V279) if V279 else "running, pending") + ".</p>") +
             ''', 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
