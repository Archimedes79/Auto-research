p = "reports/build/reports/build_status21_report.py"
s = open(p, encoding="utf-8").read()
R = [
 ('("15.5 → 10.4", "vertices per 100 m, recall → recall_dp", f"Douglas-Peucker output stage 0.2 m (E257): Q {n(\'recall\', \'Q\')} → {n(\'recall_dp\', \'Q\')}, lanes found and R@1 far unchanged"),',
  '("15.5 → 10.4", "node ids per 100 m, recall → recall_dp", f"Douglas-Peucker output stage 0.2 m (E257): Q {n(\'recall\', \'Q\')} → {n(\'recall_dp\', \'Q\')}; a re-count of the same geometry: on the budget\'s own basis the count is unchanged (reviewer, section 2)"),'),
 ('"<b>Loop 21 made the system simpler and sparser, not better.</b> Quality (Q about 0.74) has been at the same level since loop 18; what changed is the graph the system hands over: one third fewer vertices at the same Q, and a smaller, hash-verified assembly.",',
  '"<b>Loop 21 made the system simpler and its output smaller, not better.</b> Quality (Q about 0.74) has been at the same level since loop 18; what changed is the graph the system hands over (one third fewer nodes for the same geometry, so the same Q by construction) and a smaller, hash-verified assembly.",'),
 ('["E257", "Douglas-Peucker output stage on the assembled graph", "eps 0.3 m: 15.0 → 9.0 vertices per 100 m, Q −0.0008 (paired scene bootstrap of −0.0014 … −0.0001); eps 0.2 m holds every guard: fast_recall 15.0 → 9.9, champion 14.8 → 9.5, Q −0.0003 / −0.0005", OK("KEEP, adopted as D20")],',
  '["E257", "Douglas-Peucker output stage on the assembled graph", "eps 0.3 m: 15.0 → 9.0 node ids per 100 m, Q −0.0008 (paired scene bootstrap of −0.0014 … −0.0001); eps 0.2 m (chosen after the sweep, because the champion misses the F1@0.5 guard by 0.0003 at 0.3): fast_recall 15.0 → 9.9, champion 14.8 → 9.5, Q −0.0003 / −0.0005. The oracle-basis count is unchanged (12.2 / 11.3)", TIE("KEEP by pooled rule, adopted as D20; caveats")],'),
 ('"Left: Q against vertices per 100 m. The grey curve is the old way to get fewer vertices (a larger vertex radius): it loses 0.12 Q at 12 vertices; the DP output stage (blue) loses 0.0003 at 9.9. Right: the quality metrics of control and the three _dp systems.")',
  '"Left: Q against node ids per 100 m. The grey curve merges vertices (a larger vertex radius) and removes geometry: it loses 0.12 Q at 12 vertices. The blue curve (DP output stage) only removes nodes on straight runs; the two are not a like-for-like quality curve (reviewer). Right: the quality metrics of control and the three _dp systems.")'),
 ('f"<b>The stage is free:</b> Q changes by', 'f"<b>The stage costs nothing on the metric,</b> by construction (same geometry, fewer nodes): Q changes by'),
 ('for 33 - 36 % fewer vertices; lanes found', 'for 33 - 36 % fewer node ids; lanes found'),
 ('["Sparsity", "−33 … −36 % vertices at ≤ −0.0005 Q (E257); node-id counts remain for the route / map pipeline", OK("improved")],',
  '["Sparsity", "−33 … −36 % node ids at ≤ −0.0005 Q (E257); on the budget\'s own basis (per-piece DP count) 12.2 / 11.3 before and after, so no gain there; node-id graphs remain for the route / map pipeline", TIE("smaller output, same content")],'),
 ('"Speed", "fast_recall 20 - 22 ms per frame against 628 ms (loop 18); no valid new timing in loop 21", INFO("not re-measured")', '"Speed", "fast_recall 20 - 22 ms per frame against 628 ms (loop 18); no valid new timing in loop 21 (the relative stage savings of E258 / E259 are provisional)", INFO("not re-measured")'),
 ('    "<b>Housekeeping:</b> vectorise the DP stage only if it stays in the product (about 1 ms on a loaded machine); keep commits by path (the BEV chat\'s files are not mine).",',
  '    "<b>Correct the records the reviewer found</b> (small): the E258 timing header and E259 slips, the stale HANDOVER lines, a stored result file for the vertex-basis probe (done: E257 <code>vertex_basis.txt</code>); the PROBLEM.md sentence on the vertex count needs a grant naming a decision (it compares a node-id count with the 14.3 budget). Vectorise the DP stage only if it stays in the product.",'),
 ('["E257", "DP output stage", OK("KEEP, D20 adopted"),', '["E257", "DP output stage", TIE("KEEP (pooled), D20 adopted; caveats"),'),
]
for a, b in R:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
