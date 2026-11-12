p = "temp/e276/e276_build_report.py"
s = open(p, encoding="utf-8").read()
# E279 table and wording
s = s.replace('W = [card("e279", f"e279_w_s{s}.plain_t0.5") for s in (0, 1)]', 'W = [card("e279", f"e279_w_s{s}.plain_t0.5") for s in (0, 1)]\nWH = [card("e279", f"e279_w_s{s}.close1.5_t0.5") for s in (0, 1)]')
# short answer bullet: replace the E279 sentence
s = s.replace('''A smaller radius (1.25 m, E279) was queued: " + ("verdict " + V279 if V279 else "<span class='pending'>running, result pending</span>") + ".</li>"''',
 '''A smaller radius (1.25 m, E279) gives topology F1 " + m2(W, 'topo_F1') + ", junction F1 " + m2(W, 'junction_F1') + ", lanes found " + m2(W, 'lanes_found') + " and passes the same guards: verdict <b>" + (V279 or 'pending') + "</b> (thin margins on one seed, two seeds, in-sample).</li>"''')
# section 3: table with the E279 columns
old = s[s.index('rows = [[n, m2(B, k, s, f), m2(H, k, s, f), m2(U, k, s, f), m2(UH, k, s, f)]'):s.index('parts.append("<h2>3. E277: the step inside the network</h2>"')]
new = '''rows = [[n, m2(B, k, s, f), m2(H, k, s, f), m2(U, k, s, f), m2(W, k, s, f), m2(WH, k, s, f)] for n, k, s, f in (("topology F1", "topo_F1", "graph", "{:.3f}"), ("junction F1", "junction_F1", "graph", "{:.3f}"), ("F1 at 1 m", "F1@1.0", "graph", "{:.3f}"), ("lanes found", "lanes_found", "graph", "{:.3f}"),
                                                                                                         ("vertices per 100 m", "vert100_dp", "graph", "{:.1f}"), ("node recall", "node_R", "nodes", "{:.3f}"), ("node precision", "node_P", "nodes", "{:.3f}"))]
'''
s = s.replace(old, new)
s = s.replace('table(["VAL 873", "baseline B", "B + hard step (E275)", "head trained with the soft step (read at 0.5, no step)", "that + hard step"], rows)', 'table(["VAL 873", "baseline B", "B + hard step 1.5 m (E275)", "soft step 1.5 m (E277)", "soft step 1.25 m (E279)", "E279 + hard step 1.5 m"], rows)')
s = s.replace('<h2>3. E277: the step inside the network</h2>', '<h2>3. E277 / E279: the step inside the network</h2>')
s = s.replace('E279 repeats it at a radius of 1.25 m (same rule): " + (("verdict " + V279) if V279 else "running, pending") + ".</p>")', 'E279 repeats it at a radius of 1.25 m under the same rule: <b>" + (V279 or "pending") + "</b>. It keeps more lane recall (lanes found " + m2(W, "lanes_found") + " against " + m2(U, "lanes_found") + ") and gives up junction F1 (" + m2(W, "junction_F1") + " against " + m2(U, "junction_F1") + "); the margins on seed 0 are thin (topology gain 0.021 against the band 0.02). The hard 1.5 m step on top of it still drops some nodes (last column). So the thinning is learned by the network; which radius, or whether to add the hard step, is your call.</p>")')
open(p, "w", encoding="utf-8").write(s)
print("ok")
