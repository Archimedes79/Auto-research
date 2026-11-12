p = "build_system_report.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new, count)


# density oracle number from its file
rep("P14 = J(\"temp/e282/pilot_outcome.json\")", "P14 = J(\"temp/e282/pilot_outcome.json\")\nDO = J(\"temp/e276/density_oracle.json\")\nORACLE_V = None if not DO else DO[\"dense (E273)\"][\"graph\"][\"vert100_dp\"]")
rep("the perfect network needs 11.4 for any node target: the excess is jitter and duplicates in the delivered geometry",
    "a perfect network needs {num(ORACLE_V, '{:.1f}')} for any node target (E276 P1): the excess is jitter and duplicates in the delivered geometry, not the node grid")

# nine cards
rep("Since the last whole-system report ten experiments ran.", "Since the last whole-system report nine experiment cards (E274 - E282) were worked off.")
rep("one was kept (the trained step at 1.25 m, E279). Details below.", "the close-node step was kept (as a rule on the output in E275, trained inside the head at 1.25 m in E279). Details below.")

# E277 phrase per seed
rep("""{V277 or 'CUT'}: it reproduces the hard step but lanes found falls by {sg(cut(U, B, 'lanes_found'))} against B""",
    """{V277 or 'CUT'}: it reproduces the hard step, but lanes found changes by {' / '.join(sg(g(u, 'lanes_found') - g(b, 'lanes_found')) for u, b in zip(U, B))} against B (seed 0 / seed 1)""")

# the reading note under the chart: computed
rep("""P("<div class='note'>Reading. The step buys precision: the junction precision rises from about 0.07 to 0.14-0.18 while the junction recall falls from about 0.65 to 0.55; the topology F1 rises by 0.02-0.03; the lanes found fall a little. The best graph so far is still ahead on every line except vertices-vs-budget parity. The trained step at 1.25 m (E279) keeps the gain of the hard step with a slightly higher lanes-found, but the margins of the guards are thin on seed 0, and the numbers are in-sample on VAL.</div>")""",
    """def rg(cs, k):
    v = [g(c, k) for c in cs]
    return f"{min(v):.2f} - {max(v):.2f}"


P(f"<div class='note'>Reading. The step buys precision with recall: the junction precision rises from {rg(B, 'junction_P')} (B) to {rg(W, 'junction_P')} (E279) and {rg(H, 'junction_P')} (hard step), the junction recall falls from {rg(B, 'junction_R')} to {rg(W, 'junction_R')} and {rg(H, 'junction_R')}; "
  f"the topology F1 rises by {num(mean([g(c, 'topo_F1') for c in W]) - mean([g(c, 'topo_F1') for c in B]), '{:+.3f}')} (E279) and {num(mean([g(c, 'topo_F1') for c in H]) - mean([g(c, 'topo_F1') for c in B]), '{:+.3f}')} (hard step); the lanes found change by {num(mean([g(c, 'lanes_found') for c in W]) - mean([g(c, 'lanes_found') for c in B]), '{:+.3f}')} (E279). "
  f"The best graph so far is ahead on topology, F1@1.0, lanes found, junction precision and vertices; the end-to-end head has the higher junction RECALL ({rg(W, 'junction_R')} against {num(D.get('junction_R'))}), i.e. it finds more of the junctions but also invents many (precision). "
  f"The trained step at 1.25 m (E279) keeps most of the gain of the hard step with a little more lanes found; the margins of its guards are thin on seed 0, and all numbers are in-sample on VAL.</div>")""")

# captions of the pictures (written after looking at them)
rep("""fr = [(643, "frame 643"), (300, "frame 300"), (100, "frame 100")]
for i, nm in fr:
    p = os.path.join(ROOT, "results", "E279", "figures", f"frame_{i}.png")
    P(img(p, f"VAL {nm}: left to right the baseline B, B with the hard close-node step (E275), the E279 head (soft step inside)"))
P("<p>What to look at: the baseline carries a dense row of nodes and many short parallel links (red halos); the hard step and the trained step thin the row and remove a large part of the false junction links, at the price of some missed branches (blue). The long connector across the junction mouth is where the remaining red sits.</p>")""",
    """CAP = {643: "VAL frame 643, a large three-way junction. Rows: the baseline B, B with the hard close-node step (E275), the E279 head (soft step inside). Left: delivered nodes and links against the ground truth (red halo = false, blue = missed); right: the delivered graph coloured by the assigned map piece. "
              "B delivers several overlapping connector curves through the junction mouth; in the two thinned variants the overlap is largely gone and the curves are cleaner; in all three rows blue curves on the left mark ground-truth connectors that are not found. The scorer's numbers of the frame are printed under each panel.",
       300: "VAL frame 300, a road with a crossing at both ends. The long straight stretch is right in all three variants; the differences sit at the two crossings, where the nodes of B are doubled (stacked event symbols) and the thinned variants carry fewer of them; false (red) nodes remain at the upper crossing.",
       100: "VAL frame 100, a curved road with a branch at the upper right. The three long lanes are right in all variants; at the branch false nodes (red halo) remain in all three rows and a ground-truth loop (blue) is missed in all three; the E279 row carries fewer nodes there."}
for i in (643, 300, 100):
    P(img(os.path.join(ROOT, "results", "E279", "figures", f"frame_{i}.png"), CAP[i]))""")
rep("""figcaption>{caption}</figcaption></figure>"\n\n\ndef webm""", """figcaption>{caption}</figcaption></figure>"\n\n\ndef webm""", 0) if False else None

# provenance and small labels
rep("the node recall inside intersections is the weakest part (E273 analysis)", "the node recall inside intersections is the weakest part (E273 analysis of 2026-11-05: 0.54 inside the widened intersection mask against 0.80 outside)")
rep("The old chain closed its junction gap with a rule (fork pruning)", "The old chain closed its junction gap with a rule (fork pruning, loop 18)")
open(p, "w", encoding="utf-8").write(s)
print("fix2 ok")
