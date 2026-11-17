import io

p = "/mnt/c/Users/mmeut/.claude/projects/C--Auto-research/memory/lane-e2e-day-d109-d111-2026-11-16.md"
s = io.open(p, encoding="utf-8").read()
old_start = s.index("- **E358**")
old_end = s.index("**Why:**")
new = """- **E358** (learned wiring head on the ATTACHED ends, control = E324 s PRIMARY 0.6912 / 0.6948): **KEEP+** (head 0.6931 / 0.6964 = +0.0019 / +0.0016, bootstrap above 0 on both seeds, all guards held, prediction hit; +545 parameters, **+2.0 ms per frame on the host** measured on a quiet host: 'faster?' NO). The live check needed FOUR attempts (CPU REFUSED, GPU REFUSED frame 46 `pairs` by 0.0022, GPU bs 8 REFUSED, GPU faithful two-loader replay (B bs 4 exact, pair model bs 8 error) ACCEPTED), each pre-stated with the evidence of the previous one; the lead decided (13:45, not Mirko) that the reported read-out is 'attached ends + W'.
- **E360** (ex E359): Matrix-NMS layer vs its E2E control with stale attached ends: INVALID by the symmetry stop rule, would be TIE. **E363** (faithful graphs: pair run re-dumped with the node head of B' by checkpoint surgery + the layer in front of Attach by an in-process wrapper; critic-caught: the first draft lacked the layer): all stop rules held (fixture bit for bit, node check 0.0), layer vs control +0.0142 / +0.0232 topo F1, dup_frac guard broken (+0.016 / +0.012) = TIE; rule version (c2 + hard step) 0.6975 / 0.6955, the layer recovers 46 % / 67 %; **THE STACK (layer + head) 0.6820 / 0.6854**, -0.009 vs E324 s PRIMARY with both rule steps. **E364** (M bundle part 1, CPU): the long-link cover drop CARRIES 0.027 topo F1 (dup_frac +0.07 without it), the pair mask admits 97.4 % of the true connectors (98.6 / 97.7 % of the reachable ones), the ends are the bigger limit.
"""
s = s[:old_start] + new + "\n" + s[old_end:]
a = "(6) a fixture check can be too weak: a stop rule on stored numbers (base block, curves, symmetry) is better;"
assert s.count(a) == 1
s = s.replace(a, a + " (7) `ledger.py note --result / --decision` REPLACES the cell (use `--notes` for an addendum); (8) the Matrix-NMS layer is not in `nodes.*` but in `QGraph.node_stage`: a surgery of a pair model with B' needs the layer wrapper; (9) the run.py CPU scorer jobs take 4 - 10 min: a whole stack check costs about an hour;")
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
