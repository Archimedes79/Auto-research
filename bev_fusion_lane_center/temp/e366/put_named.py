p = "experiments/E366_e365-decode-selected-on-train/experiment.md"
s = open(p, encoding="utf-8").read()
a = "**Caveats:** (1) the cell was selected"
assert s.count(a) == 1
add = ("**Named hard steps of this candidate (for the report, D88 / D113: hard selection only on the output and named):** the top-K 512 budget of B' tokens by existence (an index choice, no gradient through it), the allowed-pair gate of `jtok.pair_geometry` (a mask on the cached inputs), and the decode thresholds on the output (a link where p >= 0.1, a node where p_none < 0.6, the none gate off; chosen on the TRAIN sample); B' is frozen (staged: 'the loss reaches B' only with S4). No keypoint peaks, snap, attachment, wiring rule or close-node step. "
       "**Topology up, junction detection down, said plainly:** the okayish gate has no junction condition; the junction F1 of this candidate (0.209 / 0.204) is BELOW plain B's 0.272, while topo F1, F1@1.0 and lanes found are above B; the report must say both. ")
s = s.replace(a, add + a)
open(p, "w", encoding="utf-8").write(s)
print("ok")
