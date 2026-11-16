import io
p = "experiments/E344_e2e-completion-plan/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
marker = "### 4b. Variant B0"
assert s.count(marker) == 1
note = ("**E346 result (2026-11-15, `experiments/E346_one-graph-code-ceiling/experiment.md` section 5; by hand: GT replay, 873 VAL frames, deterministic, labelled with an open kill-line item: `test_qtargets.py` red at HEAD):** B's node + link code on the 2 m grid loses 0.019 topo F1 / 0.015 lanes found / 0.004 F1@1.0 against the undecoded lane GT (rule: at most 0.03 on all three: PASS), the 1 m grid 0.017 / 0.010 / 0.003 and gains only 0.0024 over 2 m (no reason for the finer grid); the chain code (a node in every cell a connector crosses) is REJECTED on both grids: 59.7 % (2 m) / 57.3 % (1 m) of the GT connectors have a collision cell against the bar of 5 %, 36 % / 33 % by crossing alone. So the one-graph head can target B's code as it is (nodes on the 2 m grid, short and long links); the largest loss of the code is junction F1 (-0.046 at 2 m, -0.033 at 1 m, not in the rule) and the 5.5 % of the GT connector length that comes back through no link. It says nothing about learning the links (E347 starts on the ends).\n\n")
s = s.replace(marker, note + marker)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
