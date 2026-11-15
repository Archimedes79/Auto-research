def sub(path, pairs):
    s = open(path, encoding="utf-8").read()
    for a, b, n in pairs:
        assert s.count(a) == n, (path, a, s.count(a))
        s = s.replace(a, b)
    open(path, "w", encoding="utf-8").write(s)

sub("experiments/E344_e2e-completion-plan/experiment.md", [
    ("E345 / E346 (D99) size the heads", "E345 and its B-side follow-up (D99) size the heads", 1),
])
s = open("experiments/E344_e2e-completion-plan/experiment.md", encoding="utf-8").read()
s = s.replace("card E347", "card E346 (the lane lead's `one-graph-code-ceiling`, which widens Q0 by the connector code)").replace("the Q0 card E347", "the Q0 card E346")
open("experiments/E344_e2e-completion-plan/experiment.md", "w", encoding="utf-8").write(s)
sub("experiments/E345_pruning-sensitivity-step0/experiment.md", [
    ("is a later card E346 (its own probe loop", "is a later card (its id is not taken yet; its own probe loop", 1),
    ("B's pyramid and B's heads (E346)", "B's pyramid and B's heads (the later B-side card)", 1),
])
sub("HANDOVER.md", [
    ("; B side = E346 later.", "; the B side is a later card.", 1),
    ("; Q0 card E347 (CPU, GT only) not written.", "; Q0 is the lead's card E346 (one-graph code ceiling), E347 (ends on the 2 m grid) beside it: both need a critic pass and code (mine).", 1),
    ("Q0 card E347; old-chain step 1", "E346 / E347 critic + code; old-chain step 1", 1),
])
print("ok")
