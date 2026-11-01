def sub(p, a, b, cnt=1):
    t = open(p, encoding="utf-8").read()
    assert t.count(a) == cnt, (p, t.count(a), a[:60])
    open(p, "w", encoding="utf-8").write(t.replace(a, b))

P = "training/pipeline/p16_infer.py"
# guard at the top of assemble, remove the late one
sub(P, "    if dp_eps > 0:  # output stage (E257 / D20): drop interior vertices of straight runs, geometry kept within dp_eps; off by default; aux describes the unsimplified vertices\n        if aux:\n            raise ValueError(\"dp_eps and aux cannot be combined: the token map of aux refers to the unsimplified vertices\")\n        g = simplify_dp(g, dp_eps)\n",
    "    if dp_eps > 0:  # output stage (E257 / D20): drop interior vertices of straight runs, geometry kept within dp_eps; off by default (guard at the top: aux and dp_eps exclude each other)\n        g = simplify_dp(g, dp_eps)\n")
t = open(P, encoding="utf-8").read()
i = t.index("def assemble(r, b, tau, cfg, cov_r=1.0")
j = t.index('    """', i)
k = t.index('"""', j + 7) + 3  # end of the docstring
guard = "\n    if dp_eps < 0 or (dp_eps > 0 and aux):\n        raise ValueError(\"dp_eps must be >= 0 and cannot be combined with aux=True (the token map of aux refers to the unsimplified vertices)\")"
t = t[:k] + guard + t[k:]
open(P, "w", encoding="utf-8").write(t)
sub(P, "    cnt = Counter(i for nd in g[\"nodes\"] for i in nd)\n    pts, nodes = [], []\n    for q, nd in zip(g[\"pts\"], g[\"nodes\"]):\n        q = np.asarray(q, np.float64)\n        n = len(q)\n",
    "    assert g.get(\"nodes\") is not None and len(g[\"nodes\"]) == len(g[\"pts\"]), \"simplify_dp needs a node id list for every polyline\"\n    cnt = Counter(i for nd in g[\"nodes\"] for i in nd)\n    pts, nodes = [], []\n    for q, nd in zip(g[\"pts\"], g[\"nodes\"]):\n        q = np.asarray(q, np.float64)\n        n = len(q)\n        assert len(nd) == n, \"simplify_dp: node ids and points of a polyline differ in length\"\n")
sub(P, "def _dp_keep(q, eps):\n    \"\"\"indices kept by Douglas-Peucker (iterative) on the polyline q [n,2] with tolerance eps (m); the ends are always kept\"\"\"",
    "def _dp_keep(q, eps):\n    \"\"\"indices kept by Douglas-Peucker (iterative) on the polyline q [n,2] with tolerance eps (m); the ends are always kept.\n    Same rule as evaluation/sparse_lanes/sparse_metrics._dp_keep (the oracle count vert100_dp); keep the two equal.\"\"\"")

# tests
T = "evaluation/tests/test_dp_simplify.py"
sub(T, "if __name__ == \"__main__\":", '''def test_D7_a_forked_graph_from_graph_from_vertices_keeps_edges_ends_and_shared_ids():
    import p16_common as C

    V = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [30, 6], [40, 12.0]])
    E = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6)]  # a fork at vertex 2
    g = C.graph_from_vertices(V, E)
    s = PI.simplify_dp(g, 0.2)
    assert s["edges"] == list(g["edges"]) and len(s["pts"]) == len(g["pts"])
    shared = {i for i in {i for nd in g["nodes"] for i in nd} if sum(i in nd for nd in g["nodes"]) > 1}
    kept = {i for nd in s["nodes"] for i in nd}
    assert shared <= kept
    for old, new in zip(g["nodes"], s["nodes"]):
        assert new[0] == old[0] and new[-1] == old[-1]


def test_D8_assemble_refuses_dp_with_aux_and_negative_eps_before_any_work():
    for kw in (dict(dp_eps=0.2, aux=True), dict(dp_eps=-1.0)):
        try:
            PI.assemble(None, None, 0.7, None, **kw)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for {kw}")


def test_D9_inconsistent_input_is_refused():
    for g in (dict(pts=[np.zeros((3, 2))], edges=[], nodes=None), dict(pts=[np.zeros((3, 2)), np.zeros((3, 2))], edges=[], nodes=[[1, 2, 3]]),
              dict(pts=[np.zeros((4, 2))], edges=[], nodes=[[1, 2, 3]])):
        try:
            PI.simplify_dp(g, 0.2)
        except AssertionError:
            continue
        raise AssertionError("inconsistent graph accepted")


if __name__ == "__main__":''')
sub(T, "(6 spec tests", "(9 spec tests") if False else None

# wording
sub("PROBLEM.md", "- Vertex count (D20, 2026-10-30): the budget is quoted for the graph after the output stage `simplify_dp` (Douglas-Peucker 0.2 m, `assemble(dp_eps=0.2)`, registry names `*_dp`); every table also gives the node-id count of the unsimplified graph (14.8 - 15.5 per 100 m for the current systems, about 9.5 - 10 after the stage; E257). The stage is off by default; Q comparisons name which count they use.",
    "- Vertex count (D20, 2026-10-30): the vertex budget is quoted for the graph after the output stage `simplify_dp` (Douglas-Peucker 0.2 m, `assemble(dp_eps=0.2)`, registry names `*_dp`: champion 14.8 -> 9.5, fast_recall 15.0 -> 9.9 per 100 m, E257); tables of such systems give the node-id count of the unsimplified graph next to it. The stage is off by default, the route head and the map pipeline (`aux=True`) work on the unsimplified vertices, and S of a `*_dp` card is not comparable with S of other cards (it rewards the lower count); Q comparisons name which count they use.")
sub("evaluation/PROTOCOL.md", "- Vertex count (D20): the budget is quoted for the graph after the output stage `simplify_dp` (0.2 m, registry names `*_dp`, `assemble(dp_eps=0.2)`); tables also give the node-id count of the unsimplified graph; a Q comparison names which count it uses. The stage is off by default.",
    "- Vertex count (D20): the budget is quoted for the graph after the output stage `simplify_dp` (0.2 m, registry names `*_dp`, `assemble(dp_eps=0.2)`); tables of such systems also give the node-id count of the unsimplified graph; a Q comparison names which count it uses; S of a `*_dp` card is not comparable with S of other cards. The stage is off by default and excluded from the `aux=True` pipelines (route head, map): they keep the unsimplified vertices.")
sub("evaluation/README.md", "`test_dp_simplify.py` (6 spec tests", "`test_dp_simplify.py` (9 spec tests")
sub("tools/maintenance/run_all_tests.sh", 'run "output stage simplify_dp (6)"', 'run "output stage simplify_dp (9)"')
print("ok")
