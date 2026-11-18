p = "bev_fusion_lane_center/evaluation/jtoken_eval.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:60], s.count(a))
    s = s.replace(a, b)
rep("import qgraph_readout as QR  # noqa: E402\n", "import qgraph_readout as QR  # noqa: E402\nfrom pmap import pmap  # noqa: E402\n")
old_start = "    rows = []\n    for lt, ltl, kt, gate in SWEEP_GRIDS[a.grid]:\n        fr = [decode_variant"
assert s.count(old_start) == 1
i = s.index(old_start)
j = s.index("    JV.write_json(path, _jsonable(dict(run=a.run, model=\"model.pt\"", i)
body = s[i:j]
new = body.replace("    rows = []\n    for lt, ltl, kt, gate in SWEEP_GRIDS[a.grid]:\n", "\n    def cell_row(cell):\n        lt, ltl, kt, gate = cell\n")
# the loop body had 8-space indent already; close it with return and print in order
new = new.replace("        rows.append(row)\n        print(f\"link", "        return row\n\n    rows = pmap(cell_row, SWEEP_GRIDS[a.grid], getattr(a, \"jobs\", 1))                          # --jobs N: the cells are independent, N forked processes, the same numbers in the same order\n    for row in rows:\n        lt, ltl, kt, gate = row[\"link_thr\"], row[\"link_thr_long\"], row[\"keep_thr\"], row[\"none_gate\"]\n        print(f\"link")
s = s[:i] + new.lstrip("\n") .replace("def cell_row", "    def cell_row", 1) + s[j:] if False else s[:i] + new + s[j:]
rep("            s.add_argument(\"--grid\", default=\"perclass\", choices=tuple(SWEEP_GRIDS), help=\"perclass (E375, the only grid): link short x link long x keep, none gate off (36 cells)\")\n",
    "            s.add_argument(\"--grid\", default=\"perclass\", choices=tuple(SWEEP_GRIDS), help=\"perclass (E375, the only grid): link short x link long x keep, none gate off (36 cells)\")\n            s.add_argument(\"--jobs\", type=int, default=1, help=\"N forked processes score the cells (the same numbers in the same order; about 2.5x at 3); default 1 = the loop of before\")\n")
open(p, "w", encoding="utf-8", newline="\n").write(s)
