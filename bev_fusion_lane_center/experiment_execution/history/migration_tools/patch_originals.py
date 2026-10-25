"""Replace the duplicated metric functions of p16_common.py, dense_head.py and field_tools.py by imports from evaluation/ (function sources are cut out by ast ranges).
Backups go to _to_delete/scripts_junk/<name>.bak_before_eval_move.  Run: _run.sh patch_originals.py [--go]"""
import ast
import re
import shutil
import sys

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
SC = f"{R}/scripts"
GO = "--go" in sys.argv
JOBS = {
    "p16_common.py": (["vert100", "composite", "card", "frontier_scalars", "graph_from_vertices"], ["DP_ORACLE", "KEEP"],
                      "from sparse_metrics import DP_ORACLE, KEEP, card, composite, frontier_scalars, graph_from_vertices, vert100  # noqa: E402,F401  (moved to evaluation/sparse_lanes)"),
    "dense_head.py": (["thin_batch", "skeleton_points", "geom_counts", "eval_probs", "load_gt_points"], [],
                      "from grid_eval import eval_probs, geom_counts, load_gt_points, skeleton_points, thin_batch  # noqa: E402,F401  (moved to evaluation/lane_grid)"),
    "field_tools.py": (["thin_batch", "skeletons", "est_eval"], [],
                       "from grid_eval import est_eval, skeletons, thin_batch  # noqa: E402,F401  (moved to evaluation/lane_grid)"),
}
for fn, (funcs, assigns, imp) in JOBS.items():
    p = f"{SC}/{fn}"
    src = open(p, encoding="utf-8").read()
    lines = src.splitlines()
    tree = ast.parse(src)
    cut = []
    for nd in tree.body:
        if isinstance(nd, ast.FunctionDef) and nd.name in funcs:
            st = (nd.decorator_list[0].lineno if nd.decorator_list else nd.lineno) - 1
            cut.append((st, nd.end_lineno))
        elif isinstance(nd, ast.Assign) and any(isinstance(t, ast.Name) and t.id in assigns for t in nd.targets):
            cut.append((nd.lineno - 1, nd.end_lineno))
    assert len(cut) == len(funcs) + len(assigns), (fn, len(cut))
    cut.sort()
    first = cut[0][0]
    out, i = [], 0
    drop = set()
    for a, b in cut:
        drop.update(range(a, b))
    for i, l in enumerate(lines):
        if i == first:
            out.append(imp)
        if i in drop:
            continue
        out.append(l)
    new = "\n".join(out) + "\n"
    new = re.sub(r"\n{4,}", "\n\n\n", new)
    ast.parse(new)
    print(f"{fn}: {len(lines)} -> {len(new.splitlines())} lines, {len(cut)} definitions replaced")
    if GO:
        shutil.copy2(p, f"{R}/_to_delete/scripts_junk/{fn}.bak_before_eval_move")
        open(p, "w", encoding="utf-8", newline="\n").write(new)
