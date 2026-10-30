"""E260: remove the dead options of assemble() and the helper code behind them. One-off patch (idempotent check: refuses when a pattern is missing)."""
import ast

P = "training/pipeline/p16_infer.py"
src = open(P, encoding="utf-8").read()
lines = src.split("\n")


def span(name, cls=None):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and cls and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    return f.lineno, f.end_lineno
        if cls is None and isinstance(node, ast.FunctionDef) and node.name == name and node.col_offset == 0:
            return node.lineno, node.end_lineno
    raise SystemExit(f"{name} not found")


# 1. delete helper functions (bottom-up so the line numbers stay valid)
cuts = [span("glue_ends"), span("contract_junctions"), span("covered_fraction", "Union")]
for a, b in sorted(cuts, reverse=True):
    # also remove one blank line before a top-level function
    del lines[a - 1:b]
    while a - 2 >= 0 and lines[a - 2] == "" and a - 1 < len(lines) and lines[a - 1] == "":
        del lines[a - 1]
src = "\n".join(lines)


def rep(a, b):
    global src
    assert src.count(a) == 1, (src.count(a), a[:70])
    src = src.replace(a, b)


# 2. signature
rep("def assemble(r, b, tau, cfg, cov_r=1.0, vtx_r=1.5, pair=None, seg_r=1.5, seg_frac=1.0, clean=True, bubble_r=2.5, spur_len=5.0, contract=0.0, aux=False, glue_r=0.0, glue_lo=10.0, glue_hi=80.0, junc_thr=0.0, junc_edge=3.0, junc_free=8.0, junc_ang=100.0, fork_tau=0.0, fork_peak=0.6, fork_r=10.0, fork_vtx=0.0, fork_cov=0.0, reloc_thr=0.0,",
    "def assemble(r, b, tau, cfg, cov_r=1.0, vtx_r=1.5, pair=None, clean=True, bubble_r=2.5, spur_len=5.0, aux=False, junc_thr=0.0, junc_edge=3.0, junc_free=8.0, junc_ang=100.0, reloc_thr=0.0,")
# 3. fork-conditioned admission and fork-preserving union
i = src.index("    if fork_tau > 0 and \"junc\" in r:")
j = src.index("    idx = np.nonzero(alive)[0]\n    empty = dict(pts=[], edges=[], nodes=[])")
src = src[:i] + src[j:]
rep("            if dv < (fork_vtx if nearp[i] else vtx_r) and nv >= 0:", "            if dv < vtx_r and nv >= 0:")
rep("                if e is not None and de < (fork_cov if nearp[i] else cov_r):", "                if e is not None and de < cov_r:")
rep(" and not U.has_path(a, c) and not (seg_frac < 1.0 and U.covered_fraction(a, c, seg_r) >= seg_frac):", " and not U.has_path(a, c):")
rep("    if glue_r > 0 and E:\n        Vall, E = glue_ends(Vall, E, glue_r, glue_lo, glue_hi)\n", "")
rep("    if contract > 0 and E:\n        E, Vall = contract_junctions(Vall, E, contract)\n", "")
open(P, "w", encoding="utf-8").write(src)
print("patched; lines", len(src.split("\n")))
