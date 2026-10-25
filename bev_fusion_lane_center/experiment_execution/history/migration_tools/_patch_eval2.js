const fs=require("fs");let s=fs.readFileSync("build_evaluation.py","utf8");
s=s.replace("'''\nbody = head + ", `

def card_any(preds, gts, with_route=False):
    """card() for graphs of any producer: the vertex count comes from the node ids when every graph has them (learned builders), otherwise from vert100_dp
    (rule graphs of a dense head carry no vertex ids; card() would count every polyline point as a vertex and S would not be comparable)."""
    m = card(preds, gts, with_route)
    if not all(g.get("nodes") is not None for g in preds):
        m["vert100"] = vert100_dp(preds)
        m["verts_per_frame"] = float("nan")
        m["S"], m["J"], m["Sp"] = composite(m)
    return m
'''
body = head + `);
fs.writeFileSync("build_evaluation.py",s);
