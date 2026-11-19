"""Draws the architecture of the E389 route-association head (RouteHead) as an svg, FROM THE REAL MODULE: the parameter counts of every block are read
from an instance of `training/grid/jroute.py::RouteHead`, so the picture cannot drift from the code (a block that is renamed or removed fails the lookup).

    python tools/draw_route_head.py <out.svg>          # run from bev_fusion_lane_center/ with the gcar python (torch); prints the module list it drew
    python tools/draw_route_head.py --selftest         # draws into a temporary file: every block of the module is in the drawing, the parameter counts are the pinned ones

Colours as in the reports of 2026-11-18: blue = learned, green = frozen / given, amber outline = a named hard step or a read-out rule, grey dashed = training only. No red.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))                                    # bev_fusion_lane_center/
sys.path.insert(0, os.path.join(ROOT, "training", "grid"))
import jroute as J  # noqa: E402


def counts(inputs):
    m = J.RouteHead(inputs=inputs)
    c = {n: sum(p.numel() for p in ch.parameters()) for n, ch in m.named_children()}
    c["none_key"] = sum(p.numel() for n, p in m.named_parameters(recurse=False))
    c["total"] = sum(p.numel() for p in m.parameters())
    return c


def need(c, *names):
    for n in names:
        assert n in c, f"RouteHead has no block {n!r}: the drawing is out of date ({sorted(c)})"
    return [c[n] for n in names]


def box(x, y, w, h, title, lines, fill, stroke, sw=1.6, dash=False):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"' + (' stroke-dasharray="6 4"' if dash else "") + "/>\n"
    s += f'<text x="{x + w / 2}" y="{y + 17}" font-size="12" font-weight="bold" text-anchor="middle" fill="#1d2330">{title}</text>\n'
    for i, t in enumerate(lines):
        s += f'<text x="{x + w / 2}" y="{y + 32 + 14 * i}" font-size="10.5" text-anchor="middle" fill="#1d2330">{t}</text>\n'
    return s


def arrow(x1, y1, x2, y2, dash=False):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#444" stroke-width="1.5"' + (' stroke-dasharray="5 4"' if dash else "") + ' marker-end="url(#ar)"/>\n'


def main(out):
    cf, cg = counts("full"), counts("geo")
    tok, geo, lnq, q, rkey, kv, bias, outm = need(cf, "tok", "geo", "ln_q", "q", "rkey", "kv", "bias", "out")
    (none,) = need(cf, "none_key")
    BLUE, BS, GREEN, GS, AMB, AS, GREY, GRS = "#e9eef7", "#31507f", "#e3f1e6", "#2f6b3c", "#fff6dc", "#b07a00", "#f1f1f1", "#777"
    s = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 700" font-family="Segoe UI, Arial, sans-serif">\n'
    s += '<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#444"/></marker></defs>\n'
    s += '<rect x="0" y="0" width="780" height="700" fill="#fbfaf7"/>\n'
    s += '<text x="390" y="22" font-size="15" font-weight="bold" text-anchor="middle" fill="#1d2330">E389: the route-association head on the cached tokens of the champion</text>\n'
    s += '<text x="390" y="38" font-size="10.5" text-anchor="middle" fill="#5b6475">blue = learned | green = given (frozen tokens, SD route) | amber outline = a named read-out step | grey dashed = training only | parameters read from RouteHead</text>\n'
    # left column: the tokens
    s += box(30, 56, 330, 68, "Tokens of the frozen lane head B' (cached, K = 512)", ["node token tok (64-d), position, heading,", "existence logit e, 8 flag logits per token", "no gradient reaches B' (the head is not in stage 2)"], GREEN, GS)
    s += arrow(195, 124, 195, 142)
    s += box(30, 142, 330, 84, "Own features of every token", [f"tok -> Linear + LayerNorm (64): {tok:,} parameters", f"geometry MLP (position in m, heading, sigmoid(e), flags): {geo:,}", f"concat x (96-d) -> LayerNorm + Linear q (4 heads x 16): {lnq + q:,}", "arm z and the geometry-only control drop tok"], BLUE, BS)
    # right column: the route
    s += box(420, 56, 330, 68, "SD route tokens (ADASIS-like map, simulated)", ["up to 64 samples every 2 m, +-50 m: position, direction,", "lane count, valid; the lane choice is lost;", "map errors: shift 1 - 2 m, yaw, sparsified polyline"], GREEN, GS)
    s += arrow(585, 124, 585, 142)
    s += box(420, 142, 330, 84, "Route keys", [f"[position, direction, one-hot lane count] -> Linear + LayerNorm: {rkey:,}", f"keys and values (Linear 64 -> 128): {kv:,}", f"one learned none key per head (k, v, bias): {none}", "a frame without a map: the none key only (arm z)"], BLUE, BS)
    # pair features
    s += box(215, 252, 350, 62, "Pair features (token i, route key j), in metres", ["turn cos / sin, offset along / across the token heading and the", f"route direction, distance, lane count: 8 features; bias MLP (8 -> 32 -> 4): {bias}"], BLUE, BS)
    s += arrow(195, 226, 260, 252)
    s += arrow(585, 226, 520, 252)
    # attention A and B
    s += box(30, 344, 340, 88, "Attention A (learned, 4 heads)", ["softmax over {valid route keys} + {none key}:", "logit = q.k / 4 + bias(pair features)", "-> attended values, attention mass on real keys per head,", "pair features pooled by A"], BLUE, BS)
    s += box(410, 344, 340, 88, "Attention B (sharp, fixed, no parameter)", ["soft nearest key: softmax of -distance / 0.3 m over the valid keys", "(no hard selection) -> the metre-scale features of the", "nearest key: offset across the route = the registration"], AMB, AS, 2.2)
    s += arrow(195, 314, 200, 344)
    s += arrow(390, 314, 560, 344)
    s += arrow(195, 226, 120, 344)
    # output
    s += arrow(200, 432, 330, 470)
    s += arrow(580, 432, 450, 470)
    s += box(190, 472, 400, 74, "Output MLP: [x, attended, mass, pooled A, pooled B, any-map flag] (181)", [f"-> 64 -> 2 logits: {outm:,} parameters", "road: the token lies on the carriageway of the route", "lane: it lies on the lane the vehicle will drive"], BLUE, BS)
    s += arrow(390, 546, 390, 570)
    s += box(190, 572, 400, 62, "Read-out on the delivered tokens of the champion", ["p_route >= 0.5 per delivered token (the E373 head at the D115 decode)", "the scored unit is the covered GT node; no token is dropped by this head"], AMB, AS, 2.2)
    # training only
    s += box(610, 472, 150, 74, "Training only", ["BCE of both logits on the", "labelled tokens; GT route", "labels (tier 1), never an input"], GREY, GRS, 1.3, True)
    s += arrow(610, 509, 590, 509, True)
    s += f'<text x="390" y="662" font-size="11" text-anchor="middle" fill="#1d2330">{cf["total"]:,} parameters with tok (full), {cg["total"]:,} geometry-only; no self-attention among the tokens: a token reads only its own row and the route</text>\n'
    s += '<text x="390" y="680" font-size="11" text-anchor="middle" fill="#5b6475">Reference arm z: the same head with every route key invalid (the matched no-map head). Source: training/grid/jroute.py</text>\n'
    s += "</svg>\n"
    open(out, "w", encoding="utf-8", newline="\n").write(s)
    print("drew", out, "modules:", {k: v for k, v in cf.items()}, "geo total", cg["total"])


def selftest():
    """the module list of RouteHead is what the picture shows: the pinned totals, every learned block named with its count, no red"""
    import tempfile
    cf, cg = counts("full"), counts("geo")
    assert cf["total"] == 33738 and cg["total"] == 21130, (cf["total"], cg["total"])           # pinned in grid/test_jroute.py
    assert sum(v for k, v in cf.items() if k != "total") == cf["total"], cf
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "head.svg")
        main(out)
        svg = open(out, encoding="utf-8").read()
    for name in ("tok", "geo", "rkey", "kv", "bias", "out"):
        assert f"{cf[name]:,}" in svg, f"the parameter count of the block {name!r} ({cf[name]:,}) is not in the drawing"
    assert f"{cf['total']:,} parameters" in svg and f"{cg['total']:,} geometry-only" in svg
    import re
    for h in set(re.findall(r"#([0-9a-fA-F]{6})", svg)):
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        assert not (r > 150 and g < 110 and b < 110), f"a red colour #{h}: no red blocks"
    assert svg.count("<rect") >= 10 and svg.rstrip().endswith("</svg>")
    # a renamed block must fail the lookup, not silently draw a stale picture
    try:
        need({"tok": 1}, "nope")
        raise SystemExit("need() accepted a missing block")
    except AssertionError:
        pass
    print("draw_route_head selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "temp", "route_head.svg"))
