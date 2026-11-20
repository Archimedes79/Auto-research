"""Draws the architecture of the E389 route-association head (RouteHead) as an svg, FROM THE REAL MODULE: the parameter counts are read from an instance of `training/grid/jroute.py::RouteHead` and the cap of
the route table from `training/data/lane_route_data.py`, so the picture cannot drift from the code (a block that is renamed or removed fails the lookup). Few words in the boxes and large type (the report is
read on narrow screens); the numbers per block are printed as a markdown table by `--table`.

    python tools/draw_route_head.py <out.svg>          # run from bev_fusion_lane_center/ with the gcar python (torch); prints the module list it drew
    python tools/draw_route_head.py --table            # the parameter table (markdown) of the blocks, for the chapter text
    python tools/draw_route_head.py --selftest         # draws into a temporary file: the three data paths, the totals, the cap, the font size, no red

The three paths of the head (jroute.RouteHead.parts / forward): [1] the own features x of the token; [2] the learned cross-attention A of the token to the route keys (+ one none key, bias from the pair
features); [3] the sharp, parameter-free nearest-key attention B (a soft argmin of the distance, it only delivers the pair features of the nearest key; it needs the tokens AND the route keys); the three are
CONCATENATED (x, the attended values of A, the attention mass of A, g pooled by A, g pooled by B, the flag "a map is there") and one MLP gives the two logits (road, lane). The read-out threshold is NOT part of
the module (dashed box).
Colours as in the reports of 2026-11-18: blue = learned, green = given, amber = a fixed step without parameters, grey dashed = training only / outside the module. No red.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))                                    # bev_fusion_lane_center/
sys.path.insert(0, os.path.join(ROOT, "training", "grid"))
import jroute as J  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "training", "data"))
import lane_route_data as LRD  # noqa: E402                                   # MAX_SAMPLES, the cap of the route table

W, H = 780, 790
FS_TITLE, FS_TEXT, FS_NOTE = 20, 19, 17                                          # px at the viewBox width 780


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


def box(x, y, w, h, title, lines, fill, stroke, sw=1.8, dash=False):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"' + (' stroke-dasharray="7 5"' if dash else "") + "/>\n"
    s += f'<text x="{x + w / 2}" y="{y + 28}" font-size="{FS_TITLE}" font-weight="bold" text-anchor="middle" fill="#1d2330">{title}</text>\n'
    for i, t in enumerate(lines):
        s += f'<text x="{x + w / 2}" y="{y + 57 + 24 * i}" font-size="{FS_TEXT}" text-anchor="middle" fill="#1d2330">{t}</text>\n'
    return s


def arrow(x1, y1, x2, y2, dash=False):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#444" stroke-width="2"' + (' stroke-dasharray="5 4"' if dash else "") + ' marker-end="url(#ar)"/>\n'


def table_md():
    cf, cg = counts("full"), counts("geo")
    tok, geo, lnq, q, rkey, kv, bias, outm, none = need(cf, "tok", "geo", "ln_q", "q", "rkey", "kv", "bias", "out", "none_key")
    rows = [("Token-Projektion (`tok` -> 64, LayerNorm)", tok), ("Geometrie-MLP (eigene Merkmale)", geo), ("Anfrage q (LayerNorm + Linear)", lnq + q), ("Routen-Schlüssel (Eingabe -> 64, LayerNorm)", rkey),
            ("Schlüssel und Werte (64 -> 128)", kv), ("Bias-MLP der Paarmerkmale (8 -> 32 -> 4)", bias), ("None-Schlüssel (k, v, Bias; 4 Köpfe)", none), ("Ausgabe-MLP (181 -> 64 -> 2)", outm)]
    s = "| Block | Parameter |\n|---|---|\n" + "".join(f"| {n} | {v:,} |\n" for n, v in rows) + f"| **Summe** (mit `tok`) | **{cf['total']:,}** |\n| Summe nur Geometrie (ohne `tok`, Kontrolle zg / mg) | {cg['total']:,} |\n"
    assert sum(v for _, v in rows) == cf["total"], "the table rows do not add up to the module total"
    return s


def main(out):
    cf, cg = counts("full"), counts("geo")
    need(cf, "tok", "geo", "ln_q", "q", "rkey", "kv", "bias", "out", "none_key")
    BLUE, BS, GREEN, GS, AMB, AS, GREY, GRS = "#e9eef7", "#31507f", "#e3f1e6", "#2f6b3c", "#fff6dc", "#b07a00", "#f1f1f1", "#777"
    s = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">\n'
    s += '<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#444"/></marker></defs>\n'
    s += f'<rect x="0" y="0" width="{W}" height="{H}" fill="#fbfaf7"/>\n'
    s += f'<text x="{W / 2}" y="30" font-size="21" font-weight="bold" text-anchor="middle" fill="#1d2330">E389: der Kopf im Überblick</text>\n'
    # inputs (two lines at most)
    s += box(20, 52, 350, 84, "Token von B' (gecacht)", ["tok, Position, Richtung, Flags"], GREEN, GS)
    s += box(410, 52, 350, 84, "SD-Routen-Token", [f"bis zu {LRD.MAX_SAMPLES} Stützen, mit Fehler"], GREEN, GS)
    # the three paths: the tokens feed (1), (2), (3); the route feeds (2), (3)
    s += box(20, 214, 230, 112, "(1) Eigene Werte x", ["tok + Geometrie", "in Metern"], BLUE, BS)
    s += box(275, 214, 230, 112, "(2) Attention A", ["gelernt, 4 Köpfe", "+ None-Schlüssel"], BLUE, BS)
    s += box(530, 214, 230, 112, "(3) Attention B", ["fest, ohne Parameter", "Softmin über Abstand"], AMB, AS, 2.4)
    s += arrow(110, 136, 110, 214) + arrow(250, 136, 340, 214) + arrow(330, 136, 580, 214) + arrow(500, 136, 440, 214) + arrow(640, 136, 670, 214)
    # concat
    s += arrow(135, 326, 135, 372) + arrow(390, 326, 390, 372) + arrow(645, 326, 645, 372)
    s += box(20, 374, 740, 88, "Verketten (181 Werte)", ["x | Werte von A | Masse | g nach A | g nach B | Karte da"], BLUE, BS)
    s += arrow(390, 462, 390, 494)
    s += box(190, 496, 400, 88, "Ausgabe-MLP", ["zwei Logits: Fahrbahn | Spur"], BLUE, BS)
    s += box(620, 496, 140, 88, "nur Training", ["BCE, GT-Label"], GREY, GRS, 1.5, True)
    s += arrow(620, 540, 590, 540, True)
    s += arrow(390, 584, 390, 616)
    s += box(190, 618, 400, 88, "Auswertung", ["außerhalb des Kopfes: p >= 0.5"], AMB, AS, 2.0, True)
    s += f'<text x="{W / 2}" y="738" font-size="{FS_NOTE}" text-anchor="middle" fill="#1d2330">{cf["total"]:,} Parameter ({cg["total"]:,} nur Geometrie), keine Self-Attention.</text>\n'
    s += f'<text x="{W / 2}" y="762" font-size="{FS_NOTE}" text-anchor="middle" fill="#1d2330">Kopf ohne Karte (z): jede Route ungültig, nur der None-Schlüssel.</text>\n'
    s += "</svg>\n"
    open(out, "w", encoding="utf-8", newline="\n").write(s)
    print("drew", out, "modules:", {k: v for k, v in cf.items()}, "geo total", cg["total"])


def selftest():
    import tempfile
    cf, cg = counts("full"), counts("geo")
    assert cf["total"] == 33738 and cg["total"] == 21130, (cf["total"], cg["total"])           # pinned in grid/test_jroute.py
    assert sum(v for k, v in cf.items() if k != "total") == cf["total"], cf
    assert table_md().count("\n") == 12
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "head.svg")
        main(out)
        svg = open(out, encoding="utf-8").read()
    for path in ("(1) Eigene Werte x", "(2) Attention A", "(3) Attention B", "None-Schlüssel", "Verketten", "Ausgabe-MLP", "Auswertung", "außerhalb des Kopfes"):
        assert path in svg, f"the drawing lacks {path!r}"
    assert f"{cf['total']:,} Parameter" in svg and f"{cg['total']:,} nur Geometrie" in svg
    assert f"bis zu {LRD.MAX_SAMPLES} Stützen" in svg, "the route cap in the drawing is the data layer cap"
    assert len(re.findall(r"<line", svg)) >= 11, "the arrows are missing (the tokens feed the three paths, the route feeds (2) and (3))"
    sizes = [float(x) for x in re.findall(r'font-size="([0-9.]+)"', svg)]
    assert min(sizes) >= 17, "text must be >= 17 px at the viewBox width 780 (phones show it at about 0.45)"
    assert len(re.findall(r"<text", svg)) <= 26, "too many text items for a narrow screen"
    for h in set(re.findall(r"#([0-9a-fA-F]{6})", svg)):
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        assert not (r > 150 and g < 110 and b < 110), f"a red colour #{h}: no red blocks"
    assert svg.count("<rect") >= 10 and svg.rstrip().endswith("</svg>")
    try:                                                      # a renamed block must fail the lookup, not silently draw a stale picture
        need({"tok": 1}, "nope")
        raise SystemExit("need() accepted a missing block")
    except AssertionError:
        pass
    print("draw_route_head selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    elif sys.argv[1:] == ["--table"]:
        print(table_md())
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "temp", "route_head.svg"))
