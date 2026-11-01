"""Architecture drawing and the code it stands for, kept in step: a JSON spec -> SVG + file map, and a check against the code.

    python archdraw.py render <spec.json>      # writes <spec>.svg and <spec>.md next to the spec
    python archdraw.py check <spec.json>       # exit 1 when a box points at nothing, code is in no box, or svg / md are stale
    python archdraw.py --selftest

The spec (one per system, next to its code; a minimal example is in `--selftest`):
  title, subtitle, root (folder the code paths are relative to, relative to the spec),
  kinds {name: {label, fill, stroke, class, dashed}}  (omitted: the defaults of a neural pipeline; class = learned | fixed | frozen | rule | other),
  nodes [{id, label ("\\n" breaks), kind, col, row, w, h (cells spanned, default 1), hpx (a thinner box, e.g. a gate band), code ["path", "path::Symbol", "dir/"], sub, note, why, replace_with, catch}],
  edges [{from, to, label, dashed, bold, both (arrowheads at both ends), via [[col, row], ...]}]  (via: waypoints in grid units; integers are the gaps between boxes: a routed arrow), groups [{label, nodes [ids], caption (text in the side column; the spec's extra_w pixels of width hold it)}],
  main_path {from, to}  (the stages between an input and an output: counted by class, drawn bold),
  covers [{dir, glob, recursive}] + not_drawn {path: reason}  (every file found must be in a box's code or in not_drawn).
`check` is what keeps drawing and code honest in both directions: box -> code (path and symbol exist) and code -> box (no file without a box
or a stated reason); the svg / md are generated, so a drawing cannot drift from its spec.
"""
import argparse
import fnmatch
import json
import os
import re
import sys
import tempfile
from html import escape

COLW, ROWH, BOXH, MARGIN, TITLE_H, LEGEND_H = 200, 100, 62, 24, 78, 64
FONT = "Segoe UI, Helvetica, Arial, sans-serif"
DEFAULT_KINDS = {
    "input": {"label": "input", "fill": "#fef9c3", "stroke": "#ca8a04", "class": "other"},
    "learned": {"label": "learned network (gradient flows)", "fill": "#dbeafe", "stroke": "#2563eb", "class": "learned"},
    "fixed": {"label": "fixed math, differentiable", "fill": "#ccfbf1", "stroke": "#0d9488", "class": "fixed"},
    "frozen": {"label": "frozen / cached (no gradient)", "fill": "#e5e7eb", "stroke": "#6b7280", "class": "frozen"},
    "rule": {"label": "rule step (no gradient)", "fill": "#fee2e2", "stroke": "#dc2626", "class": "rule", "dashed": True},
    "output": {"label": "delivered output", "fill": "#dcfce7", "stroke": "#16a34a", "class": "other"},
    "train": {"label": "training only (data, loss)", "fill": "#f3e8ff", "stroke": "#7c3aed", "class": "other"},
}
CLASS_WORD = {"learned": "learned", "fixed": "fixed-differentiable", "frozen": "frozen", "rule": "rule step"}


def load(path):
    global COLW, ROWH
    spec = json.load(open(path, encoding="utf-8"))
    COLW, ROWH = spec.get("colw", 200), spec.get("rowh", 100)
    spec.setdefault("kinds", DEFAULT_KINDS)
    spec.setdefault("edges", [])
    spec.setdefault("groups", [])
    spec["_dir"] = os.path.dirname(os.path.abspath(path))
    spec["_root"] = os.path.normpath(os.path.join(spec["_dir"], spec.get("root", ".")))
    spec["_by_id"] = {n["id"]: n for n in spec["nodes"]}
    return spec


def span(n):
    return n.get("w", 1), n.get("h", 1)


def box(n):
    """x, y, w, h of a node in svg pixels."""
    w, h = span(n)
    full = (h - 1) * ROWH + BOXH
    hpx = min(n.get("hpx", full), full)                            # a thinner band (a gate between layers), centred in its cell
    return (MARGIN + n["col"] * COLW + 20, TITLE_H + n["row"] * ROWH + 19 + (full - hpx) / 2, w * COLW - 40, hpx)


# ------------------------------------------------------------------------------------------------ the main path
def ends(v):
    return v if isinstance(v, list) else [v]


def main_path(spec):
    """ids of the nodes on a path from main_path.from to main_path.to (each an id or a list of ids: reachable from a start AND reaching an end),
    in a stable order."""
    mp = spec.get("main_path")
    if not mp:
        return []
    out, back = {}, {}
    for e in spec["edges"]:
        out.setdefault(e["from"], []).append(e["to"])
        back.setdefault(e["to"], []).append(e["from"])

    def reach(start, adj):
        seen, todo = set(start), list(start)
        while todo:
            for m in adj.get(todo.pop(), []):
                if m not in seen:
                    seen.add(m)
                    todo.append(m)
        return seen
    ids = reach(ends(mp["from"]), out) & reach(ends(mp["to"]), back)
    depth = {}
    for i in sorted(ids, key=lambda i: (spec["_by_id"][i]["col"], spec["_by_id"][i]["row"])):         # longest path from the start, cycles ignored
        depth[i] = 0
    for _ in range(len(ids)):
        for e in spec["edges"]:
            if e["from"] in ids and e["to"] in ids and depth[e["to"]] < depth[e["from"]] + 1 and depth[e["from"]] < len(ids):
                depth[e["to"]] = depth[e["from"]] + 1
    return sorted(ids, key=lambda i: (depth[i], spec["_by_id"][i]["row"], spec["_by_id"][i]["col"]))


def analysis(spec):
    """counts of the main path by class and the ids of its rule / frozen stages; None without a main_path."""
    ids = main_path(spec)
    if not ids:
        return None
    cls = {i: spec["kinds"][spec["_by_id"][i]["kind"]].get("class", "other") for i in ids}
    count = {c: sum(1 for v in cls.values() if v == c) for c in ("learned", "fixed", "frozen", "rule", "other")}
    return {"ids": ids, "class": cls, "count": count, "stages": len(ids) - count["other"]}


def badge(spec):
    a = analysis(spec)
    if not a or not a["stages"]:
        return ""
    c = a["count"]
    return (f"main path (bold): {a['stages']} stages = {c['learned']} learned, {c['fixed']} fixed-differentiable, "
            f"{c['frozen']} frozen, {c['rule']} rule step{'s' if c['rule'] != 1 else ''}")


# ------------------------------------------------------------------------------------------------ svg
def wrap(text, width_chars):
    out = []
    for part in text.split("\n"):
        line = ""
        for word in part.split(" "):
            if line and len(line) + 1 + len(word) > width_chars:
                out.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        out.append(line)
    return out


def sub_text(n):
    if "sub" in n:
        return n["sub"]
    if n.get("code"):
        first = n["code"][0]
        base = first.rstrip("/").split("/")[-1] + ("/" if first.endswith("/") else "")
        return base + (f" +{len(n['code']) - 1}" if len(n["code"]) > 1 else "")
    return ""


def side_point(a, p):
    """the middle of the side of node a that faces the pixel point p."""
    x, y, w, h = box(a)
    if p[1] > y + h:
        return x + w / 2, y + h
    if p[1] < y:
        return x + w / 2, y
    return (x + w, y + h / 2) if p[0] > x + w else (x, y + h / 2)


def routed_path(a, b, via):
    """an arrow through waypoints (grid units: x = columns, y = rows; whole numbers lie in the gaps between boxes), corners rounded."""
    wp = [(MARGIN + x * COLW, TITLE_H + y * ROWH) for x, y in via]
    pts = [side_point(a, wp[0])] + wp + [side_point(b, wp[-1])]
    d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    for i in range(1, len(pts) - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
        l0, l1 = max(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, 1e-6), max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1e-6)
        r = min(10.0, l0 / 2, l1 / 2)
        d += f" L{x1 - (x1 - x0) / l0 * r:.1f},{y1 - (y1 - y0) / l0 * r:.1f} Q{x1:.1f},{y1:.1f} {x1 + (x2 - x1) / l1 * r:.1f},{y1 + (y2 - y1) / l1 * r:.1f}"
    d += f" L{pts[-1][0]:.1f},{pts[-1][1]:.1f}"
    k = max(range(len(pts) - 1), key=lambda i: (pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2)
    return d, ((pts[k][0] + pts[k + 1][0]) / 2, (pts[k][1] + pts[k + 1][1]) / 2), True, 1e9


def edge_path(a, b, via=None):
    if via:
        return routed_path(a, b, via)
    ax, ay, aw, ah = box(a)
    bx, by, bw, bh = box(b)
    if bx >= ax + aw:                                              # target to the right
        x1, y1, x2, y2, vertical = ax + aw, ay + ah / 2, bx, by + bh / 2, False
    elif bx + bw <= ax:                                            # target to the left
        x1, y1, x2, y2, vertical = ax, ay + ah / 2, bx + bw, by + bh / 2, False
    else:                                                          # same columns: vertical
        cx = (max(ax, bx) + min(ax + aw, bx + bw)) / 2
        down = by > ay
        x1, y1, x2, y2, vertical = cx, (ay + ah if down else ay), cx, (by if down else by + bh), True
    if vertical:
        d = (y2 - y1) / 2
        return f"M{x1:.1f},{y1:.1f} C{x1:.1f},{y1 + d:.1f} {x2:.1f},{y2 - d:.1f} {x2:.1f},{y2:.1f}", ((x1 + x2) / 2, (y1 + y2) / 2), True, abs(y2 - y1)
    d = max(18.0, abs(x2 - x1) / 2) * (1 if x2 > x1 else -1)
    return f"M{x1:.1f},{y1:.1f} C{x1 + d:.1f},{y1:.1f} {x2 - d:.1f},{y2:.1f} {x2:.1f},{y2:.1f}", ((x1 + x2) / 2, (y1 + y2) / 2), False, abs(x2 - x1)


def render_svg(spec):
    cols = max(n["col"] + span(n)[0] for n in spec["nodes"])
    rows = max(n["row"] + span(n)[1] for n in spec["nodes"])
    extra = spec.get("extra_w", 0)
    width, height = 2 * MARGIN + cols * COLW + extra, TITLE_H + rows * ROWH + LEGEND_H + 8
    on_path = set(main_path(spec))
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="{FONT}">',
         f'<title>{escape(spec["title"])}</title>',
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         '<path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker></defs>',
         f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
         f'<text x="{MARGIN}" y="30" font-size="19" font-weight="700" fill="#0f172a">{escape(spec["title"])}</text>',
         f'<text x="{MARGIN}" y="50" font-size="12" fill="#475569">{escape(spec.get("subtitle", ""))}</text>']
    if badge(spec):
        o.append(f'<text x="{MARGIN}" y="67" font-size="12" font-weight="600" fill="#b91c1c">{escape(badge(spec))}</text>')
    glabels = []
    for g in spec["groups"]:                                       # frames of real groups, behind everything
        xs = [box(spec["_by_id"][i]) for i in g["nodes"]]
        x0, y0 = min(b[0] for b in xs) - 12, min(b[1] for b in xs) - 24
        xn, y1 = max(b[0] + b[2] for b in xs) + 12, max(b[1] + b[3] for b in xs) + 12
        x1 = xn + extra - 12 if g.get("caption") else xn
        o.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-dasharray="4 3"/>')
        glabels.append(f'<text x="{x0 + 10:.1f}" y="{y0 + 15:.1f}" font-size="11" font-weight="600" fill="#64748b" stroke="#f8fafc" stroke-width="4" '
                       f'paint-order="stroke">{escape(g["label"])}</text>')
        if g.get("caption"):
            ty = y0 + 32
            for ln in wrap(g["caption"], int((x1 - xn - 24) / 5.6)):
                o.append(f'<text x="{xn + 12:.1f}" y="{ty:.1f}" font-size="11" fill="#475569">{escape(ln)}</text>')
                ty += 15
    labels = []
    for e in spec["edges"]:
        d, (mx, my), vertical, free = edge_path(spec["_by_id"][e["from"]], spec["_by_id"][e["to"]], e.get("via"))
        bold = e.get("bold") or (e["from"] in on_path and e["to"] in on_path)
        dash = ' stroke-dasharray="5 4"' if e.get("dashed") else ""
        start = ' marker-start="url(#ah)"' if e.get("both") else ""
        o.append(f'<path d="{d}" fill="none" stroke="{"#334155" if bold else "#94a3b8"}" stroke-width="{2.6 if bold else 1.4}"{dash}{start} marker-end="url(#ah)"/>')
        fits = free >= 90 if vertical else len(e.get("label", "")) * 5.4 <= free - 6
        if e.get("label") and fits and mx + 6 + len(e["label"]) * 5.4 < width - 4:       # a label that does not fit is only in the file map
            labels.append(f'<text x="{mx + (6 if vertical else 0):.1f}" y="{my + (3 if vertical else -5):.1f}" font-size="10" text-anchor="{"start" if vertical else "middle"}" '
                          f'fill="#334155" stroke="#ffffff" stroke-width="3" paint-order="stroke">{escape(e["label"])}</text>')
    for n in spec["nodes"]:
        x, y, w, h = box(n)
        k = spec["kinds"][n["kind"]]
        dash = ' stroke-dasharray="6 3"' if k.get("dashed") else ""
        o.append(f'<g><title>{escape(n["id"] + ": " + "; ".join(n.get("code", [])))}</title>'
                 f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{k["fill"]}" stroke="{k["stroke"]}" stroke-width="{2.2 if n["id"] in on_path else 1.4}"{dash}/>')
        lines = wrap(n["label"], int((w - 12) / 6.6))
        sub = sub_text(n)
        total = len(lines) * 15 + (13 if sub else 0)
        ty = y + (h - total) / 2 + 12
        for ln in lines:
            o.append(f'<text x="{x + w / 2}" y="{ty:.1f}" font-size="12.5" font-weight="600" text-anchor="middle" fill="#0f172a">{escape(ln)}</text>')
            ty += 15
        if sub:
            o.append(f'<text x="{x + w / 2}" y="{ty - 1:.1f}" font-size="10" text-anchor="middle" fill="#475569" font-family="Consolas, Menlo, monospace">'
                     f'{escape(sub[:int((w - 8) / 6.1)])}</text>')
        o.append("</g>")
    o += glabels + labels                                           # titles and labels above the arrows
    lx, ly = MARGIN, height - LEGEND_H + 14
    o.append(f'<text x="{lx}" y="{ly}" font-size="11" font-weight="600" fill="#64748b">legend (bold arrow = main path or the information flow; dashed arrow = optional or no data flow through it)</text>')
    used = [k for k in spec["kinds"] if any(n["kind"] == k for n in spec["nodes"])]
    cx, cy = lx, ly + 12
    for k in used:
        kd = spec["kinds"][k]
        dash = ' stroke-dasharray="4 2"' if kd.get("dashed") else ""
        step = 26 + 6.2 * len(kd["label"]) + 14
        if cx + step > width - MARGIN:
            cx, cy = lx, cy + 18
        o.append(f'<rect x="{cx}" y="{cy}" width="18" height="12" rx="3" fill="{kd["fill"]}" stroke="{kd["stroke"]}" stroke-width="1.4"{dash}/>'
                 f'<text x="{cx + 24}" y="{cy + 10}" font-size="11" fill="#334155">{escape(kd["label"])}</text>')
        cx += step
    o.append("</svg>")
    return "\n".join(o) + "\n"


# ------------------------------------------------------------------------------------------------ markdown
def cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_md(spec):
    stem = spec["_stem"]
    o = [f"<!-- generated by autoresearch/tools/archdraw.py from {stem}.json: edit the JSON, then `archdraw.py render`; `archdraw.py check` fails when this file is stale -->",
         f"# {spec['title']}", "", f"![{spec['title']}]({stem}.svg)", "", spec.get("subtitle", ""), ""]
    o += ["## File map: every box and the code it stands for", "", "| box | kind | code | note |", "|---|---|---|---|"]
    for n in sorted(spec["nodes"], key=lambda n: (n["col"], n["row"])):
        o.append(f"| {cell(n['label'])} | {spec['kinds'][n['kind']]['label']} | {cell(', '.join('`' + c + '`' for c in n.get('code', [])))} | {cell(n.get('note', ''))} |")
    labelled = [e for e in spec["edges"] if e.get("label")]
    if labelled:
        o += ["", "## Arrows", "", "| from | to | carries |", "|---|---|---|"]
        o += [f"| {cell(spec['_by_id'][e['from']]['label'])} | {cell(spec['_by_id'][e['to']]['label'])} | {cell(e['label'])}{' (no data flow in training)' if e.get('dashed') else ''} |" for e in labelled]
    a = analysis(spec)
    if a:
        o += ["", "## Main path from input to output", "", f"{badge(spec)}.", "",
              "| stage | class | code |", "|---|---|---|"]
        for i in a["ids"]:
            n = spec["_by_id"][i]
            if a["class"][i] != "other":
                o.append(f"| {cell(n['label'])} | {CLASS_WORD[a['class'][i]]} | {cell(', '.join('`' + c + '`' for c in n.get('code', [])))} |")
        gaps = [i for i in a["ids"] if a["class"][i] in ("rule", "frozen")]
        if gaps:
            o += ["", "## Where the path is not end to end, and what could stand in for it", "",
                  "| step | class | what it does and what it carries | building block that emulates it differentiably | catch |", "|---|---|---|---|---|"]
            for i in gaps:
                n = spec["_by_id"][i]
                o.append(f"| {cell(n['label'])} | {CLASS_WORD[a['class'][i]]} | {cell(n.get('why', ''))} | {cell(n.get('replace_with', ''))} | {cell(n.get('catch', ''))} |")
    off = [n for n in spec["nodes"] if n["id"] not in (a["ids"] if a else []) and spec["kinds"][n["kind"]].get("class") == "rule"]
    if off:
        o += ["", "## Rule steps outside the main path", "", "| step | what it does | catch |", "|---|---|---|"]
        o += [f"| {cell(n['label'])} | {cell(n.get('why', ''))} | {cell(n.get('catch', ''))} |" for n in off]
    return "\n".join(o) + "\n"


# ------------------------------------------------------------------------------------------------ check
def has_symbol(text, name):
    for part in name.split("."):
        if not re.search(rf"(?m)^\s*(?:async\s+)?(?:def|class)\s+{re.escape(part)}\b|^\s*{re.escape(part)}\s*(?::[^=\n]+)?=", text):
            return False
    return True


def covered_files(spec):
    found = []
    for c in spec.get("covers", []):
        base = os.path.join(spec["_root"], c["dir"])
        for dirpath, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in c.get("exclude_dirs", ["__pycache__"]))
            if not c.get("recursive", True):
                dirs[:] = []
            for f in sorted(files):
                if fnmatch.fnmatch(f, c.get("glob", "*.py")):
                    found.append(os.path.relpath(os.path.join(dirpath, f), spec["_root"]).replace(os.sep, "/"))
    return found


def problems(spec):
    errs = []
    ids = [n["id"] for n in spec["nodes"]]
    errs += [f"duplicate node id {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    cells = {}
    for n in spec["nodes"]:
        if n["kind"] not in spec["kinds"]:
            errs.append(f"{n['id']}: unknown kind {n['kind']}")
        w, h = span(n)
        for c in range(n["col"], n["col"] + w):
            for r in range(n["row"], n["row"] + h):
                if (c, r) in cells:
                    errs.append(f"{n['id']} overlaps {cells[(c, r)]} at column {c}, row {r}")
                cells[(c, r)] = n["id"]
    for e in spec["edges"]:
        errs += [f"edge {e['from']} -> {e['to']}: unknown id {i}" for i in (e["from"], e["to"]) if i not in spec["_by_id"]]
    for g in spec["groups"]:
        errs += [f"group {g['label']}: unknown id {i}" for i in g["nodes"] if i not in spec["_by_id"]]
    for key in ("from", "to"):
        errs += [f"main_path.{key}: unknown id {i}" for i in (ends(spec["main_path"][key]) if spec.get("main_path") else []) if i not in spec["_by_id"]]
    if spec.get("main_path") and not main_path(spec):
        errs.append("main_path: no path from the start to the end")
    refs = []
    for n in spec["nodes"]:
        for ref in n.get("code", []):
            path, _, sym = ref.partition("::")
            full = os.path.join(spec["_root"], path)
            if not os.path.exists(full):
                errs.append(f"{n['id']}: {path} does not exist")
            elif sym and (os.path.isdir(full) or not has_symbol(open(full, encoding="utf-8", errors="replace").read(), sym)):
                errs.append(f"{n['id']}: {path} has no {sym}")
            refs.append(path.rstrip("/"))
    nd = spec.get("not_drawn", {})
    errs += [f"not_drawn: {p} does not exist" for p in nd if not os.path.exists(os.path.join(spec["_root"], p))]
    for f in covered_files(spec):
        if not any(f == r or f.startswith(r + "/") for r in refs) and not any(f == p.rstrip("/") or f.startswith(p.rstrip("/") + "/") for p in nd):
            errs.append(f"{f}: in no box and not in not_drawn")
    return errs


def outputs(spec):
    return {"svg": render_svg(spec), "md": render_md(spec)}


def check(path):
    spec = load(path)
    spec["_stem"] = os.path.splitext(os.path.basename(path))[0]
    errs = problems(spec)
    if not errs:
        for ext, text in outputs(spec).items():
            f = os.path.join(spec["_dir"], f"{spec['_stem']}.{ext}")
            if not os.path.exists(f) or open(f, encoding="utf-8", newline="").read() != text:
                errs.append(f"{spec['_stem']}.{ext} is missing or stale: run `archdraw.py render {path}`")
    return spec, errs


def render(path):
    spec = load(path)
    spec["_stem"] = os.path.splitext(os.path.basename(path))[0]
    errs = problems(spec)
    if errs:
        return spec, errs
    for ext, text in outputs(spec).items():
        with open(os.path.join(spec["_dir"], f"{spec['_stem']}.{ext}"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return spec, []


# ------------------------------------------------------------------------------------------------ self-test
def selftest():
    with tempfile.TemporaryDirectory() as t:
        os.makedirs(f"{t}/code")
        open(f"{t}/code/a.py", "w").write("class Lift:\n    def forward(self): pass\n\ndef decode(): pass\n")
        open(f"{t}/code/b.py", "w").write("def f(): pass\n")
        open(f"{t}/code/c.py", "w").write("x = 1\n")
        spec = {"title": "t", "root": ".", "nodes": [
            {"id": "i", "label": "in", "kind": "input", "col": 0, "row": 0, "code": ["code/b.py"]},
            {"id": "l", "label": "lift", "kind": "learned", "col": 1, "row": 0, "code": ["code/a.py::Lift.forward"]},
            {"id": "r", "label": "decode with a long label that wraps", "kind": "rule", "col": 2, "row": 0, "code": ["code/a.py::decode"],
             "why": "peaks", "replace_with": "soft peak"},
            {"id": "o", "label": "out", "kind": "output", "col": 3, "row": 0}],
            "edges": [{"from": "i", "to": "l"}, {"from": "l", "to": "r", "label": "x"}, {"from": "r", "to": "o"}, {"from": "i", "to": "o", "dashed": True, "via": [[0.5, 1.0], [3.5, 1.0]]}],
            "main_path": {"from": ["i"], "to": ["o"]}, "covers": [{"dir": "code"}], "not_drawn": {"code/c.py": "constants"}}
        p = f"{t}/architecture.json"
        json.dump(spec, open(p, "w"))
        _, errs = render(p)
        assert not errs, errs
        a = analysis(load(p))
        assert a["count"] == {"learned": 1, "fixed": 0, "frozen": 0, "rule": 1, "other": 2} and a["stages"] == 2, a
        assert "Q" in render_svg(load(p)) and "L" in render_svg(load(p))                            # the routed arrow has rounded corners
        assert "1 rule step" in badge(load(p)) and "soft peak" in open(f"{t}/architecture.md").read()
        assert check(p)[1] == [], check(p)[1]
        assert render_svg(load(p)) == render_svg(load(p)) and open(f"{t}/architecture.svg").read().startswith("<svg")
        open(f"{t}/architecture.svg", "a").write("<!-- edited -->")                       # a hand-edited drawing is stale
        assert any("stale" in e for e in check(p)[1])
        render(p)
        bad = json.loads(json.dumps(spec))
        bad["nodes"][1]["code"] = ["code/a.py::Missing"]                                  # a box whose symbol is gone
        bad["nodes"][3]["code"] = ["code/nope.py"]
        bad["not_drawn"] = {}                                                             # c.py in no box
        bad["nodes"].append({"id": "o2", "label": "x", "kind": "output", "col": 3, "row": 0})
        json.dump(bad, open(p, "w"))
        errs = " | ".join(check(p)[1])
        for want in ("has no Missing", "nope.py does not exist", "c.py: in no box", "overlaps o"):
            assert want in errs, (want, errs)
    print("archdraw selftest ok: render (svg + md deterministic), main-path count, symbol / path / coverage / overlap / stale checks")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["render", "check"])
    ap.add_argument("spec")
    a = ap.parse_args()
    spec, errs = (render if a.cmd == "render" else check)(a.spec)
    for e in errs:
        print("ERROR:", e)
    if not errs:
        print(f"{a.cmd} ok: {a.spec} ({len(spec['nodes'])} boxes, {len(spec['edges'])} arrows{', ' + badge(spec) if badge(spec) else ''})")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
