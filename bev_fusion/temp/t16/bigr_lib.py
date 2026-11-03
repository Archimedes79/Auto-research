"""scratch of the big t16 report: helpers (markdown digest parser, tiny inline-markdown converter, SVG charts, stored-table readers).
Everything here reads stored files; no number is typed except the ladder values that are copied from the digest (experiment cards) and named as such."""
import html
import json
import math
import os
import re

ROOT = "bev_fusion/runs"


def esc(s):
    return html.escape(str(s))


def md_inline(s):
    """**bold** and `code` of a markdown cell -> html (everything else escaped)"""
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def read_table(run, n=1000, tag=""):
    """the stored evaluation table of a run: {key: value}"""
    d = {}
    p = f"{ROOT}/{run}/eval/by_n/table{tag}_n{n}.txt"
    if tag == "_train" and not os.path.exists(p):
        p = f"{ROOT}/{run}/eval/table_train.txt"
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            m = re.match(r"^(\S+)\s+(-?[0-9.]+(?:e-?[0-9]+)?|nan)\s*$", line.rstrip())
            if m:
                d[m.group(1)] = float(m.group(2))
    return d


def read_metrics(run, n=1000, tag=""):
    """all numbers of metrics_n1000.json (range bins, tolerances, stop line ...): {key: value}"""
    p = f"{ROOT}/{run}/eval/by_n/metrics{tag}_n{n}.json"
    if tag == "_train" and not os.path.exists(p):
        p = f"{ROOT}/{run}/eval/metrics_train.json"
    if not os.path.exists(p):
        return {}
    d = json.load(open(p, encoding="utf-8")).get("metrics", {})
    return {k: v for k, v in d.items() if isinstance(v, (int, float))}


def read_frames(run):
    """per-frame scores of the 1000 val frames: [(label, scores, support)]"""
    p = f"{ROOT}/{run}/eval/suite/quant/eval_curves_fusion_vel_n1000/frames.jsonl"
    out = []
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            d = json.loads(line.replace("NaN", "null"))
            out.append((d["label"], d["scores"], d["support"]))
    return out


# ------------------------------------------------------------------------------------------------ markdown digests
def md_tables(text):
    """all pipe tables of a markdown text -> [(nearest heading above, header cells, [row cells])]"""
    out, heading, lines, i = [], "", text.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            heading = ln.lstrip("# ").strip()
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|\s*$", lines[i + 1]):
            hdr = split_row(ln)
            rows, j = [], i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append(split_row(lines[j]))
                j += 1
            out.append((heading, hdr, rows))
            i = j
            continue
        i += 1
    return out


def split_row(ln):
    return [c.strip() for c in ln.strip()[1:-1].split(" | ")]


def md_bullets(text, start_marker, end_marker):
    """the '- ' bullets between two marker lines"""
    lines, out, on = text.split("\n"), [], False
    for ln in lines:
        if ln.startswith(start_marker):
            on = True
            continue
        if on and (ln.startswith(end_marker) or (ln.startswith("#") and not ln.startswith(start_marker))):
            break
        if on and ln.startswith("- "):
            out.append(ln[2:])
    return out


# ------------------------------------------------------------------------------------------------ charts
def bar_chart(items, title, vmax=None, width=640, row_h=22, label_w=250, fmt="{:.3f}", note=""):
    """horizontal bars: items = [(label, value, css class)]; classes v5 / v8 / old / ours choose the fill (CSS variables of the page)"""
    vmax = vmax or max(v for _, v, _ in items) * 1.1
    h = 30 + row_h * len(items) + (18 if note else 4)
    bw = width - label_w - 70
    s = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="{esc(title)}" class="chart"><text x="0" y="14" class="ct">{esc(title)}</text>']
    for i, (lab, v, cls) in enumerate(items):
        y = 24 + i * row_h
        w = max(1.0, bw * v / vmax)
        s.append(f'<text x="{label_w - 6}" y="{y + 13}" text-anchor="end" class="cl">{esc(lab)}</text>'
                 f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{row_h - 6}" rx="3" class="bar {cls}"/>'
                 f'<text x="{label_w + w + 5:.1f}" y="{y + 13}" class="cv">{fmt.format(v)}</text>')
    if note:
        s.append(f'<text x="0" y="{h - 4}" class="cn">{esc(note)}</text>')
    s.append("</svg>")
    return "".join(s)


def histogram(values, title, bins=10, width=560, height=170, note=""):
    """vertical histogram of values in [0, 1]"""
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, int(v * bins))] += 1
    top = max(counts) or 1
    bw = (width - 50) / bins
    s = [f'<svg viewBox="0 0 {width} {height + 44}" role="img" aria-label="{esc(title)}" class="chart"><text x="0" y="14" class="ct">{esc(title)}</text>']
    for i, c in enumerate(counts):
        hh = (height - 40) * c / top
        x = 40 + i * bw
        s.append(f'<rect x="{x:.1f}" y="{height - hh:.1f}" width="{bw - 3:.1f}" height="{hh:.1f}" rx="2" class="bar ours"/>'
                 f'<text x="{x + (bw - 3) / 2:.1f}" y="{height - hh - 3:.1f}" text-anchor="middle" class="cv">{c}</text>'
                 f'<text x="{x + (bw - 3) / 2:.1f}" y="{height + 14}" text-anchor="middle" class="cl">{i / bins:.1f}-{(i + 1) / bins:.1f}</text>')
    s.append(f'<text x="40" y="{height + 34}" class="cn">{esc(note)}</text></svg>')
    return "".join(s)


def quantiles(v, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    v = sorted(v)
    return [v[min(len(v) - 1, int(q * len(v)))] for q in qs]


def paired_table(path, a_label, b_label):
    """the output of `bev_suite.py compare` -> html table (metric, a, b, b-a, 95 % interval, excludes 0)"""
    if not os.path.exists(path):
        return ""
    rows = []
    head = ""
    for ln in open(path, encoding="utf-8"):
        ln = ln.rstrip("\n")
        if ln.startswith("paired scene bootstrap"):
            head = ln
            continue
        m = re.match(r"^(\S+)\s+a\s+(-?[0-9.]+)\s+b\s+(-?[0-9.]+)\s+b-a\s+([+-][0-9.]+)\s+95% \[([+-][0-9.]+), ([+-][0-9.]+)\]\s*(\(excludes 0\))?", ln)
        if m:
            rows.append(m.groups())
    if not rows:
        return ""
    out = [f'<p class=cap>{esc(head)}</p><table><tr><th>metric (key of the evaluation)</th><th>{esc(a_label)}</th><th>{esc(b_label)}</th><th>b - a</th><th>95 % interval</th><th></th></tr>']
    for k, a, b, d, lo, hi, ex in rows:
        out.append(f"<tr><td><code>{esc(k)}</code></td><td>{a}</td><td>{b}</td><td>{d}</td><td>[{lo}, {hi}]</td><td>{'excludes 0' if ex else ''}</td></tr>")
    return "".join(out) + "</table>"
