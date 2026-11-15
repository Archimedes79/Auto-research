"""Helpers of the result-report builder: readers of the stored evaluation files, html pieces, charts, the text bank.
No section logic and no run names: what a run is called and which files it has comes from report_runs.json (build_report.py).
A file that the config names must exist: `need` stops the build with a message instead of leaving an empty cell."""
import base64
import html
import io
import json
import os
import re


def fail(msg):
    raise SystemExit(f"report build FAILED: {msg}")


def need(path, what):
    """the file a config entry names; a missing one stops the build"""
    if not os.path.isfile(path):
        fail(f"{what}: {path} does not exist")
    return path


def esc(s):
    return html.escape(str(s))


def md_inline(s):
    """**bold** and `code` of a markdown cell -> html (everything else escaped)"""
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


# ------------------------------------------------------------------------------------------------ stored evaluation files
def read_table(path):
    """the stored evaluation table (one `key value` line per metric) -> {key: value}; an unreadable or empty file stops the build"""
    need(path, "evaluation table")
    d = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^(\S+)\s+(-?[0-9.]+(?:e-?[0-9]+)?|nan)\s*$", line.rstrip())
        if m:
            d[m.group(1)] = float(m.group(2))
    if not d:
        fail(f"{path} holds no `key value` line")
    return d


def read_metrics(path):
    """all numbers of a metrics_n<N>.json (range bins, tolerances ...): {key: value}"""
    need(path, "metrics file")
    d = json.load(open(path, encoding="utf-8")).get("metrics", {})
    return {k: v for k, v in d.items() if isinstance(v, (int, float))}


def read_frames(path):
    """per-frame scores of the validation frames: [(label, scores, support)]"""
    need(path, "per-frame scores")
    out = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line.replace("NaN", "null"))
        out.append((d["label"], d["scores"], d["support"]))
    return out


def read_paired(path):
    """the output of `bev_suite.py compare` -> (header line, {key: (a, b, b-a, lo, hi, excludes 0)}); the numbers stay the strings of the file"""
    need(path, "paired comparison")
    head, rows = "", {}
    for ln in open(path, encoding="utf-8"):
        ln = ln.rstrip("\n")
        if ln.startswith("paired scene bootstrap"):
            head = ln
            continue
        m = re.match(r"^(\S+)\s+a\s+(-?[0-9.]+)\s+b\s+(-?[0-9.]+)\s+b-a\s+([+-][0-9.]+)\s+95% \[([+-][0-9.]+), ([+-][0-9.]+)\]\s*(\(excludes 0\))?", ln)
        if m:
            rows[m.group(1)] = m.groups()[1:6] + (bool(m.group(7)),)
    if not rows:
        fail(f"{path} holds no paired row (is it the output of `bev_suite.py compare`?)")
    return head, rows


def paired_table(path, a_label, b_label):
    """a `compare` output -> html table (metric, a, b, b-a, 95 % interval, excludes 0)"""
    head, rows = read_paired(path)
    out = [f'<p class=cap>{esc(head)}</p><table><tr><th>metric (key of the evaluation)</th><th>{esc(a_label)}</th><th>{esc(b_label)}</th><th>b - a</th><th>95 % interval</th><th></th></tr>']
    for k, (a, b, d, lo, hi, ex) in rows.items():
        out.append(f"<tr><td><code>{esc(k)}</code></td><td>{a}</td><td>{b}</td><td>{d}</td><td>[{lo}, {hi}]</td><td>{'excludes 0' if ex else ''}</td></tr>")
    return "".join(out) + "</table>"


def compare_key(table_key):
    """the key of a `compare` row for a key of the evaluation table: seg_fov/seg/dt+0.00/x -> seg.fov/dt+0.00/x, detf/dt+0.00/x -> detf.dt+0.00/x"""
    m = re.match(r"^(\w+?)_(\w+)/seg/(.*)$", table_key)
    return f"{m.group(1)}.{m.group(2)}/{m.group(3)}" if m else table_key.replace("/", ".", 1)


# ------------------------------------------------------------------------------------------------ markdown pieces
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
    out, on = [], False
    for ln in text.split("\n"):
        if ln.startswith(start_marker):
            on = True
            continue
        if on and (ln.startswith(end_marker) or (ln.startswith("#") and not ln.startswith(start_marker))):
            break
        if on and ln.startswith("- "):
            out.append(ln[2:])
    return out


def md_table_html(lines):
    rows = [[c.strip() for c in ln.strip()[1:-1].split("|")] for ln in lines if not re.match(r"^\|[\s:|-]+\|\s*$", ln)]
    if not rows:
        return ""
    return "<table class=wide><tr>" + "".join(f"<th class=l>{md_inline(c)}</th>" for c in rows[0]) + "</tr>" + "".join(
        "<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:]) + "</table>"


def render_md(text, bullets=False):
    """a small markdown text (headings, pipe tables, paragraphs; with bullets=True also '- ' lists) -> html. A result file of the evaluation tools keeps bullets=False"""
    out, buf, items = [], [], []

    def flush():
        nonlocal buf, items
        if buf:
            out.append(md_table_html(buf))
        if items:
            out.append("<ul>" + "".join(f"<li>{md_inline(i)}</li>" for i in items) + "</ul>")
        buf, items = [], []

    for ln in text.split("\n"):
        if ln.startswith("|"):
            buf.append(ln)
            continue
        if bullets and ln.startswith("- "):
            items.append(ln[2:])
            continue
        flush()
        if ln.startswith("#"):
            out.append(f"<h4>{md_inline(ln.lstrip('# '))}</h4>")
        elif ln.strip():
            out.append(f"<p>{md_inline(ln)}</p>")
    flush()
    return "".join(out)


# ------------------------------------------------------------------------------------------------ text bank and markers
def read_bank(path):
    """a text bank: blocks that start with a line `@@@ name`; the block is the text up to the next such line, without the one blank line that separates the blocks
    (so a block keeps its own leading and trailing newlines)"""
    need(path, "text bank")
    out, name, buf = {}, None, []

    def close():
        if buf and buf[-1] == "":
            buf.pop()
        out[name] = "\n".join(buf)

    for ln in open(path, encoding="utf-8").read().split("\n"):
        if ln.startswith("@@@ "):
            if name is not None:
                close()
            name, buf = ln[4:].strip(), []
        elif name is not None:
            buf.append(ln)
    if name is not None:
        close()
    return out


def fill(text, fact, what="text"):
    """replace every @@name@@ by fact(name) (a function: the facts are computed only when a text asks for them); an unknown name stops the build"""
    def one(m):
        v = fact(m.group(1))
        if v is None:
            fail(f"{what}: no fact named @@{m.group(1)}@@")
        return str(v)
    return re.sub(r"@@(\w+)@@", one, text)


# ------------------------------------------------------------------------------------------------ html pieces
def table_html(header, rows, cls="", left=()):
    h = "".join(f"<th class=l>{esc(c)}</th>" if i in left else f"<th>{esc(c)}</th>" for i, c in enumerate(header))
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table class='{cls}'><tr>{h}</tr>{b}</table>"


def rows_table(spec, tabs, labels, fmt="{:.3f}"):
    """spec = [(name, key, lower is better)]; a value that a run lacks is `-`; the small number is the difference to the first column"""
    out = ["<table><tr><th>metric</th>" + "".join(f"<th>{esc(l)}</th>" for l in labels) + "</tr>"]
    for name, key, lower in spec:
        vals = [t.get(key) for t in tabs]
        cells = []
        for i, v in enumerate(vals):
            if v is None:
                cells.append("<td>-</td>")
                continue
            c = fmt.format(v)
            if i > 0 and vals[0] is not None:
                d = v - vals[0]
                good = d < 0 if lower else d > 0
                cls = "" if abs(d) < 0.005 else (" good" if good else " bad")
                c += f' <span class="delta{cls}">{d:+.3f}</span>'
            cells.append(f"<td>{c}</td>")
        out.append(f"<tr><td>{esc(name)}</td>" + "".join(cells) + "</tr>")
    return "\n".join(out) + "</table>"


def img(path, w=1100):
    """a picture as an embedded jpeg (the page is one file); needs PIL"""
    from PIL import Image
    if not os.path.exists(path):
        return ""
    im = Image.open(path).convert("RGB")
    if im.width > w:
        im = im.resize((w, int(im.height * w / im.width)))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=82)
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode()}" alt="{esc(os.path.basename(path))}">'


def html_to_text(s):
    """a page as plain text for proofreading (tags stripped, block tags -> newlines)"""
    s = re.sub(r"<style.*?</style>", "", s, flags=re.S)
    s = re.sub(r"<svg.*?</svg>", "[chart]", s, flags=re.S)
    s = re.sub(r"<img[^>]*>", "[image]", s)
    s = re.sub(r"<video[^>]*>.*?</video>", "[video]", s, flags=re.S)
    s = re.sub(r"</(p|div|h\d|li|tr|table|ul|details|summary|figure)>", "\n", s)
    s = re.sub(r"</t[dh]>", " | ", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    return re.sub(r"\n\s*\n+", "\n", s)


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
