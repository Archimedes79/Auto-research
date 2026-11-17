"""one-off: small inline-SVG chart helpers for the E273 report; colours come from CSS classes / variables of the page (light and dark), text from var(--muted)"""
from html import escape


def _fmt(v, d=3):
    return f"{v:.{d}f}"


def _nice_max(v):
    for m in (0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0):
        if v <= m:
            return m
    return v


def bars(groups, series, values, *, ymax=None, ref=None, ref_label=None, fmt=".2f", width=640, height=280, ylabel=None, legend=True, label_every=True, left=44, bottom=46, cls=("s0", "s1", "s2", "s3")):
    """grouped bars. groups: x labels; series: names; values[s][g] (None = no bar); ref: a dashed horizontal line (value) with its label; classes s0.. colour the series"""
    ns, ng = len(series), len(groups)
    allv = [v for row in values for v in row if v is not None]
    ymax = ymax or _nice_max(max(allv + ([ref] if ref else [0])) * 1.12)
    top, right = 24, 12
    W, H = width, height
    pw, ph = W - left - right, H - top - bottom
    gw = pw / ng
    bw = min(34, gw * 0.8 / ns)
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" xmlns="http://www.w3.org/2000/svg">']
    ticks = 5
    for i in range(ticks + 1):
        v = ymax * i / ticks
        y = top + ph - ph * v / ymax
        out.append(f'<line class="grid" x1="{left}" x2="{W - right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{left - 6}" y="{y + 3.5:.1f}" text-anchor="end">{v:{fmt}}</text>')
    for g, name in enumerate(groups):
        cx = left + gw * (g + 0.5)
        for s in range(ns):
            v = values[s][g]
            if v is None:
                continue
            h = ph * v / ymax
            x = cx - bw * ns / 2 + bw * s
            y = top + ph - h
            out.append(f'<rect class="{cls[s % len(cls)]}" x="{x:.1f}" y="{y:.1f}" width="{bw - 2:.1f}" height="{max(h, 0.5):.1f}" rx="1.5"/>')
            if label_every:
                out.append(f'<text class="val" x="{x + (bw - 2) / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle">{v:{fmt}}</text>')
        out.append(f'<text class="xl" x="{cx:.1f}" y="{H - bottom + 16}" text-anchor="middle">{escape(str(name))}</text>')
    if ref is not None:
        y = top + ph - ph * ref / ymax
        out.append(f'<line class="ref" x1="{left}" x2="{W - right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        if ref_label:
            out.append(f'<text class="reflab" x="{W - right - 2}" y="{y - 5:.1f}" text-anchor="end">{escape(ref_label)}</text>')
    if ylabel:
        out.append(f'<text class="tick" x="{left}" y="12" text-anchor="start">{escape(ylabel)}</text>')
    if legend and ns > 1:
        lx = left
        for s, name in enumerate(series):
            out.append(f'<rect class="{cls[s % len(cls)]}" x="{lx}" y="{H - 14}" width="10" height="10" rx="2"/><text class="leg" x="{lx + 14}" y="{H - 5}">{escape(name)}</text>')
            lx += 14 + 7 * len(name) + 18
    out.append("</svg>")
    return "".join(out)


def lines(xs, series, values, *, ymax=1.0, ymin=0.0, ref=None, ref_label=None, mark=None, mark_label=None, width=640, height=300, xfmt="{:.0f} %", yfmt=".2f", left=44, cls=("s0", "s1", "s2")):
    """line chart; xs numeric (positions) with labels via xfmt; mark = (x, y) a highlighted operating point with a label"""
    top, right, bottom = 24, 16, 52
    W, H = width, height
    pw, ph = W - left - right, H - top - bottom
    x0, x1 = min(xs), max(xs)
    px = lambda x: left + pw * (x - x0) / (x1 - x0)  # noqa: E731
    py = lambda y: top + ph - ph * (y - ymin) / (ymax - ymin)  # noqa: E731
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" xmlns="http://www.w3.org/2000/svg">']
    for i in range(6):
        v = ymin + (ymax - ymin) * i / 5
        out.append(f'<line class="grid" x1="{left}" x2="{W - right}" y1="{py(v):.1f}" y2="{py(v):.1f}"/><text class="tick" x="{left - 6}" y="{py(v) + 3.5:.1f}" text-anchor="end">{v:{yfmt}}</text>')
    for x in xs:
        out.append(f'<text class="xl" x="{px(x):.1f}" y="{H - bottom + 16}" text-anchor="middle">{xfmt.format(x)}</text>')
    if ref is not None:
        out.append(f'<line class="ref" x1="{left}" x2="{W - right}" y1="{py(ref):.1f}" y2="{py(ref):.1f}"/>')
        if ref_label:
            out.append(f'<text class="reflab" x="{W - right - 2}" y="{py(ref) - 5:.1f}" text-anchor="end">{escape(ref_label)}</text>')
    for s, name in enumerate(series):
        pts = " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in zip(xs, values[s]) if v is not None)
        out.append(f'<polyline class="ln {cls[s]}" points="{pts}" fill="none"/>')
        for x, v in zip(xs, values[s]):
            if v is not None:
                out.append(f'<circle class="{cls[s]}" cx="{px(x):.1f}" cy="{py(v):.1f}" r="3.2"/>')
    if mark:
        out.append(f'<circle class="mk" cx="{px(mark[0]):.1f}" cy="{py(mark[1]):.1f}" r="6" fill="none"/>')
        if mark_label:
            out.append(f'<text class="reflab" x="{px(mark[0]) + 10:.1f}" y="{py(mark[1]) + 4:.1f}">{escape(mark_label)}</text>')
    lx = left
    for s, name in enumerate(series):
        out.append(f'<rect class="{cls[s]}" x="{lx}" y="{H - 14}" width="10" height="10" rx="2"/><text class="leg" x="{lx + 14}" y="{H - 5}">{escape(name)}</text>')
        lx += 14 + 7 * len(name) + 18
    out.append("</svg>")
    return "".join(out)


def stacked(parts, *, total_label="", width=640, height=96, fmt=".1f", unit=" ms", cls=("s0", "s1", "s2", "s3", "s4")):
    """one horizontal stacked bar: parts = [(name, value)]"""
    left, right = 8, 8
    W, H = width, height
    tot = sum(v for _, v in parts)
    pw = W - left - right
    x = left
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" xmlns="http://www.w3.org/2000/svg">']
    for i, (name, v) in enumerate(parts):
        w = pw * v / tot
        out.append(f'<rect class="{cls[i % len(cls)]}" x="{x:.1f}" y="14" width="{w - 1.5:.1f}" height="34" rx="2"/>')
        out.append(f'<text class="barlab" x="{x + w / 2:.1f}" y="35" text-anchor="middle">{v:{fmt}}</text>')
        out.append(f'<text class="xl" x="{x + w / 2:.1f}" y="66" text-anchor="middle">{escape(name)}</text>')
        x += w
    out.append(f'<text class="tick" x="{left}" y="{H - 6}">{escape(total_label)} {tot:{fmt}}{unit}</text>')
    out.append("</svg>")
    return "".join(out)
