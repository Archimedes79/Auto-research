"""Make a shorter copy of results/E273.report in results/E273.report_short (the original stays untouched):
drop the GT section, the plan recap, the threshold table, the failed-attempt gate charts, three of six scenes, the seed-1 and 'earlier work' videos;
copy only the images and videos the shortened page still uses."""
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "results", "E273.report"))
DST = SRC + "_short"

h = open(os.path.join(SRC, "index.html"), encoding="utf-8").read()
n0 = len(h)


def drop_section(h, sid):
    i = h.index(f'<section id="{sid}"')
    j = h.index("</section>", i) + len("</section>")
    return h[:i] + h[j:]


def _innermost_open_div(h, pos):
    stack = []
    for m in re.finditer(r"<div[^>]*>|</div>", h[:pos]):
        if m.group(0) == "</div>":
            if stack:
                stack.pop()
        else:
            stack.append(m)
    return stack[-1] if stack else None


def _match_close(h, start):
    depth = 0
    for m in re.finditer(r"<div[^>]*>|</div>", h[start:]):
        depth += -1 if m.group(0) == "</div>" else 1
        if depth == 0:
            return start + m.end()
    raise ValueError


def bal(h):
    return h.count("<div"), h.count("</div>")


def drop_h3(h, title_start, section=None):
    print(title_start[:30], bal(h), end=" -> ")
    r = _drop_h3(h, title_start)
    print(bal(r))
    return r


def _drop_h3(h, title_start):
    """remove the panel (a div that starts with the h3) or, for a bare h3, the chunk up to the next <h3> / </section>"""
    m = re.search(r"<h3[^>]*>" + re.escape(title_start), h)
    assert m, title_start
    i = m.start()
    d = _innermost_open_div(h, i)
    if d is not None and d.end() == i:
        return h[:d.start()] + h[_match_close(h, d.start()):]
    ends = [x for x in (h.find("<h3", i + 3), h.find("</section>", i)) if x != -1]
    return _cut_balanced(h, i, min(ends))


def _cut_balanced(h, i, j):
    """cut h[i:j] but never a closing div of a wrapper opened before i, and take along the closing div of a div opened inside the cut"""
    depth = 0
    for m in re.finditer(r"<div[^>]*>|</div>", h[i:j]):
        depth += -1 if m.group(0) == "</div>" else 1
        if depth < 0:
            j = i + m.start()
            break
    depth = 0
    for m in re.finditer(r"<div[^>]*>|</div>", h[i:j]):
        depth += -1 if m.group(0) == "</div>" else 1
    while depth > 0:
        m = re.compile(r"<div[^>]*>|</div>").search(h, j)
        depth += -1 if m.group(0) == "</div>" else 1
        j = m.end()
    return h[:i] + h[j:]


# 1 whole sections
print("start", bal(h))
h = drop_section(h, "gt")
print("gt", bal(h))
h = h.replace('<a href="#gt">GT work</a>', "")

# 2 plan recap
for t in ("The original plan, recapitulated", "Your brief of 2026-11-04", "E273, the eight steps of the card", "The continuation (E274) and what comes after"):
    try:
        h = drop_h3(h, t)
    except AssertionError:
        print("already gone:", t)
h = h.replace("<h2>Status board and the original plan</h2>", "<h2>Status board</h2>").replace(">Status and plan<", ">Status<")

# 3 threshold table, failed-attempt gate charts, phase table, duplicate density chunk
for t in ("Read-out thresholds (node e, link p)", "Phase-4 gate after each attempt", "The six phases", "Density"):
    try:
        h = drop_h3(h, t)
    except AssertionError:
        print("not found:", t)

# 4 scenes and videos
for sc in ("0096", "0017", "0015"):
    m = re.search(r'<div class="scene"><h3>scene-' + sc, h)
    i = m.start()
    j = h.index("</div>", i) + len("</div>")
    # scene divs hold only figures: the first </div> closes the scene (figures do not use div)
    h = h[:i] + h[j:]
h = re.sub(r"<figure><video[^>]*src=\"vid/scene-\d+_s1\.webm\"></video><p class=\"cap\">[^<]*</p></figure>", "", h)
h = drop_h3(h, "Earlier work")
h = h.replace("<h3>scene-0221</h3>", "<h3>scene-0221</h3>", 1)

open(os.path.join(HERE, "short_index.html"), "w", encoding="utf-8").write(h)

# 5 copy what is used
os.makedirs(os.path.join(DST, "img"), exist_ok=True)
os.makedirs(os.path.join(DST, "vid"), exist_ok=True)
used = set(re.findall(r'(?:src|href|poster)="((?:img|vid)/[^"]+)"', h))
tot = 0
for u in sorted(used):
    s = os.path.join(SRC, u)
    if not os.path.exists(s):
        print("missing:", u)
        continue
    shutil.copy2(s, os.path.join(DST, u))
    tot += os.path.getsize(s)
open(os.path.join(DST, "index.html"), "w", encoding="utf-8").write(h)
print(f"page {n0} -> {len(h)} bytes; {len(used)} files, {tot / 1048576:.1f} MB; div balance {h.count('<div')} / {h.count('</div>')}; sections {h.count('<section')} / {h.count('</section>')}")
