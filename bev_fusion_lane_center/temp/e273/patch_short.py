p='temp/e273/shorten_report.py'
s=open(p,encoding='utf-8').read()
old=s[s.index('def drop_h3'):s.index('# 1 whole sections')]
new='''def _innermost_open_div(h, pos):
    stack = []
    for m in re.finditer(r"<div\b[^>]*>|</div>", h[:pos]):
        if m.group(0) == "</div>":
            stack.pop()
        else:
            stack.append(m)
    return stack[-1] if stack else None


def _match_close(h, start):
    depth = 0
    for m in re.finditer(r"<div\b[^>]*>|</div>", h[start:]):
        depth += -1 if m.group(0) == "</div>" else 1
        if depth == 0:
            return start + m.end()
    raise ValueError


def drop_h3(h, title_start, section=None):
    """remove the panel (a div that starts with the h3) or, for a bare h3, the chunk up to the next <h3> / </section>"""
    m = re.search(r"<h3[^>]*>" + re.escape(title_start), h)
    assert m, title_start
    i = m.start()
    d = _innermost_open_div(h, i)
    if d is not None and d.end() == i:
        return h[:d.start()] + h[_match_close(h, d.start()):]
    ends = [x for x in (h.find("<h3", i + 3), h.find("</section>", i)) if x != -1]
    return h[:i] + h[min(ends):]


'''
s=s.replace(old,new)
s=s.replace('for t in ("Read-out thresholds (node e, link p)", "Phase-4 gate after each attempt", "junction F1", "F1@1.0", "topo F1", "The six phases", "Density"):','for t in ("Read-out thresholds (node e, link p)", "Phase-4 gate after each attempt", "The six phases", "Density"):')
open(p,'w',encoding='utf-8').write(s)
