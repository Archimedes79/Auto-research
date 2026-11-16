p='temp/e273/shorten_report.py'
s=open(p,encoding='utf-8').read()
old='''    ends = [x for x in (h.find("<h3", i + 3), h.find("</section>", i)) if x != -1]
    return h[:i] + h[min(ends):]
'''
new='''    ends = [x for x in (h.find("<h3", i + 3), h.find("</section>", i)) if x != -1]
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
'''
assert old in s
s=s.replace(old,new)
open(p,'w',encoding='utf-8').write(s)
