p = "evaluation/bev_fusion_area.py"
s = open(p, encoding="utf-8").read()
old = '''    n1, n2 = max(int(math.ceil(np.hypot(*e1) / STEP)), 3), max(int(math.ceil(np.hypot(*e2) / STEP)), 3)'''
new = '''    n1, n2 = max(int(math.ceil(np.hypot(*e1) / CORNER_STEP)), 12), max(int(math.ceil(np.hypot(*e2) / CORNER_STEP)), 12)'''
assert old in s
s = s.replace(old, new)
s = s.replace('''STEP = 0.25                                       # m: footprint sampling''', '''STEP = 0.25                                       # m: footprint sampling of a detection
CORNER_STEP = 0.125                               # m: sampling of a GT box (4 x 4 per cell, at least 12 per axis): the same as autogt.coverage, so the shares equal the stored `share_fusion`''')
open(p, "w", encoding="utf-8").write(s)
p = "evaluation/bev_response.py"
s = open(p, encoding="utf-8").read()
old = '''            if b["group"] == CLASS and "length" in b and in_front(b):'''
new = '''            if b["group"] == CLASS and "length" in b and in_front(b) and rb.get("observed", True):             # (an unobserved box may have no pixels in the layer)'''
assert old in s
s = s.replace(old, new)
s = s.replace("vehicle footprints against the cached vehicle layer as the area", "observed vehicle footprints against the cached vehicle layer as the area")
open(p, "w", encoding="utf-8").write(s)
