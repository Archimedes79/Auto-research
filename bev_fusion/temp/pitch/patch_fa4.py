p = "evaluation/bev_fusion_area.py"
s = open(p, encoding="utf-8").read()
old = '''    return [boxes[i] for i in keep], [corner_share(raw[i]["corners"], area) for i in keep]'''
new = '''    return [boxes[i] for i in keep], [corner_share(raw[i]["corners"], area) if "corners" in raw[i] else footprint_share(boxes[i], area) for i in keep]'''
assert old in s
open(p, "w", encoding="utf-8").write(s.replace(old, new))
