p = "../autoresearch/evalsuite/comparison/mutation_check.py"
s = open(p, encoding="utf-8").read()
old = '''    ("maps.py", "if dx * dx + dy * dy <= radius_cells * radius_cells + 1e-9:", "if dx * dx + dy * dy < radius_cells * radius_cells - 1e-9:", "dilate: disc open"),'''
new = '''    ("maps.py", "<= radius_cells * radius_cells + 1e-9", "< radius_cells * radius_cells - 1e-9", "dilate: disc open"),
    ("maps.py", "cv2.dilate(np.ascontiguousarray(m, np.uint8), _disc(radius_cells).astype(np.uint8))", "cv2.dilate(np.ascontiguousarray(m, np.uint8), np.ones((3, 3), np.uint8))",
     "dilate: OpenCV with a square instead of the disc"),
    ("maps.py", "cols += [(p & near_g).ravel(), (p & ~near_g).ravel(), (g & near_p).ravel(), (g & ~near_p).ravel()]",
     "cols += [(p & near_p).ravel(), (p & ~near_p).ravel(), (g & near_g).ravel(), (g & ~near_g).ravel()]", "tolerant_counts_many: dilations swapped"),
    ("maps.py", "return out.reshape(len(valids), len(nears), 4).transpose(1, 0, 2)", "return out.reshape(len(nears), len(valids), 4)", "tolerant_counts_many: tolerance and selection axes mixed"),
    ("maps.py", "astype(np.float32)\n    if p.size", "astype(np.float16)\n    if p.size", "tolerant_counts_many: counts in half precision"),'''
assert old in s
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
