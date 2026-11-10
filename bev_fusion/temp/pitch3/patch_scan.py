s = open("bev_fusion/temp/pitch3/scan.py", encoding="utf-8").read()
old_start = s.index("def uses(")
old_end = s.index("def scan(")
new = '''WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
COUNTS = {f: __import__("collections").Counter(WORD.findall(t)) for f, t in BLOB.items()}


def uses(name, own_file, own_line_span):
    out = {}
    for f, c in COUNTS.items():
        n = c.get(name, 0) - (1 if f == own_file else 0)
        if n > 0:
            out[f] = n
    return out


'''
open("bev_fusion/temp/pitch3/scan.py", "w", encoding="utf-8").write(s[:old_start] + new + s[old_end:])
