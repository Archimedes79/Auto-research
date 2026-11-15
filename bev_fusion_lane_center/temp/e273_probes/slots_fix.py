p = "temp/slots_probe.py"
s = open(p).read()
s = s.replace("        for p in g[\"pts\"]:\n            p = np.asarray(p, float)", "        for pid, p in enumerate(g[\"pts\"]):\n            p = np.asarray(p, float)")
s = s.replace("cells.setdefault(key, []).append((x, h))", "cells.setdefault(key, {}).setdefault(pid, []).append((x, h))")
s = s.replace("""            for x, h in sorted(lst, key=lambda e: np.linalg.norm(e[0] - c)):""", """            near = [min(pl, key=lambda e: np.linalg.norm(e[0] - c)) for pl in lst.values()]   # closest point of each piece to the cell centre
            for x, h in sorted(near, key=lambda e: np.linalg.norm(e[0] - c)):""")
open(p, "w").write(s)
