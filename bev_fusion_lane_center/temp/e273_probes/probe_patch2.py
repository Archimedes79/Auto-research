p = "temp/query_probe.py"
s = open(p, encoding="utf-8").read()
# events of suppressed connectors are kept: window ends AND dead ends (no predecessor / no successor)
s = s.replace("""        if suppress and kd == "connector":
            if not (keep_open and g is not None and (g["start_open"][pi] or g["end_open"][pi])):
                continue""", """        if suppress and kd == "connector":
            if not (keep_open and g is not None):
                continue""")
s = s.replace("""            if suppress and kd == "connector" and not ((k == 0 and g["start_open"][pi]) or (k == len(p) - 1 and g["end_open"][pi])):
                continue""", """            if suppress and kd == "connector":
                has_pred = any(j == pi for _, j in g["edges"])
                has_succ = any(i == pi for i, _ in g["edges"])
                is_ev = (k == 0 and (g["start_open"][pi] or not has_pred)) or (k == len(p) - 1 and (g["end_open"][pi] or not has_succ))
                if not is_ev:
                    continue""")
open(p, "w", encoding="utf-8").write(s)
