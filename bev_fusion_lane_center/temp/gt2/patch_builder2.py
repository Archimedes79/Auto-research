p = "temp/jr/build_junction_report.py"
s = open(p, encoding="utf-8").read()
marker = "exec(open(os.path.join(HERE, \"report_text.py\"), encoding=\"utf-8\").read())"
assert marker in s
new_funcs = '''def common_table():
    """the pair heads on the SAME ends: the output of temp/gt2/pair_diag_common.py (candidates whose two ends are GT ends that all three systems found, strict one-to-one matching)"""
    import re
    rows = []
    for s in (0, 1):
        p = os.path.join(HERE, "..", "gt2", f"common_s{s}.log")
        if not os.path.exists(p):
            continue
        txt = open(p, encoding="utf-8").read()
        n_common = re.search(r"with both ends found by all \d+ systems: (\d+)", txt)
        for m in re.finditer(r"^(\S+)\s+candidates\s+(\d+), true\s+(\d+), AUC ([\d.]+), at logit >= 0: recall ([\d.]+) precision ([\d.]+); precision at the recall of \S+ \(([\d.]+)\): ([\d.]+)", txt, re.M):
            name = {"e284": "E284 end head", "e287": "E287 p2", "e308": "E308 (p2 + node tokens)"}[m.group(1)[:4]]
            rows.append([f"{name}, seed {s}", n_common.group(1) if n_common else "-", m.group(2), f3(float(m.group(4))), f3(float(m.group(5))), f3(float(m.group(6))), f3(float(m.group(8)))])
    return tbl(["head", "GT pairs with both ends found by all", "candidates", "AUC", "recall at p >= 0.5", "precision at p >= 0.5", "precision at the recall of the E284 head"], rows, num=range(1, 7))


def thr_table():
    """E309: the delivery threshold; every number is a difference to the plain delivered graph of the same seed (the head's own ends_card at 0.5 for the first row of a run)"""
    rows = []
    keys = ("lanes_found", "junction_F1", "junction_R", "dup_frac", "F1@1.0")
    for run in sorted({k[0] for k in THR}):
        for thr in sorted({k[1] for k in THR if k[0] == run}, reverse=True):
            d = THR[(run, thr)]
            n, b = d["numbers"]["head"], d["numbers"]["base"]
            rows.append([run, f"{thr:.1f}", str(d["curves"]), f3(n["topo_F1"]), sg(n["topo_F1"] - b["topo_F1"])] + [sg(n[k] - b[k]) for k in keys])
    if not rows:
        return "<p>pending</p>"
    return tbl(["run", "delivery threshold", "curves", "topo F1", "vs plain graph", "lanes found", "junction F1", "junction recall", "dup_frac", "F1@1.0"], rows, num=range(1, 10))


'''
s = s.replace(marker, new_funcs + marker, 1)
old = "{split_table()}{PAIRS2}</section>"
assert old in s
s = s.replace(old, "{split_table()}{PAIRS2}<h3>The same ends, the same pairs: the E284 head against the new heads</h3>{common_table()}{PAIRS3}<h3>E309: the delivery threshold of the pairs (differences to the plain delivered graph)</h3>{thr_table()}{PAIRS4}</section>", 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
