p = "evaluation/jprune_b.py"
s = open(p, encoding="utf-8").read()
old = '''    return dict(nodes={k: np.asarray(d[k])[order] for k in ("e", "pos", "dirv", "flags", "s") if d.get(k) is not None}, section=rank[np.asarray(d["section"])[order]],'''
new = '''    sec = np.asarray(d["section"])                                                                       # `dump_full` keeps the section start as an index of the UNFILTERED node list (`idx` holds the kept ones): -1 when the start is not a kept node
    where = -np.ones(int(max(d["idx"].max(initial=0), sec.max(initial=0))) + 1, int)
    where[d["idx"]] = np.arange(len(d["idx"]))
    sec = np.where(where[sec] >= 0, rank[where[sec]], -1)
    return dict(nodes={k: np.asarray(d[k])[order] for k in ("e", "pos", "dirv", "flags", "s") if d.get(k) is not None}, section=sec[order],'''
assert old in s
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
