import re
p = "temp/e276/exp_oracle_links.py"
s = open(p, encoding="utf-8").read()
s = s.replace("""    cat = np.full(len(p), 2, int)
    for i, (s_, t_) in enumerate(zip(d["link_src"], d["link_dst"])):
        a, b = local[s_], local[t_]
        if a < 0 or b < 0 or g_of_p[a] < 0 or g_of_p[b] < 0:
            continue
""", """    cat = np.full(len(p), 2, int)
    for i, (s_, t_) in enumerate(zip(d["link_src"], d["link_dst"])):
        a, b = local[s_], local[t_]
        if a < 0 or b < 0:
            cat[i] = 3                                                    # an end below e 0.5: not delivered (the read-out drops the link anyway)
            continue
        if g_of_p[a] < 0 or g_of_p[b] < 0:
            continue
""")
s = s.replace("cats = np.zeros(3)", "cats = np.zeros(4)")
s = s.replace("    for k in range(3):\n        cats[k] += int(((cat == k) & on).sum())", "    for k in range(4):\n        cats[k] += int(((cat == k) & on).sum())")
s = s.replace('VARIANTS = {"B": (0, 1, 2), "only right": (0,), "no wrong": (0, 2), "no false-end": (0, 1)}', 'VARIANTS = {"B": (0, 1, 2, 3), "only right": (0,), "no wrong": (0, 2, 3), "no unmatched-end": (0, 1, 3)}')
s = s.replace('print(f"links p >= 0.5: right {int(cats[0])}, wrong between matched nodes {int(cats[1])}, at a false node {int(cats[2])}")', 'print(f"links p >= 0.5: right {int(cats[0])}, wrong between matched nodes {int(cats[1])}, an end delivered but unmatched {int(cats[2])}, an end not delivered (e < 0.5, dropped by the read-out) {int(cats[3])}")')
s = s.replace('link_categories=dict(right=int(cats[0]), wrong_matched=int(cats[1]), false_end=int(cats[2]))', 'link_categories=dict(right=int(cats[0]), wrong_matched=int(cats[1]), unmatched_end=int(cats[2]), not_delivered_end=int(cats[3]))')
s = s.replace("  right (0), wrong between two matched nodes (1), or has an unmatched end (2, at a false node).", "  right (0), wrong between two matched nodes (1), an end delivered but unmatched (2: a false node, or a surplus node of a denser row: the one-to-one match leaves one of two close nodes unmatched), an end below e 0.5 (3).")
s = s.replace("  no false-end drop category 2 (the links at false nodes), keep 0 and 1.", "  no unmatched-end drop category 2, keep 0, 1 and 3.")
open(p, "w", encoding="utf-8").write(s)
print("patched")
